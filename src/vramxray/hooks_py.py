"""Pure Python live mode: no compiled code, works on any torch since 2.5.

torch lets Python register an out-of-memory observer. It fires inside the failing allocation,
after the allocator has dropped its lock, so taking a snapshot there is safe. We combine that
snapshot with an NVML read and print the report before the exception reaches the user.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable

from .accounting import Accounting, reconcile
from .frag import Explanation, explain
from .nvml import NVML, Sampler
from .report import Report
from .snapshot import from_dict
from .suggest import suggest


class Watcher:
    def __init__(
        self,
        stacks: str | None = None,
        max_entries: int = 200_000,
        interval_ms: int = 50,
        report_dir: str | None = None,
        on_report: Callable[[Report], None] | None = None,
        quiet: bool = False,
    ) -> None:
        self.stacks = stacks
        self.max_entries = max_entries
        self.interval_ms = interval_ms
        self.report_dir = report_dir or os.getcwd()
        self.on_report = on_report
        self.quiet = quiet
        self.nvml = NVML()
        self.baseline: dict[int, int] = {}
        self.samplers: dict[int, Sampler] = {}
        self.uuids: dict[int, str | None] = {}
        self.reports: list[Report] = []
        self._last: tuple[int, int, float] | None = None
        self._lock = threading.Lock()
        self._installed = False

    def install(self) -> Watcher:
        import torch

        if self._installed:
            return self
        already_up = torch.cuda.is_initialized()
        before = {} if already_up else {d: self.nvml.memory(d) for d in range(_count())}
        torch.cuda.init()
        for d in range(torch.cuda.device_count()):
            if not already_up:
                # init() alone makes no context; the first kernel does, so run one
                torch.empty(1, device=f"cuda:{d}").zero_()
                torch.cuda.synchronize(d)
            self.uuids[d] = _uuid(d)
            m = self.nvml.memory(d, self.uuids[d])
            if m is not None:
                s = Sampler(self.nvml, d, self.uuids[d], self.interval_ms)
                s.start()
                self.samplers[d] = s
            # the context is only measurable as a delta across our own init. If CUDA was already
            # up, fall back to NVML's per-process number, which WSL2 does not provide.
            own = next(
                (p.used for p in self.nvml.processes(d, self.uuids[d]) if p.pid == os.getpid()),
                None,
            )
            if not already_up and m is not None and before.get(d) is not None:
                m = self.nvml.memory(d, self.uuids[d]) or m
                grown = m.used - before[d].used - torch.cuda.memory_reserved(d)
                self.baseline[d] = max(grown, 0)
            elif own is not None:
                self.baseline[d] = max(own - torch.cuda.memory_reserved(d), 0)
            else:
                self.baseline[d] = -1  # unknown
        if self.stacks:
            torch.cuda.memory._record_memory_history(
                enabled="all", stacks=self.stacks, max_entries=self.max_entries
            )
        torch._C._cuda_attach_out_of_memory_observer(self._on_oom)
        self._installed = True
        return self

    def _on_oom(self, device: int, alloc: int, allowed_max: int, device_free: int) -> None:
        # torch frees its cache and retries, so the observer can fire twice for one failure
        now = time.monotonic()
        with self._lock:
            if (
                self._last
                and self._last[0] == device
                and self._last[1] == alloc
                and now - self._last[2] < 2
            ):
                return
            self._last = (device, alloc, now)
        try:
            rep = self.report(
                device, request=alloc, allowed_max=allowed_max, device_free=device_free
            )
        except Exception as e:  # never let the report break the OOM that is already in flight
            print(f"vramxray: could not build report: {e!r}", file=sys.stderr)
            return
        if not self.quiet:
            print(str(rep), file=sys.stderr, flush=True)
        path = os.path.join(self.report_dir, _report_name(rep))
        try:
            rep.write(path)
            if not self.quiet:
                print(f"  full report: {path}", file=sys.stderr, flush=True)
        except OSError:
            pass
        if self.on_report:
            self.on_report(rep)

    def report(
        self,
        device: int | None = None,
        request: int | None = None,
        allowed_max: int | None = None,
        device_free: int | None = None,
    ) -> Report:
        import torch

        if device is None:
            device = torch.cuda.current_device()
        snap = from_dict(torch.cuda.memory._snapshot(device))
        stats = torch.cuda.memory_stats(device)
        reserved = stats.get("reserved_bytes.all.current", 0)
        allocated = stats.get("allocated_bytes.all.current", 0)
        ex: Explanation = explain(snap, device, request=request, device_free=device_free)
        nvml = self.nvml.memory(device, self.uuids.get(device))
        if allowed_max is not None and nvml is not None and allowed_max >= nvml.total:
            allowed_max = None  # torch passes device_total when no fraction is set
        others = [
            p for p in self.nvml.processes(device, self.uuids.get(device)) if p.pid != os.getpid()
        ]
        acc: Accounting = reconcile(
            nvml,
            reserved,
            allocated,
            self.baseline.get(device, 0),
            other_processes=others,
            allowed_max=allowed_max,
        )
        rep = Report(device, ex, acc, suggest(ex, acc), request, _rank(), torch.__version__)
        self.reports.append(rep)
        return rep


def _count() -> int:
    import torch

    return torch.cuda.device_count()


def _uuid(device: int) -> str | None:
    import torch

    try:
        u = torch.cuda.get_device_properties(device).uuid
        return f"GPU-{u}"
    except Exception:
        return None


def _rank() -> int | None:
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        v = os.environ.get(key)
        if v is not None and v.isdigit():
            return int(v)
    return None


def _report_name(rep: Report) -> str:
    rank = f"rank{rep.rank}" if rep.rank is not None else f"pid{os.getpid()}"
    return f"vramxray-oom-{rank}-{time.strftime('%Y%m%dT%H%M%S')}.json"
