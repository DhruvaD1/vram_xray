from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable

from .accounting import Accounting, reconcile
from .frag import Explanation, explain
from .nvml import NVML
from .report import Report
from .snapshot import from_dict
from .suggest import suggest
from .timeseries import History


class Watcher:
    def __init__(
        self,
        stacks: str | None = None,
        max_entries: int = 200_000,
        interval_ms: int = 200,
        report_dir: str | None = None,
        on_report: Callable[[Report], None] | None = None,
        quiet: bool = False,
        mode: str = "auto",
        cupti: str | bool = "auto",
    ) -> None:
        self.stacks = stacks
        self.max_entries = max_entries
        self.interval_ms = interval_ms
        self.report_dir = report_dir or os.getcwd()
        self.on_report = on_report
        self.quiet = quiet
        self.nvml = NVML()
        self.baseline: dict[int, int] = {}
        self.history: History | None = None
        self.uuids: dict[int, str | None] = {}
        self.reports: list[Report] = []
        self.mode = mode
        self.cupti = cupti
        self.native = None
        self.cupti_rc: int | None = None
        self._last: tuple[int, int, float] | None = None
        self._lock = threading.Lock()
        self._installed = False

    def install(self) -> Watcher:
        import torch

        if self._installed:
            return self
        if self.mode in ("auto", "native"):
            from .native import load

            self.native = load()
            if self.native is None and self.mode == "native":
                raise RuntimeError(
                    "vramxray: native mode requested but the extension did not build"
                )
        # subscribe before the context exists so torch's own kernel images get counted too
        if self.native is not None and self.cupti in ("auto", True):
            self.cupti_rc = self.native.cupti_start()
            if self.cupti_rc != 0 and not self.quiet:
                why = (
                    "torch.profiler or another tool holds it"
                    if self.cupti_rc == 39
                    else f"error {self.cupti_rc}"
                )
                print(
                    f"vramxray: CUPTI unavailable ({why}). Libraries will not be named",
                    file=sys.stderr,
                )
        already_up = torch.cuda.is_initialized()
        before = {} if already_up else {d: self.nvml.memory(d) for d in range(_count())}
        torch.cuda.init()
        for d in range(torch.cuda.device_count()):
            if not already_up:
                # init() alone makes no context. The first kernel does, so run one
                torch.empty(1, device=f"cuda:{d}").zero_()
                torch.cuda.synchronize(d)
            self.uuids[d] = _uuid(d)
            m = self.nvml.memory(d, self.uuids[d])
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
        if self.native is not None:
            self.native.install()
            self._seed_mirror()
        self.history = History(self.nvml, self.uuids, self.native, self.interval_ms).start()
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
        nvml = self.nvml.memory(device, self.uuids.get(device))
        if allowed_max is not None and nvml is not None and allowed_max >= nvml.total:
            allowed_max = None  # torch passes device_total when no fraction is set
        ex: Explanation = explain(
            snap, device, request=request, device_free=device_free, cap=allowed_max
        )
        procs = self.nvml.processes(device, self.uuids.get(device))
        others = [p for p in procs if p.pid != os.getpid()]
        libs, images = self._native_buckets()
        acc: Accounting = reconcile(
            nvml,
            reserved,
            allocated,
            self.baseline.get(device, -1),
            libs=libs,
            images_by_lib=images,
            other_processes=others,
            processes_known=any(p.used is not None for p in procs),
            allowed_max=allowed_max,
        )
        peak = self.history.peak(device) if self.history else None
        stalls = dict(self.native.stalls()) if self.native is not None else {}
        trend = self.history.largest_free_trend(device) if self.history else 0.0
        rep = Report(
            device,
            ex,
            acc,
            suggest(ex, acc, trend),
            request=request,
            rank=_rank(),
            torch_version=torch.__version__,
            peak=peak,
            stalls=stalls,
        )
        self.reports.append(rep)
        return rep

    def _seed_mirror(self) -> None:
        """Tell the native mirror about segments that already existed when we attached.

        It only sees events from now on, so without this the numbers are short by whatever
        torch had already reserved, which is at least the warm up kernel's segment.
        """
        import torch

        from .native import TA_ALLOC, TA_SEGMENT_ALLOC

        for device in range(torch.cuda.device_count()):
            for seg in torch.cuda.memory._snapshot(device).get("segments", []):
                if seg.get("device", device) != device:
                    continue
                self.native.mirror_event(
                    TA_SEGMENT_ALLOC, device, seg["address"], seg["total_size"]
                )
                addr = seg["address"]
                for blk in seg.get("blocks", []):
                    size = blk["size"]
                    if blk["state"] != "inactive":
                        self.native.mirror_event(
                            TA_ALLOC, device, blk.get("address", addr), blk["requested_size"]
                        )
                    addr += size

    def _native_buckets(self) -> tuple[dict[str, int], dict[str, int]]:
        """Live bytes and kernel-image bytes per library. torch's own rows are dropped because
        torch_reserved already counts them."""
        if self.native is None:
            return {}, {}
        own = ("libc10_cuda", "libtorch", "torch")
        libs = {k: v for k, v in self.native.libs().items() if not k.startswith(own) and v}
        images = {k: v for k, v in self.native.kernel_images().items() if v}
        return libs, images

    def release_cupti(self) -> None:
        """Give the CUPTI subscription back, for example before a torch.profiler session."""
        if self.native is not None:
            self.native.cupti_stop()

    def timeline(self):
        from .events import Timeline

        if self.native is None:
            raise RuntimeError("timeline needs native mode")
        return Timeline(self.native.drain())


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
