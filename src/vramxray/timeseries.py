"""What memory did over time, sampled cheaply in the background.

torch can tell you the peak as a single number and nothing else. This keeps a time series, so
the report can say what the peak was, when it happened, and whether the largest free block has
been shrinking, which is what a slow fragmentation problem looks like before it becomes an OOM.

Sampling is cheap because nothing here takes a snapshot. In native mode the numbers come from
the allocator mirror the C++ core keeps up to date, and otherwise from torch's own counters.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from .frag import live_sites
from .nvml import NVML
from .snapshot import from_dict

# Naming the call sites needs a snapshot, which is tens of milliseconds even on a small process,
# so it is sampled rarely at a rate picked to stay under this share of the run. The first sample
# also waits, because taking snapshots while the main thread is still pulling in torch submodules
# has been seen to break torch's own lazy imports.
SITE_SAMPLE_BUDGET = 0.005
MIN_SITE_INTERVAL = 5.0
FIRST_SITE_DELAY = 10.0


@dataclass
class Row:
    t: float  # seconds since watching started
    device: int
    nvml_used: int = 0
    reserved: int = 0
    live: int = 0
    largest_free: int = -1  # native mode only, -1 means we do not know
    old_bytes: int = 0  # live bytes allocated more than old_seconds ago
    segments: int = 0
    blocks: int = 0
    old_blocks: int = 0


@dataclass(frozen=True)
class Growth:
    where: str
    bytes_now: int
    bytes_per_minute: float
    samples: int
    seconds: float

    @property
    def over(self) -> str:
        """How long it has been watched, in whichever unit reads better."""
        if self.seconds < 120:
            return f"{self.seconds:.0f}s"
        return f"{self.seconds / 60:.0f} min"


@dataclass
class History:
    nvml: NVML
    uuids: dict[int, str | None]
    native: object | None = None
    interval_ms: int = 200
    keep: int = 100_000
    old_seconds: float = 60.0
    # a fraction of the device when it is 1 or less, otherwise an absolute byte count
    warn_at: float = 0.0
    on_warn: Callable[[int, float], None] | None = None
    track_sites: bool = False
    rows: deque[Row] = field(default_factory=deque)
    site_rows: deque[tuple[float, int, dict[str, int]]] = field(default_factory=deque)
    started: float = field(default_factory=time.monotonic)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _warned: set[int] = field(default_factory=set)
    _site_interval: float = MIN_SITE_INTERVAL
    _next_site: float = 0.0

    def start(self) -> History:
        self.rows = deque(maxlen=self.keep)
        self._next_site = time.monotonic() + FIRST_SITE_DELAY
        self._thread = threading.Thread(target=self._run, name="vramxray-history", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def sample(self) -> list[Row]:
        import torch

        out = []
        for device in range(torch.cuda.device_count()):
            row = Row(time.monotonic() - self.started, device)
            mem = self.nvml.memory(device, self.uuids.get(device))
            if mem is not None:
                row.nvml_used = mem.used
            if self.native is not None:
                m = self.native.mirror_stats(device, self.old_seconds)
                row.reserved = m["reserved"]
                row.live = m["live"]
                row.largest_free = m["largest_free"]
                row.old_bytes = m["old_bytes"]
                row.segments = m["segments"]
                row.blocks = m["blocks"]
                row.old_blocks = m["old_blocks"]
            else:
                row.reserved = torch.cuda.memory_reserved(device)
                row.live = torch.cuda.memory_allocated(device)
            out.append(row)
            self.rows.append(row)
            self._check_threshold(device, row)
        if self.track_sites and time.monotonic() >= self._next_site:
            self._sample_sites()
        return out

    def _check_threshold(self, device: int, row: Row) -> None:
        """Say something while the process is still alive, rather than only after it dies."""
        if not self.warn_at or self.on_warn is None:
            return
        mem = self.nvml.memory(device, self.uuids.get(device))
        if self.warn_at > 1:
            # an absolute mark, measured against what this process holds
            used, limit = row.reserved, self.warn_at
        elif mem:
            used, limit = (row.nvml_used or row.reserved), mem.total * self.warn_at
        else:
            return
        if used < limit:
            self._warned.discard(device)  # dropped back down, so arm it again
            return
        if device in self._warned:
            return
        self._warned.add(device)
        share = used / mem.total if mem and mem.total else 0.0
        self.on_warn(device, share)

    def _sample_sites(self) -> None:
        """Record live bytes per call site, and pick the next interval from what it just cost."""
        import torch

        started = time.monotonic()
        for device in range(torch.cuda.device_count()):
            try:
                sites = live_sites(
                    from_dict(torch.cuda.memory._snapshot(device)).for_device(device), top=64
                )
            except Exception:
                return
            self.site_rows.append(
                (started - self.started, device, {s.where: s.bytes for s in sites})
            )
        cost = time.monotonic() - started
        self._site_interval = max(MIN_SITE_INTERVAL, cost / SITE_SAMPLE_BUDGET)
        self._next_site = time.monotonic() + self._site_interval

    def _run(self) -> None:
        while not self._stop.wait(self.interval_ms / 1000):
            try:
                self.sample()
            except Exception:
                return  # interpreter shutting down, or CUDA gone. Nothing useful left to do.

    def for_device(self, device: int) -> list[Row]:
        return [r for r in self.rows if r.device == device]

    def peak(self, device: int = 0) -> Row | None:
        rows = self.for_device(device)
        return max(rows, key=lambda r: r.reserved) if rows else None

    def growth(self, device: int = 0, min_bytes_per_minute: int = 1 << 20) -> list[Growth]:
        """Call sites whose live memory keeps climbing, which is what a leak looks like.

        Needs at least two site samples, so a short run reports nothing.
        """
        rows = [(t, s) for t, d, s in self.site_rows if d == device]
        if len(rows) < 2:
            return []
        (t0, first), (t1, last) = rows[0], rows[-1]
        minutes = max((t1 - t0) / 60, 1e-9)
        out = []
        for where, now in last.items():
            rate = (now - first.get(where, 0)) / minutes
            if rate >= min_bytes_per_minute:
                out.append(Growth(where, now, rate, len(rows), t1 - t0))
        out.sort(key=lambda g: -g.bytes_per_minute)
        return out

    def largest_free_trend(self, device: int = 0, window: int = 40) -> float:
        """Change per sample in the biggest free block, negative when fragmentation is building.

        Zero when we cannot tell, which includes pure mode and a series that is too short.
        """
        rows = [r for r in self.for_device(device) if r.largest_free >= 0]
        if len(rows) < window:
            return 0.0
        first = rows[-window : -window // 2]
        last = rows[-window // 2 :]
        a = sum(r.largest_free for r in first) / len(first)
        b = sum(r.largest_free for r in last) / len(last)
        return (b - a) / (window / 2)

    def columns(self) -> dict[str, list]:
        names = (
            "t",
            "device",
            "nvml_used",
            "reserved",
            "live",
            "largest_free",
            "old_bytes",
            "segments",
            "blocks",
            "old_blocks",
        )
        return {n: [getattr(r, n) for r in self.rows] for n in names}

    def to_polars(self):
        import polars as pl

        return pl.DataFrame(self.columns())

    def to_arrow(self):
        import pyarrow as pa

        return pa.table(self.columns())

    def __len__(self) -> int:
        return len(self.rows)
