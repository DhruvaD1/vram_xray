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
from dataclasses import dataclass, field

from .nvml import NVML


@dataclass
class Row:
    t: float  # seconds since watching started
    device: int
    nvml_used: int = 0
    reserved: int = 0
    live: int = 0
    largest_free: int = -1  # native mode only, -1 means we do not know
    segments: int = 0
    blocks: int = 0


@dataclass
class History:
    nvml: NVML
    uuids: dict[int, str | None]
    native: object | None = None
    interval_ms: int = 200
    keep: int = 100_000
    rows: deque[Row] = field(default_factory=deque)
    started: float = field(default_factory=time.monotonic)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> History:
        self.rows = deque(maxlen=self.keep)
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
                m = self.native.mirror_stats(device)
                row.reserved = m["reserved"]
                row.live = m["live"]
                row.largest_free = m["largest_free"]
                row.segments = m["segments"]
                row.blocks = m["blocks"]
            else:
                row.reserved = torch.cuda.memory_reserved(device)
                row.live = torch.cuda.memory_allocated(device)
            out.append(row)
            self.rows.append(row)
        return out

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
            "segments",
            "blocks",
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
