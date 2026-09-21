"""Turn a torch.cuda.memory snapshot into plain objects the rest of the package can work with.

A snapshot is what `torch.cuda.memory._snapshot()` returns and `_dump_snapshot()` pickles.
The keys have not changed between torch 2.5 and 2.9, only a couple were added, so we read
what we know and default anything missing.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MiB = 1 << 20
GiB = 1 << 30

# block states as torch spells them
ACTIVE = "active_allocated"
PENDING = "active_pending_free"
INACTIVE = "inactive"


@dataclass(frozen=True)
class Frame:
    filename: str
    line: int
    name: str

    def __str__(self) -> str:
        return f"{self.filename}:{self.line} {self.name}"


@dataclass
class Block:
    address: int
    size: int
    requested_size: int  # what the caller asked for; torch rounds up to `size`
    state: str
    frames: tuple[Frame, ...] = ()

    @property
    def end(self) -> int:
        return self.address + self.size

    @property
    def live(self) -> bool:
        # pending-free blocks still occupy their bytes until the stream catches up
        return self.state != INACTIVE


@dataclass
class Segment:
    """One cudaMalloc (or one mapped range with expandable segments), split into blocks."""

    address: int
    total_size: int
    stream: int
    device: int
    segment_type: str  # "small" holds requests up to 1 MiB, "large" everything else
    is_expandable: bool
    blocks: list[Block]

    @property
    def end(self) -> int:
        return self.address + self.total_size

    @property
    def free(self) -> int:
        return sum(b.size for b in self.blocks if not b.live)

    @property
    def live(self) -> int:
        return self.total_size - self.free


@dataclass
class Trace:
    action: str
    addr: int  # for an "oom" entry this is the device's free bytes, not an address
    size: int  # for an "oom" entry this is the failed request
    stream: int
    time_us: int
    frames: tuple[Frame, ...] = ()


@dataclass
class Snapshot:
    segments: list[Segment]
    traces: dict[int, list[Trace]] = field(default_factory=dict)  # per device, oldest first
    settings: dict[str, Any] = field(default_factory=dict)

    def devices(self) -> list[int]:
        return sorted({s.device for s in self.segments} | set(self.traces))

    def for_device(self, device: int) -> list[Segment]:
        return [s for s in self.segments if s.device == device]

    def ooms(self, device: int) -> list[Trace]:
        return [t for t in self.traces.get(device, []) if t.action == "oom"]

    def alloc_index(self, device: int) -> dict[int, tuple[int, int]]:
        """Map each address to (position of its latest alloc, time_us).

        Position counts allocs only, so "age" of a block is how many allocs came after it.
        """
        out: dict[int, tuple[int, int]] = {}
        n = 0
        for t in self.traces.get(device, []):
            if t.action == "alloc":
                out[t.addr] = (n, t.time_us)
                n += 1
        return out

    def alloc_count(self, device: int) -> int:
        return sum(1 for t in self.traces.get(device, []) if t.action == "alloc")


def _frames(raw: Iterable[dict[str, Any]] | None) -> tuple[Frame, ...]:
    if not raw:
        return ()
    return tuple(
        Frame(str(f.get("filename", "?")), int(f.get("line", 0)), str(f.get("name", "?")))
        for f in raw
    )


def _segment(raw: dict[str, Any]) -> Segment:
    blocks = []
    addr = int(raw["address"])
    for b in raw.get("blocks", []):
        size = int(b["size"])
        # older dumps have no per-block address, but blocks are listed back to back
        blocks.append(
            Block(
                int(b.get("address", addr)),
                size,
                int(b.get("requested_size", size)),
                str(b["state"]),
                _frames(b.get("frames")),
            )
        )
        addr += size
    blocks.sort(key=lambda b: b.address)
    return Segment(
        int(raw["address"]),
        int(raw["total_size"]),
        int(raw.get("stream", 0)),
        int(raw.get("device", 0)),
        str(raw.get("segment_type", "large")),
        bool(raw.get("is_expandable", False)),
        blocks,
    )


def _trace(raw: dict[str, Any]) -> Trace:
    addr = raw.get("addr", 0)
    if raw.get("action") == "oom":
        # torch 2.9 started writing the free bytes under its own key, older versions reused addr
        addr = raw.get("device_free", addr)
    return Trace(
        str(raw["action"]),
        int(addr),
        int(raw.get("size", 0)),
        int(raw.get("stream", 0)),
        int(raw.get("time_us", 0)),
        _frames(raw.get("frames")),
    )


def from_dict(raw: dict[str, Any]) -> Snapshot:
    segments = [_segment(s) for s in raw.get("segments", [])]
    traces = {d: [_trace(t) for t in ents] for d, ents in enumerate(raw.get("device_traces", []))}
    return Snapshot(segments, traces, dict(raw.get("allocator_settings") or {}))


def load(source: str | Path | dict[str, Any]) -> Snapshot:
    if isinstance(source, dict):
        return from_dict(source)
    with open(source, "rb") as f:
        return from_dict(pickle.load(f))


CPP_SUFFIXES = (".cpp", ".cc", ".cu", ".cuh", ".h", ".hpp", ".c")


def _is_cpp(f: Frame) -> bool:
    return f.filename in ("??", "") or f.filename.endswith(CPP_SUFFIXES)


def user_frames(frames: Iterable[Frame]) -> list[Frame]:
    """The frames worth showing, which means the user's own Python.

    With stacks="all" torch also hands back its C++ internals and unresolved "??" frames. Those
    are never what someone wants to read, so keep them only if there is no Python at all.
    Interpreter pseudo files like <string> and <stdin> count as Python.
    """
    frames = list(frames)
    py = [f for f in frames if not _is_cpp(f) and "/torch/" not in f.filename]
    if py:
        return py
    return [f for f in frames if f.filename not in ("??", "") and "/torch/" not in f.filename]


def fmt_bytes(n: int | float | None) -> str:
    n = float(n or 0)
    if abs(n) >= GiB:
        return f"{n / GiB:.2f} GiB"
    if abs(n) >= MiB:
        return f"{n / MiB:.1f} MiB"
    return f"{n / 1024:.0f} KiB"
