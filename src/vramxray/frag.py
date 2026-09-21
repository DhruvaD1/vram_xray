"""Work out why a request could not be served from what torch already had reserved.

The idea is simple: free memory inside a segment only helps if it is contiguous. A small
long-lived block in the middle of a segment splits the free space into holes, and a request
bigger than the biggest hole fails even though the total free looks fine. We find the holes,
name the blocks pinning them, and check what would actually fix it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .snapshot import Block, Frame, MiB, Segment, Snapshot, fmt_bytes, user_frames

# torch puts requests up to 1 MiB in its own pool of 2 MiB segments. A bigger request can
# never be placed in one of those, so they should not count as "free" for it.
SMALL_POOL_MAX = 1 * MiB


@dataclass
class Hole:
    segment: Segment
    address: int
    size: int
    left: Block | None  # live block just below, None at the segment start
    right: Block | None  # live block just above, None at the segment end

    @property
    def end(self) -> int:
        return self.address + self.size


@dataclass
class Pin:
    """A live block that sits next to one or two holes and keeps them from merging."""

    block: Block
    segment: Segment
    wasted: int  # bytes of hole this block bounds
    age: int | None = None  # allocs that happened after this one, if history was on
    holes: list[Hole] = field(default_factory=list)

    @property
    def frames(self) -> tuple[Frame, ...]:
        return self.block.frames

    @property
    def leverage(self) -> float:
        # hole bytes per byte held. A 4 MiB block pinning 500 MiB is the one to go after.
        return self.wasted / max(self.block.size, 1)


@dataclass
class Explanation:
    device: int
    request: int | None
    stream: int | None
    device_free: int | None  # what the driver had free at OOM time, if we know
    reserved: int
    live: int
    free_in_segments: int  # free bytes in segments this request could actually use
    largest_hole: int
    trapped: int  # free_in_segments minus largest_hole
    holes: list[Hole]
    pins: list[Pin]
    verdict: str  # fragmentation, exhaustion, stream, or unknown
    verdict_text: str
    expandable_recoverable: int  # trapped bytes sitting in segments that are not expandable
    sites: list[Site] = field(default_factory=list)  # live memory by call site, biggest first
    cap: int | None = None  # per-process limit in force, if any
    # the env var this torch actually reads. 2.9 warns that PYTORCH_CUDA_ALLOC_CONF is deprecated
    # but still ignores PYTORCH_ALLOC_CONF, so take the name from the snapshot instead of guessing
    alloc_conf_var: str = "PYTORCH_CUDA_ALLOC_CONF"
    expandable_on: bool = False


@dataclass
class Site:
    """Where the live memory came from: one entry per allocation site, summed."""

    where: str
    bytes: int
    count: int


def live_sites(segments: list[Segment], top: int = 6) -> list[Site]:
    """Live bytes grouped by the user-code frames that allocated them."""
    totals: dict[str, list[int]] = {}
    for seg in segments:
        for b in seg.blocks:
            if not b.live:
                continue
            key = _site_key(b.frames)
            t = totals.setdefault(key, [0, 0])
            t[0] += b.size
            t[1] += 1
    sites = [Site(k, v[0], v[1]) for k, v in totals.items()]
    sites.sort(key=lambda s: -s.bytes)
    return sites[:top]


def _site_key(frames: tuple[Frame, ...]) -> str:
    user = user_frames(frames)
    if not user:
        # no Python frames usually means the autograd engine allocated it on its own thread
        return (
            "(no Python stack: autograd/backward or history off)"
            if not frames
            else "torch internals"
        )
    return " from ".join(f"{f.filename.rsplit('/', 1)[-1]}:{f.line} {f.name}" for f in user[:2])


def holes_in(seg: Segment) -> list[Hole]:
    """Runs of free blocks in a segment, with the live neighbours on each side."""
    out: list[Hole] = []
    blocks = seg.blocks
    i = 0
    while i < len(blocks):
        if blocks[i].live:
            i += 1
            continue
        j = i
        while j < len(blocks) and not blocks[j].live:
            j += 1
        left = blocks[i - 1] if i > 0 else None
        right = blocks[j] if j < len(blocks) else None
        start = blocks[i].address
        out.append(Hole(seg, start, blocks[j - 1].end - start, left, right))
        i = j
    return out


def usable_segments(
    segments: list[Segment], request: int | None, stream: int | None
) -> list[Segment]:
    """Segments torch would even consider for this request: same stream, right pool."""
    segs = segments
    if stream is not None:
        segs = [s for s in segs if s.stream == stream]
    if request is not None and request > SMALL_POOL_MAX:
        segs = [s for s in segs if s.segment_type != "small"]
    return segs


def pins_for(
    holes: list[Hole], alloc_index: dict[int, tuple[int, int]] | None, alloc_count: int
) -> list[Pin]:
    by_block: dict[int, Pin] = {}
    for h in holes:
        for nb in (h.left, h.right):
            if nb is None:
                continue
            p = by_block.get(nb.address)
            if p is None:
                age = None
                if alloc_index and nb.address in alloc_index:
                    age = alloc_count - 1 - alloc_index[nb.address][0]
                p = by_block[nb.address] = Pin(nb, h.segment, 0, age)
            p.wasted += h.size
            p.holes.append(h)
    pins = list(by_block.values())
    # small and old first: those are the ones someone can realistically move or free
    pins.sort(key=lambda p: (-p.leverage, -(p.age or 0)))
    return pins


def if_freed(holes: list[Hole], freed: list[Block]) -> int:
    """Largest hole we would get if these live blocks were gone and their holes merged."""
    gone = {b.address for b in freed}
    best = 0
    for seg in {h.segment.address: h.segment for h in holes}.values():
        run = 0
        for b in seg.blocks:
            if b.live and b.address not in gone:
                run = 0
            else:
                run += b.size
                best = max(best, run)
    return best


def _verdict(
    request: int | None,
    largest: int,
    free_in: int,
    other_stream_free: int,
    device_free: int | None,
    cap: int | None = None,
) -> tuple[str, str]:
    b = fmt_bytes
    room = "under the process cap" if cap else "on the device"
    if request is None:
        return "unknown", (
            "No OOM entry in the trace. Pass --request to evaluate a hypothetical allocation."
        )
    if request <= largest:
        return "unknown", (
            f"A {b(request)} request fits in the largest hole ({b(largest)}), so "
            "the OOM was not caused by this layout."
        )
    if request <= free_in:
        return "fragmentation", (
            f"{b(free_in)} is free inside torch's segments but the largest contiguous hole "
            f"is {b(largest)}, so a {b(request)} request could not be placed. "
            "This is fragmentation, not exhaustion."
        )
    if other_stream_free >= request:
        return "stream", (
            f"{b(other_stream_free)} is free in segments owned by other streams, and torch never "
            f"serves a request from another stream's pool. The requesting stream only had "
            f"{b(free_in)} free."
        )
    need = max(request - free_in - (device_free or 0), 0)
    return "exhaustion", (
        f"{b(request)} requested with {b(free_in)} free in segments and "
        f"{b(device_free)} free {room}. Even after returning every cached block "
        f"to the driver, {b(need)} would still be missing. This is real exhaustion. "
        "The live memory below says what it is spent on."
    )


def explain(
    snap: Snapshot,
    device: int = 0,
    request: int | None = None,
    stream: int | None = None,
    device_free: int | None = None,
    cap: int | None = None,
) -> Explanation:
    """Explain the last OOM on a device, or a hypothetical request if one is given.

    cap is torch's per-process limit when one is set; the room left under it, not the
    device's free memory, is then what a new segment could have used.
    """
    segments = snap.for_device(device)
    if request is None:
        ooms = snap.ooms(device)
        if ooms:
            last = ooms[-1]
            request, stream, device_free = last.size, last.stream, last.addr

    segs = usable_segments(segments, request, stream)
    holes = sorted((h for s in segs for h in holes_in(s)), key=lambda h: -h.size)
    reserved = sum(s.total_size for s in segments)
    live = sum(s.live for s in segments)
    free_in = sum(h.size for h in holes)
    largest = holes[0].size if holes else 0
    have_history = bool(snap.traces.get(device))
    pins = pins_for(
        holes, snap.alloc_index(device) if have_history else None, snap.alloc_count(device)
    )

    # with expandable segments, free pages anywhere in a segment can back one allocation,
    # so everything except the biggest hole becomes usable again
    fixed = [h.size for h in holes if not h.segment.is_expandable]
    expandable_recoverable = sum(fixed) - max(fixed, default=0)

    other_stream_free = sum(s.free for s in segments if s not in segs and s.segment_type != "small")
    headroom = device_free
    if cap is not None:
        headroom = max(cap - reserved, 0)
        if device_free is not None:
            headroom = min(headroom, device_free)
    verdict, text = _verdict(request, largest, free_in, other_stream_free, headroom, cap)
    if cap is not None and verdict != "unknown":
        text += (
            f" This process is capped at {fmt_bytes(cap)} (set_per_process_memory_fraction or "
            f"a serving framework's budget) and had {fmt_bytes(headroom)} left under it."
        )
    elif verdict != "unknown" and request is not None and device_free is not None:
        if device_free >= request:
            # the driver said there was room, so something other than the GPU said no
            text += (
                f" The driver reported {fmt_bytes(device_free)} free, more than the request, "
                "yet torch could not open a new segment: a per-process limit "
                "(set_per_process_memory_fraction or a serving framework's budget) or another "
                "process claimed that memory first."
            )
    return Explanation(
        device,
        request,
        stream,
        device_free,
        reserved,
        live,
        free_in,
        largest,
        free_in - largest,
        holes,
        pins,
        verdict,
        text,
        expandable_recoverable,
        live_sites(segments),
        cap,
        alloc_conf_var(snap.settings),
        bool(snap.settings.get("expandable_segments", False)),
    )


def alloc_conf_var(settings: dict) -> str:
    """Which environment variable this torch build actually reads."""
    for key in settings:
        if key.endswith("ALLOC_CONF"):
            return key
    return "PYTORCH_CUDA_ALLOC_CONF"
