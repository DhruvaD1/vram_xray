from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass, field

from .accounting import Accounting
from .frag import Explanation
from .snapshot import fmt_bytes, user_frames
from .suggest import Suggestion
from .timeseries import Row


@dataclass
class Report:
    device: int
    explanation: Explanation
    accounting: Accounting | None
    suggestions: list[Suggestion]
    request: int | None = None
    rank: int | None = None
    torch_version: str = ""
    created: float = field(default_factory=time.time)
    peak: Row | None = None  # high water mark seen since watch() started
    stalls: dict[str, tuple[int, int]] = field(default_factory=dict)  # library -> (ns, calls)

    def __str__(self) -> str:
        return render(self)

    def to_dict(self) -> dict:
        from .cli import to_json  # avoids a circular import at module load

        acc = self.accounting
        return {
            "device": self.device,
            "rank": self.rank,
            "torch": self.torch_version,
            "created": self.created,
            "request": self.request,
            "explanation": to_json(self.explanation),
            "accounting": None
            if acc is None
            else {
                "nvml_total": acc.nvml.total if acc.nvml else None,
                "nvml_used": acc.nvml.used if acc.nvml else None,
                "torch_reserved": acc.torch_reserved,
                "torch_allocated": acc.torch_allocated,
                "baseline": acc.baseline,
                "libs": acc.libs,
                "kernel_images": acc.images_by_lib,
                "other_processes": [(p.pid, p.used) for p in acc.other_processes],
                "allowed_max": acc.allowed_max,
                "unattributed": acc.unattributed,
                "overcommitted": acc.overcommitted,
            },
            "peak": None if self.peak is None else vars(self.peak),
            "stalls": {k: {"ns": ns, "calls": n} for k, (ns, n) in self.stalls.items()},
            "suggestions": [(s.text, s.recovers) for s in self.suggestions],
        }

    def write(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path


def _row(label: str, size: int | None, note: str = "") -> str:
    return f"    {label:<20} {fmt_bytes(size):>11}   {note}".rstrip()


def render(r: Report) -> str:
    b = fmt_bytes
    ex = r.explanation
    acc = r.accounting
    head = f"vramxray: cuda:{r.device}"
    if r.rank is not None:
        head += f" rank {r.rank}"
    if r.request is not None:
        head = f"vramxray: OOM on cuda:{r.device}" + (
            f" rank {r.rank}" if r.rank is not None else ""
        )
        head += f", requesting {b(r.request)}"
    lines = [head]

    if acc is not None and acc.nvml is not None:
        cap = (
            f" (process capped at {b(acc.allowed_max)})"
            if acc.allowed_max and acc.allowed_max < acc.nvml.total
            else ""
        )
        over = (
            "  (more than the card has: the driver is paging into host RAM)"
            if acc.overcommitted
            else ""
        )
        lines.append(
            f"  device (NVML)        {b(acc.nvml.used):>11} used of {b(acc.nvml.total)}{cap}{over}"
        )
        lines.append(
            _row(
                "torch reserved",
                acc.torch_reserved,
                f"allocated {b(acc.torch_allocated)} · free in segments "
                f"{b(acc.torch_reserved - acc.torch_allocated)} · largest hole "
                f"{b(ex.largest_hole)}",
            )
        )
        for name, size in sorted(acc.libs.items(), key=lambda kv: -kv[1]):
            lines.append(_row(name, size))
        if acc.kernel_images:
            top = sorted(acc.images_by_lib.items(), key=lambda kv: -kv[1])[:3]
            lines.append(
                _row("kernel images", acc.kernel_images, ", ".join(f"{k} {b(v)}" for k, v in top))
            )
        if acc.context_known:
            lines.append(_row("CUDA context", acc.baseline, "measured across torch.cuda.init()"))
        if acc.other_processes:
            pids = ", ".join(str(p.pid) for p in acc.other_processes[:4])
            lines.append(_row("other processes", acc.other_total, f"pid {pids}"))
        # say what is mixed into this number rather than pretending it is one thing
        outside = []
        if not acc.context_known:
            outside.append("this CUDA context")
        if not acc.processes_known:
            outside.append("other processes")
        if not acc.libs:
            outside.append("non-torch libraries")
        label = "unattributed" if not outside else "outside torch"
        note = "could be " + " or ".join(outside) if outside else ""
        lines.append(_row(label, acc.unattributed, note))
    elif acc is not None:
        lines.append("  device (NVML)        unavailable, only torch's own view below")
        lines.append(
            _row("torch reserved", acc.torch_reserved, f"allocated {b(acc.torch_allocated)}")
        )

    if r.peak is not None and acc is not None and r.peak.reserved > acc.torch_reserved:
        over = r.peak.reserved - acc.torch_reserved
        lines.append(
            f"    peak so far         {b(r.peak.reserved):>11}   reserved at t={r.peak.t:.0f}s, "
            f"{b(over)} above now"
        )
    if r.stalls:
        total_ns = sum(ns for ns, _ in r.stalls.values())
        calls = sum(n for _, n in r.stalls.values())
        if total_ns > 5_000_000:  # below a few milliseconds nobody cares
            worst = max(r.stalls.items(), key=lambda kv: kv[1][0])
            lines.append(
                f"    allocator stalls    {total_ns / 1e6:>8.0f} ms   over {calls} driver calls, "
                f"mostly {worst[0]}"
            )

    lines.append("")
    if r.request is not None:
        lines.append(
            f"  why {b(ex.free_in_segments)} free inside torch could not serve {b(r.request)}:"
        )
    else:
        lines.append(f"  layout: {b(ex.free_in_segments)} free inside torch's segments")
    lines.extend(
        textwrap.wrap(
            f"verdict: {ex.verdict}. {ex.verdict_text}",
            96,
            initial_indent="    ",
            subsequent_indent="      ",
        )
    )
    # holes and pins only matter when they are the reason. Otherwise say what the memory is
    if ex.verdict == "fragmentation":
        for h in ex.holes[:3]:
            seg = h.segment
            lines.append(
                f"    hole {b(h.size):>10} in a {b(seg.total_size)} segment at 0x{seg.address:x}"
            )
        if ex.pins:
            lines.append("    pinned by:")
            for p in ex.pins[:4]:
                age = f"age {p.age} allocs" if p.age is not None else ""
                lines.append(
                    f"      {b(p.block.size):>10}  holds {b(p.wasted):>10}  {age:<16} "
                    f"{_stack(p.frames)}"
                )
    if ex.sites and ex.verdict != "fragmentation":
        lines.append(f"    live memory ({b(ex.live)}) by call site:")
        for s in ex.sites:
            share = 100 * s.bytes / max(ex.live, 1)
            lines.append(f"      {b(s.bytes):>10}  {share:4.0f}%  {s.count:5d} blocks  {s.where}")

    if r.suggestions:
        lines.append("")
        lines.append("  try:")
        for s in r.suggestions:
            lines.extend(
                textwrap.wrap(str(s), 96, initial_indent="    ", subsequent_indent="      ")
            )
    return "\n".join(lines)


def _stack(frames, n: int = 2) -> str:
    keep = user_frames(frames) or list(frames)
    if not keep:
        return "(no stack, use watch(stacks='python'))"
    return " from ".join(f"{f.filename.rsplit('/', 1)[-1]}:{f.line} {f.name}" for f in keep[:n])
