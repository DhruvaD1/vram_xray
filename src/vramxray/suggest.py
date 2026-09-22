"""Turn the analysis into a short list of things worth trying, each with a size estimate."""

from __future__ import annotations

from dataclasses import dataclass

from .accounting import Accounting
from .frag import Explanation, if_freed
from .snapshot import fmt_bytes, user_frames


@dataclass(frozen=True)
class Suggestion:
    text: str
    recovers: int  # rough bytes this would make available, 0 if we cannot estimate

    def __str__(self) -> str:
        est = f"   est. {fmt_bytes(self.recovers)}" if self.recovers else ""
        return f"{self.text}{est}"


def suggest(
    ex: Explanation,
    acc: Accounting | None,
    free_block_trend: float = 0.0,
    growth: list | None = None,
) -> list[Suggestion]:
    out: list[Suggestion] = []
    req = ex.request or 0

    # memory that climbs every minute is a leak, and no single allocation looks wrong
    if growth:
        g = growth[0]
        out.append(
            Suggestion(
                f"{g.where} has grown by {fmt_bytes(g.bytes_per_minute)} per minute over "
                f"{g.over} and now holds {fmt_bytes(g.bytes_now)}. Something "
                "there is kept alive between steps, often a list of tensors or a loss that was "
                "never detached",
                0,
            )
        )

    # a biggest free block that keeps shrinking is fragmentation building up over a run, which
    # looks nothing like a single bad allocation and is otherwise very hard to notice
    if free_block_trend < -256 * 1024:
        out.append(
            Suggestion(
                f"the largest free block has been shrinking by about "
                f"{fmt_bytes(-free_block_trend)} per sample, so fragmentation is building over "
                "the run rather than coming from one allocation. Check what is allocated once "
                "per step and never freed",
                0,
            )
        )

    if ex.verdict == "fragmentation":
        if ex.free_in_segments >= req and ex.expandable_recoverable and not ex.expandable_on:
            out.append(
                Suggestion(
                    f"{ex.alloc_conf_var}=expandable_segments:True, so free pages from different "
                    f"holes can back one block. The {fmt_bytes(ex.free_in_segments)} free would "
                    "then fit this request",
                    0,
                )
            )
        for k in (1, 2, 3):
            cand = ex.pins[:k]
            if not cand:
                break
            merged = if_freed(ex.holes, [p.block for p in cand])
            if merged >= req:
                where = ", ".join(_where(p) for p in cand)
                out.append(
                    Suggestion(
                        f"free or move the {k} pinning block{'s' if k > 1 else ''} above "
                        f"({where}), for example by allocating them before the big tensors. "
                        f"That gives a {fmt_bytes(merged)} hole",
                        0,
                    )
                )
                break

    if ex.verdict == "stream":
        out.append(
            Suggestion(
                "the free memory belongs to another stream's pool. Call torch.cuda.empty_cache() "
                "or keep this allocation on the stream that owns the cached blocks",
                0,
            )
        )

    if acc is not None:
        if acc.allowed_max and acc.nvml and acc.allowed_max < acc.nvml.total:
            out.append(
                Suggestion(
                    f"this process is capped at {fmt_bytes(acc.allowed_max)} by "
                    "set_per_process_memory_fraction (or the serving framework). Raise it if the "
                    "device really has room",
                    0,
                )
            )
        if acc.other_total:
            pids = ", ".join(
                f"pid {p.pid} ({fmt_bytes(p.used)})" for p in acc.other_processes[:4] if p.used
            )
            out.append(
                Suggestion(f"other processes hold memory on this device: {pids}", acc.other_total)
            )
        nccl = sum(v for k, v in acc.libs.items() if "nccl" in k.lower())
        if acc.nvml and nccl > acc.nvml.total // 10:
            out.append(
                Suggestion(
                    f"NCCL holds {fmt_bytes(nccl)}. Check NCCL_BUFFSIZE and how many "
                    "communicators are alive",
                    0,
                )
            )
        if (
            acc.unattributed
            and acc.nvml
            and acc.unattributed > acc.nvml.total // 20
            and not acc.libs
        ):
            out.append(
                Suggestion(
                    f"{fmt_bytes(acc.unattributed)} is used by something outside torch. Install "
                    "vramxray[native] to see which library (NCCL, cuBLAS, Triton, ...)",
                    0,
                )
            )
    if ex.verdict == "exhaustion":
        if ex.sites and ex.sites[0].where != "(no Python stack: autograd/backward or history off)":
            top = ex.sites[0]
            share = 100 * top.bytes / max(ex.live, 1)
            out.append(
                Suggestion(
                    f"{share:.0f}% of live memory ({fmt_bytes(top.bytes)}) comes from {top.where}. "
                    "That is the place to shrink: smaller batch, activation checkpointing, "
                    "or offloading",
                    0,
                )
            )
        else:
            out.append(
                Suggestion(
                    "this is genuine exhaustion: smaller batch, activation checkpointing, or "
                    "offloading. Use watch(stacks='python') to see which call sites hold it",
                    0,
                )
            )
    return out


def _where(p) -> str:
    frames = user_frames(p.frames) or list(p.frames)
    if not frames:
        return f"{fmt_bytes(p.block.size)} block, no stack"
    f = frames[0]
    return f"{fmt_bytes(p.block.size)} at {f.filename.rsplit('/', 1)[-1]}:{f.line}"
