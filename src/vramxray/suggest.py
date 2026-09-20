"""Turn the analysis into a short list of things worth trying, each with a size estimate."""

from __future__ import annotations

from dataclasses import dataclass

from .accounting import Accounting
from .frag import Explanation, if_freed
from .snapshot import fmt_bytes


@dataclass(frozen=True)
class Suggestion:
    text: str
    recovers: int  # rough bytes this would make available, 0 if we cannot estimate

    def __str__(self) -> str:
        est = f"   est. {fmt_bytes(self.recovers)}" if self.recovers else ""
        return f"{self.text}{est}"


def suggest(ex: Explanation, acc: Accounting | None) -> list[Suggestion]:
    out: list[Suggestion] = []
    req = ex.request or 0

    if ex.verdict == "fragmentation":
        if ex.free_in_segments >= req and ex.expandable_recoverable:
            out.append(
                Suggestion(
                    "PYTORCH_ALLOC_CONF=expandable_segments:True, so free pages from different "
                    f"holes can back one block; the {fmt_bytes(ex.free_in_segments)} free would "
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
                where = "; ".join(_where(p) for p in cand)
                out.append(
                    Suggestion(
                        f"free or move the {k} pinning block{'s' if k > 1 else ''} above "
                        f"({where}), for example by allocating them before the big tensors; "
                        f"that gives a {fmt_bytes(merged)} hole",
                        0,
                    )
                )
                break

    if ex.verdict == "stream":
        out.append(
            Suggestion(
                "the free memory belongs to another stream's pool; call torch.cuda.empty_cache() "
                "or keep this allocation on the stream that owns the cached blocks",
                0,
            )
        )

    if acc is not None:
        if acc.allowed_max and acc.nvml and acc.allowed_max < acc.nvml.total:
            out.append(
                Suggestion(
                    f"this process is capped at {fmt_bytes(acc.allowed_max)} by "
                    "set_per_process_memory_fraction (or the serving framework); raise it if the "
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
                    f"NCCL holds {fmt_bytes(nccl)}; check NCCL_BUFFSIZE and how many "
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
                    f"{fmt_bytes(acc.unattributed)} is used by something outside torch; install "
                    "vramxray[native] to see which library (NCCL, cuBLAS, Triton, ...)",
                    0,
                )
            )
    if ex.verdict == "exhaustion" and not out:
        out.append(
            Suggestion(
                "this is genuine exhaustion: smaller batch, activation checkpointing, or "
                "offloading",
                0,
            )
        )
    return out


def _where(p) -> str:
    frames = [f for f in p.frames if "/torch/" not in f.filename] or list(p.frames)
    if not frames:
        return f"{fmt_bytes(p.block.size)} block, no stack"
    f = frames[0]
    return f"{fmt_bytes(p.block.size)} at {f.filename.rsplit('/', 1)[-1]}:{f.line}"
