"""vramxray: see the GPU memory PyTorch can't show you."""

from __future__ import annotations

__version__ = "0.0.1"

from .frag import Explanation, explain
from .report import Report
from .snapshot import Snapshot, load

_watcher = None


def watch(
    stacks: str | None = None,
    max_entries: int = 200_000,
    interval_ms: int = 50,
    report_dir: str | None = None,
    quiet: bool = False,
    on_report=None,
    mode: str = "auto",
    cupti: str | bool = "auto",
):
    """Start watching. Best called before the first CUDA call, but works after too.

    stacks: None (cheapest), "python", or "all" (python + C++ frames) for allocation stacks.
    mode: "auto" uses the C++ core when it builds, "python" never tries, "native" insists.
    cupti: whether to name the libraries behind non-torch memory; conflicts with torch.profiler.
    """
    global _watcher
    if _watcher is None:
        from .hooks_py import Watcher

        _watcher = Watcher(
            stacks, max_entries, interval_ms, report_dir, on_report, quiet, mode, cupti
        ).install()
    return _watcher


def release_cupti() -> None:
    """Hand the CUPTI subscription back so torch.profiler can take it."""
    if _watcher is not None:
        _watcher.release_cupti()


def timeline():
    """Everything recorded so far as columns; native mode only."""
    return watch().timeline()


def report(device: int | None = None) -> Report:
    """Build a report right now, no OOM needed. Starts watching if nothing has yet."""
    return watch().report(device)


def analyze(source) -> Explanation:
    """Explain a snapshot pickle or dict offline."""
    return explain(load(source))


__all__ = [
    "Explanation",
    "Report",
    "Snapshot",
    "analyze",
    "explain",
    "load",
    "report",
    "watch",
    "__version__",
]
