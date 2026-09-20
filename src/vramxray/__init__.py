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
):
    """Start watching. Call once, before or after CUDA is initialized.

    stacks: None (cheapest), "python", or "all" (python + C++ frames) for allocation stacks.
    """
    global _watcher
    if _watcher is None:
        from .hooks_py import Watcher

        _watcher = Watcher(stacks, max_entries, interval_ms, report_dir, on_report, quiet).install()
    return _watcher


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
