from __future__ import annotations

__version__ = "0.1.0"

from . import events, timeseries  # noqa: F401  imported so they cannot shadow the names below
from .frag import Explanation, explain
from .report import Report
from .snapshot import Snapshot, load

_watcher = None


def watch(
    stacks: str | None = None,
    max_entries: int = 200_000,
    interval_ms: int = 200,
    report_dir: str | None = None,
    quiet: bool = False,
    on_report=None,
    mode: str = "auto",
    cupti: str | bool = "auto",
    warn_at: float = 0.0,
    track_sites: bool = False,
):
    """Start watching. Best called before the first CUDA call, but works after too.

    stacks: None (cheapest), "python", or "all" (python + C++ frames) for allocation stacks.
    interval_ms: how often to record a row of memory history in the background.
    mode: "auto" uses the C++ core when it builds, "python" never tries, "native" insists.
    cupti: whether to name the libraries behind non-torch memory; conflicts with torch.profiler.
    """
    global _watcher
    if _watcher is None:
        from .hooks_py import Watcher

        _watcher = Watcher(
            stacks,
            max_entries,
            interval_ms,
            report_dir,
            on_report,
            quiet,
            mode,
            cupti,
            warn_at,
            track_sites,
        ).install()
    return _watcher


from . import streams  # noqa: E402


def release_cupti() -> None:
    """Hand the CUPTI subscription back so torch.profiler can take it."""
    if _watcher is not None:
        _watcher.release_cupti()


def history():
    """What memory has been doing since watch() started, as rows you can turn into a table."""
    return watch().history


def growth(device: int = 0):
    """Call sites whose live memory keeps climbing over the run, biggest first."""
    return watch().history.growth(device)


def stalls() -> dict[str, dict[str, float]]:
    """Time spent inside driver allocation calls, per library. Native mode only."""
    w = watch()
    if w.native is None:
        return {}
    return {
        lib: {"seconds": ns / 1e9, "calls": calls} for lib, (ns, calls) in w.native.stalls().items()
    }


def timeline():
    """Everything recorded so far as columns. Native mode only."""
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
    "growth",
    "history",
    "load",
    "release_cupti",
    "report",
    "stalls",
    "streams",
    "timeline",
    "watch",
    "__version__",
]
