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
    html: bool = True,
):
    """Start watching. Best called before the first CUDA call, but works after too.

    stacks: None (cheapest), "python", or "all" (python plus C++ frames) for allocation stacks.
    interval_ms: how often to record a row of memory history in the background.
    mode: "auto" uses the C++ core when it builds, "python" never tries, "native" insists.
    cupti: whether to name the libraries behind non-torch memory. Conflicts with torch.profiler.
    warn_at: report before the OOM, once memory passes this mark. A value of 1 or less is a
        share of the whole device, anything larger is a byte count for this process.
    track_sites: sample live memory per call site every few seconds so growth() can spot a leak.
        Off by default because it takes a snapshot, which costs tens of milliseconds.
    html: also write a page next to the json report when an OOM happens.
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
            html,
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


def html(path: str = "vramxray.html", device: int | None = None) -> str:
    """Write the current report as a page, with the segment map drawn to scale."""
    w = watch()
    rep = w.report(device)
    from .html import write

    rows = w.history.for_device(rep.device) if w.history else []
    return write(
        path,
        ex=rep.explanation,
        acc=rep.accounting,
        rows=rows,
        suggestions=rep.suggestions,
        title=f"vramxray cuda:{rep.device}",
    )


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
    "html",
    "load",
    "release_cupti",
    "report",
    "stalls",
    "streams",
    "timeline",
    "watch",
    "__version__",
]
