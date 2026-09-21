"""Stream hygiene: the checks an NVIDIA engineer listed in pytorch/pytorch#167885.

Needs native mode with CUPTI. Findings are counted, not logged one by one, so leaving the
checks on for a whole training run costs a hash lookup per launch.
"""

from __future__ import annotations

from dataclasses import dataclass

EXPLAIN = {
    "blocking_stream": (
        "a stream was created without cudaStreamNonBlocking, so it synchronizes with the "
        "legacy default stream in both directions. Most of torch runs on that default stream, "
        "so every launch on this stream serializes against torch"
    ),
    "launch_outside_capture": (
        "work was launched on a stream that is not part of an active CUDA graph capture. It "
        "ran once during capture and will not be in the graph on replay"
    ),
    "legacy_stream_launch": (
        "launches on the legacy null stream. Fine for torch itself, but a library doing this "
        "while torch uses side streams gets implicit synchronization it did not ask for"
    ),
    "shared_channel": (
        "two or more streams were mapped to the same hardware channel, so their kernels "
        "serialize even though the streams are independent (a false dependency)"
    ),
}


@dataclass(frozen=True)
class Finding:
    kind: str
    lib: str
    detail: str
    count: int

    def __str__(self) -> str:
        who = f" by {self.lib}" if self.lib else ""
        return f"{self.kind:<24} x{self.count:<6} {self.detail}{who}"


def _native():
    from . import _watcher

    if _watcher is None or _watcher.native is None:
        raise RuntimeError("stream checks need vramxray.watch(mode='native') with CUPTI")
    return _watcher.native


def start(channels: bool = False, check_legacy: bool = False) -> None:
    """Turn the checks on.

    channels=True also records which hardware channel each kernel ran on, which is what finds
    false dependencies between streams.

    check_legacy=True reports launches on the legacy null stream. It is off by default because
    torch puts everything there, so it fires on every kernel and costs real time.
    """
    rc = _native().streams_start(channels, check_legacy)
    if rc != 0:
        raise RuntimeError(f"could not enable stream checks (CUPTI result {rc})")


def stop() -> None:
    _native().streams_stop()


def findings() -> list[Finding]:
    out = [Finding(**f) for f in _native().streams_findings()]
    return sorted(out, key=lambda f: (f.kind, -f.count))


def report() -> str:
    fs = findings()
    if not fs:
        return "vramxray streams: nothing to report"
    lines = ["vramxray streams:"]
    seen = set()
    for f in fs:
        if f.kind not in seen:
            seen.add(f.kind)
            lines.append(f"  {f.kind}: {EXPLAIN.get(f.kind, '')}")
        lines.append(f"    {f}")
    return "\n".join(lines)
