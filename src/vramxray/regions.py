"""Tag memory with the phase of the step that allocated it."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps


def _native():
    from . import _watcher

    return None if _watcher is None else _watcher.native


@contextmanager
def region(name: str) -> Iterator[None]:
    """Everything allocated inside is attributed to this name, including on the backward thread."""
    core = _native()
    if core is None:
        yield
        return
    previous = core.region_begin(name)
    try:
        yield
    finally:
        core.region_end(previous)


def tagged(name: str | None = None) -> Callable:
    def wrap(fn):
        label = name or fn.__qualname__

        @wraps(fn)
        def inner(*args, **kwargs):
            with region(label):
                return fn(*args, **kwargs)

        return inner

    return wrap


def live(device: int = 0) -> list[dict]:
    core = _native()
    return list(core.mirror_regions(device)) if core is not None else []
