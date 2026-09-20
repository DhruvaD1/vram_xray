"""Just enough NVML through ctypes: how much of the device is used, and by which processes.

pynvml would do, but it is one more dependency and we only need four calls. Everything here
degrades to "unavailable" instead of raising, because a missing driver library should never
break the OOM report.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

_CANDIDATES = (
    "libnvidia-ml.so.1",
    "/usr/lib/wsl/lib/libnvidia-ml.so.1",
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
    "nvml.dll",
)

NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
NVML_VALUE_NOT_AVAILABLE = (
    1 << 64
) - 1  # what NVML puts in a field it cannot fill (WSL2 does this)


class _Memory2(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("total", ctypes.c_ulonglong),
        ("reserved", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _Memory1(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _Process(ctypes.Structure):
    _fields_ = [
        ("pid", ctypes.c_uint),
        ("used", ctypes.c_ulonglong),
        ("gpu_instance", ctypes.c_uint),
        ("compute_instance", ctypes.c_uint),
    ]


@dataclass(frozen=True)
class MemoryInfo:
    total: int
    used: int
    free: int
    reserved: int = 0  # driver-side reserved, only reported by the v2 call


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    used: int | None  # None when the driver will not say


def _load() -> ctypes.CDLL | None:
    names = list(_CANDIDATES)
    found = ctypes.util.find_library("nvidia-ml")
    if found:
        names.insert(0, found)
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


class NVML:
    """One NVML session. Call `close()` when done, or use it as a context manager."""

    def __init__(self) -> None:
        self._lib = _load()
        self._ok = False
        if self._lib is not None and self._lib.nvmlInit_v2() == NVML_SUCCESS:
            self._ok = True
        self._handles: dict[int, ctypes.c_void_p] = {}

    @property
    def available(self) -> bool:
        return self._ok

    def close(self) -> None:
        if self._ok:
            self._lib.nvmlShutdown()
            self._ok = False

    def __enter__(self) -> NVML:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _handle(self, index: int, uuid: str | None = None) -> ctypes.c_void_p | None:
        if not self._ok:
            return None
        h = self._handles.get(index)
        if h is not None:
            return h
        h = ctypes.c_void_p()
        rc = -1
        # torch's device 0 is not always NVML's index 0 (CUDA_VISIBLE_DEVICES), so prefer the uuid
        if uuid:
            rc = self._lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(h))
        if rc != NVML_SUCCESS:
            rc = self._lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(h))
        if rc != NVML_SUCCESS:
            return None
        self._handles[index] = h
        return h

    def memory(self, index: int, uuid: str | None = None) -> MemoryInfo | None:
        h = self._handle(index, uuid)
        if h is None:
            return None
        m2 = _Memory2()
        m2.version = (2 << 24) | ctypes.sizeof(_Memory2)
        if self._lib.nvmlDeviceGetMemoryInfo_v2(h, ctypes.byref(m2)) == NVML_SUCCESS:
            return MemoryInfo(m2.total, m2.used, m2.free, m2.reserved)
        m1 = _Memory1()
        if self._lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(m1)) == NVML_SUCCESS:
            return MemoryInfo(m1.total, m1.used, m1.free)
        return None

    def processes(self, index: int, uuid: str | None = None) -> list[ProcessInfo]:
        """Other processes on the device. Empty under WSL2, which does not report them."""
        h = self._handle(index, uuid)
        if h is None:
            return []
        count = ctypes.c_uint(0)
        rc = self._lib.nvmlDeviceGetComputeRunningProcesses_v3(h, ctypes.byref(count), None)
        if rc == NVML_SUCCESS or count.value == 0:
            return []
        if rc != NVML_ERROR_INSUFFICIENT_SIZE:
            return []
        buf = (_Process * (count.value + 8))()
        count = ctypes.c_uint(len(buf))
        if self._lib.nvmlDeviceGetComputeRunningProcesses_v3(h, ctypes.byref(count), buf) != 0:
            return []
        return [
            ProcessInfo(p.pid, None if p.used == NVML_VALUE_NOT_AVAILABLE else p.used)
            for p in buf[: count.value]
        ]


class Sampler:
    """Background thread that records NVML used bytes for one device at a fixed interval."""

    def __init__(
        self, nvml: NVML, index: int, uuid: str | None, interval_ms: int = 50, keep: int = 20_000
    ):
        self._nvml = nvml
        self._index = index
        self._uuid = uuid
        self._interval = interval_ms / 1000
        self.samples: deque[tuple[float, int]] = deque(maxlen=keep)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vramxray-nvml", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            m = self._nvml.memory(self._index, self._uuid)
            if m is not None:
                self.samples.append((time.monotonic(), m.used))

    def peak(self) -> int:
        return max((u for _, u in self.samples), default=0)


def own_pid() -> int:
    return os.getpid()
