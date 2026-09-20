from __future__ import annotations

from dataclasses import dataclass, field

from .nvml import MemoryInfo, ProcessInfo


@dataclass
class Accounting:
    nvml: MemoryInfo | None
    torch_reserved: int
    torch_allocated: int
    baseline: int  # CUDA context (and whatever else was loaded before watch()); -1 if unknown
    libs: dict[str, int] = field(default_factory=dict)  # library name -> bytes, native mode
    images_by_lib: dict[str, int] = field(default_factory=dict)  # kernel image bytes per library
    other_processes: list[ProcessInfo] = field(default_factory=list)
    processes_known: bool = False  # NVML gave usable per-process numbers (not on WSL2)
    allowed_max: int | None = None  # per-process cap torch enforces, if one is set
    unattributed: int = 0
    overcommitted: bool = False  # WDDM and WSL2 can report more used than the card has

    @property
    def kernel_images(self) -> int:
        return sum(self.images_by_lib.values())

    @property
    def other_total(self) -> int:
        return sum(p.used or 0 for p in self.other_processes)

    @property
    def context_known(self) -> bool:
        return self.baseline >= 0


def reconcile(
    nvml: MemoryInfo | None,
    torch_reserved: int,
    torch_allocated: int,
    baseline: int,
    libs: dict[str, int] | None = None,
    images_by_lib: dict[str, int] | None = None,
    other_processes: list[ProcessInfo] | None = None,
    processes_known: bool = False,
    allowed_max: int | None = None,
) -> Accounting:
    acc = Accounting(
        nvml=nvml,
        torch_reserved=torch_reserved,
        torch_allocated=torch_allocated,
        baseline=baseline,
        libs=dict(libs or {}),
        images_by_lib=dict(images_by_lib or {}),
        other_processes=list(other_processes or []),
        processes_known=processes_known,
        allowed_max=allowed_max,
    )
    if nvml is None:
        return acc
    known = torch_reserved + max(baseline, 0) + sum(acc.libs.values()) + acc.kernel_images
    known += acc.other_total
    acc.unattributed = max(nvml.used - known, 0)
    acc.overcommitted = nvml.used > nvml.total
    return acc
