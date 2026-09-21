"""Helpers every stress scenario wants. Scenarios run as scripts, so this imports by name."""

from __future__ import annotations

MiB = 1 << 20


def cap(mb: int) -> None:
    """Limit this process to mb megabytes so an OOM happens the same way on any GPU."""
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(mb * MiB / total)
