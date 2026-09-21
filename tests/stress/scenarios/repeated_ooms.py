"""Several OOMs in one process should give one report each and leave the process usable."""

import time

import torch
from _common import MiB, cap

import vramxray

cap(1024)
reports = []
vramxray.watch(stacks="python", on_report=reports.append, quiet=True)

keep = []
for _ in range(4):
    try:
        keep.append(torch.empty(700 * MiB, dtype=torch.uint8, device="cuda"))
    except torch.OutOfMemoryError:
        keep.clear()
        torch.cuda.empty_cache()
    time.sleep(2.1)  # longer than the window that folds a retry into one report

print("reports", len(reports), [r.explanation.verdict for r in reports])
assert len(reports) == 2, "expected exactly one report per OOM"

still_works = torch.ones(10, device="cuda").sum().item()
assert still_works == 10
