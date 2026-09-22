"""Cross a memory threshold on purpose. The warning should arrive before any OOM does.

Run: python tests/repro/warn_early.py
"""

import time

import torch

import vramxray

MiB = 1 << 20
BUDGET = 1024 * MiB
torch.cuda.set_per_process_memory_fraction(
    BUDGET / torch.cuda.get_device_properties(0).total_memory
)

warnings = []
# an absolute mark, so the test does not depend on what else is using the card
vramxray.watch(
    stacks="python", interval_ms=50, warn_at=700 * MiB, on_report=warnings.append, quiet=True
)

# climb past half the budget without ever running out
keep = []
for _ in range(10):
    keep.append(torch.empty(96 * MiB, dtype=torch.uint8, device="cuda"))
    time.sleep(0.2)

time.sleep(0.5)
print("WARNINGS", len(warnings))
if warnings:
    print("VERDICT", warnings[0].explanation.verdict)
    print("HAS_ACCOUNTING", warnings[0].accounting is not None)
