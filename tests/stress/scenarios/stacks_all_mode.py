"""stacks="all" adds C++ frames. The call sites must still read as the user's own code."""

import torch
from _common import MiB, cap

import vramxray

cap(1024)
reports = []
vramxray.watch(stacks="all", on_report=reports.append, quiet=True)

try:
    _keep = [torch.empty(700 * MiB, dtype=torch.uint8, device="cuda") for _ in range(3)]
except torch.OutOfMemoryError:
    pass

ex = reports[-1].explanation
print(ex.verdict, [s.where for s in ex.sites][:2])
assert ex.sites
assert not ex.sites[0].where.endswith(".cpp:0"), "a torch C++ frame is posing as a call site"
