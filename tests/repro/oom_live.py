"""Same fragmentation setup as split_remainder.py, but the OOM is left uncaught so the live
report prints in front of the traceback. Run under `vramxray run` or with watch() as below."""

import torch

import vramxray

MiB = 1 << 20
BUDGET = 2048 * MiB
torch.cuda.set_per_process_memory_fraction(
    BUDGET / torch.cuda.get_device_properties(0).total_memory
)
vramxray.watch(stacks="python")

filler = [torch.empty(256 * MiB, dtype=torch.uint8, device="cuda") for _ in range(6)]
big = torch.empty(512 * MiB, dtype=torch.uint8, device="cuda")
del big
mid = torch.empty(500 * MiB, dtype=torch.uint8, device="cuda")
pin = torch.empty(4 * MiB, dtype=torch.uint8, device="cuda")
del mid
boom = torch.empty(505 * MiB, dtype=torch.uint8, device="cuda")
