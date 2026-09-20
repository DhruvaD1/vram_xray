import sys

import torch

MiB = 1 << 20
BUDGET = 2048 * MiB
total = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(BUDGET / total)
torch.cuda.memory._record_memory_history(max_entries=100_000, stacks="python")

filler = [
    torch.empty(256 * MiB, dtype=torch.uint8, device="cuda") for _ in range(6)
]  # 1536 MiB of the 2048 budget
big = torch.empty(512 * MiB, dtype=torch.uint8, device="cuda")
del big  # 512 MiB segment now cached; budget fully reserved
mid = torch.empty(500 * MiB, dtype=torch.uint8, device="cuda")  # splits it: 500 live + 12 free
pin = torch.empty(4 * MiB, dtype=torch.uint8, device="cuda")  # lands in the remainder
del mid  # 500 MiB free, but the segment is pinned

try:
    torch.empty(
        505 * MiB, dtype=torch.uint8, device="cuda"
    )  # more than the 500 MiB hole, less than the 508 free
    print("no OOM; rerun")
except torch.OutOfMemoryError as e:
    print("OOM:", str(e)[:300])
torch.cuda.memory._dump_snapshot(sys.argv[1] if len(sys.argv) > 1 else "split_remainder.pickle")
print("dumped")
