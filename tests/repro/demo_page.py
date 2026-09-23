"""Produce one page that exercises every section of the report.

Useful for looking at the HTML by hand and for README screenshots. Writes ~/vramxray-demo.html.

Run: python tests/repro/demo_page.py
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

import vramxray

MiB = 1 << 20
BUDGET = 2560 * MiB
torch.cuda.set_per_process_memory_fraction(
    BUDGET / torch.cuda.get_device_properties(0).total_memory
)
vramxray.watch(stacks="python", mode="native", interval_ms=60, quiet=True)

# bring NCCL and cuBLAS in so the accounting has libraries to name
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29533")
dist.init_process_group("nccl", rank=0, world_size=1, device_id=torch.device("cuda:0"))
dist.all_reduce(torch.ones(1024, device="cuda"))

# a short training run so the timeline has a shape
model = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 2048)).cuda()
opt = torch.optim.AdamW(model.parameters())
x = torch.randn(192, 2048, device="cuda")
for _ in range(25):
    model(x).square().mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    time.sleep(0.03)
torch.cuda.synchronize()

# Now the classic pinned remainder. A hole only survives if a live block sits beside it,
# otherwise torch hands the whole cached segment back to the driver and the request fits.
# The filler has to come first, or those blocks land in the hole we are trying to create.
torch.cuda.empty_cache()
filler = []
while torch.cuda.memory_reserved() + 128 * MiB <= BUDGET - 520 * MiB:
    filler.append(torch.empty(128 * MiB, dtype=torch.uint8, device="cuda"))


def largest_hole() -> int:
    snap = torch.cuda.memory._snapshot(0)
    return max(
        (b["size"] for sg in snap["segments"] for b in sg["blocks"] if b["state"] == "inactive"),
        default=0,
    )


# The allocator serves a request from the smallest block that fits, so the pin has to be bigger
# than every hole the training run left behind, or it lands in one of those instead.
before = largest_hole()
pin_size = before + 8 * MiB
big = torch.empty(512 * MiB, dtype=torch.uint8, device="cuda")
del big  # a 512 MiB segment, now cached
mid = torch.empty(512 * MiB - pin_size, dtype=torch.uint8, device="cuda")  # splits it
pin = torch.empty(pin_size, dtype=torch.uint8, device="cuda")  # only the remainder can hold it
del mid  # a big hole, pinned open
torch.cuda.synchronize()
time.sleep(0.4)

# bigger than the largest hole, smaller than the free total, so only the layout can explain it
captured = []
vramxray.watch().on_report = captured.append
try:
    t = torch.empty(largest_hole() + 4 * MiB, dtype=torch.uint8, device="cuda")
    print("no OOM, rerun")
    del t
except torch.OutOfMemoryError:
    pass

rep = captured[-1]
rows = vramxray.history().for_device(0)
out = os.path.expanduser("~/vramxray-demo.html")
rep.write_html(out, rows)
print("PAGE", out)
print(
    "verdict",
    rep.explanation.verdict,
    "| segments",
    len(rep.explanation.segments),
    "| pins",
    len(rep.explanation.pins),
    "| libs",
    list(rep.accounting.libs),
    "| rows",
    len(rows),
)
dist.destroy_process_group()
