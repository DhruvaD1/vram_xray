"""A classic leak: every step's output is collected in a list and never moved off the GPU.

Nothing about any single allocation looks wrong, and the process only dies much later.

Run: python tests/repro/leak.py
"""

import time

import torch
import torch.nn as nn

import vramxray

vramxray.watch(stacks="python", mode="native", interval_ms=100, track_sites=True)

model = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 2048)).cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(128, 2048, device="cuda")

# long enough for several site samples, since naming call sites is deliberately infrequent
predictions = []
for _ in range(220):
    out = model(x)
    out.square().mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    predictions.append(out.detach())  # the leak: 1 MiB a step, never moved to the host
    time.sleep(0.2)

torch.cuda.synchronize()
time.sleep(1.0)

print("site samples:", len(vramxray.history().site_rows))
for g in vramxray.growth():
    print(f"GROWTH {g.bytes_per_minute / (1 << 20):.1f} MiB/min over {g.seconds:.0f}s  {g.where}")
print(vramxray.report())
