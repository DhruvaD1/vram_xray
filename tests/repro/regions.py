"""Attribute memory to the phase of the step that allocated it.

Run: python tests/repro/regions.py
"""

import torch
import torch.nn as nn

import vramxray

vramxray.watch(mode="native", stacks="python")

model = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 2048)).cuda()
opt = torch.optim.AdamW(model.parameters())
x = torch.randn(256, 2048, device="cuda")

for _ in range(3):
    with vramxray.region("forward"):
        loss = model(x).square().mean()
    with vramxray.region("backward"):
        loss.backward()
    with vramxray.region("optimizer"):
        opt.step()
    opt.zero_grad(set_to_none=False)
torch.cuda.synchronize()

for r in vramxray.regions.live(0):
    print(f"REGION {r['name']} {r['bytes']} {r['blocks']}")
print("IN_REPORT", "live memory by region" in str(vramxray.report()))
