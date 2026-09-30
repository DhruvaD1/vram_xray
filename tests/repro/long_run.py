"""Leave the tracker on for a long run and check that host memory settles.

The allocation stacks torch gathers are held until something drains them, so a run that never
asks for a timeline used to keep every one of them alive for ever. Both buffers are capped now,
which should show up as a second phase that grows far less than the first.

Run: python tests/repro/long_run.py
"""

import resource

import torch
import torch.nn as nn

import vramxray

w = vramxray.watch(mode="native", stacks="python", interval_ms=50)

model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)).cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(64, 512, device="cuda")


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def steps(n: int) -> None:
    for _ in range(n):
        model(x).square().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


steps(2000)  # settle, so the baseline is not just warm up
base = rss_mb()
steps(30000)
first = rss_mb()
steps(30000)
second = rss_mb()

print(f"RSS_BASE {base:.0f}")
print(f"PHASE1 {first - base:.0f}")
print(f"PHASE2 {second - first:.0f}")
print("EVENTS", w.native.stats()["events"])
