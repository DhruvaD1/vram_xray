"""What watching costs on a training loop, measured against the same loop unwatched."""

import time

import torch
import torch.nn as nn

import vramxray

model = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 1024)).cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(64, 1024, device="cuda")


def run(steps):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        model(x).square().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t


run(200)
bare = run(2000)

vramxray.watch(mode="native", stacks="python")
vramxray.streams.start(channels=True)
run(200)
watched = run(2000)

print(f"bare {bare:.2f}s watched {watched:.2f}s overhead {100 * (watched / bare - 1):.1f}%")
print("events", len(vramxray.timeline()))
assert watched < bare * 1.4, "overhead above 40% with everything on"
assert vramxray.report().accounting.unattributed >= 0
