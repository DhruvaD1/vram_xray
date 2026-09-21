"""Six threads allocating at once while the tracker and the CUPTI callbacks are live."""

import threading

import torch

import vramxray

vramxray.watch(mode="native", stacks="python")


def work():
    for _ in range(300):
        t = torch.randn(256, 256, device="cuda")
        (t @ t).sum().item()


threads = [threading.Thread(target=work) for _ in range(6)]
for t in threads:
    t.start()
for t in threads:
    t.join()

events = len(vramxray.timeline())
print("events", events)
assert events > 1000
assert vramxray.report().accounting.torch_reserved > 0
