"""The event buffer must stay bounded between drains, and empty after one."""

import torch

import vramxray

vramxray.watch(mode="native")
for _ in range(300000):
    torch.empty(16, device="cuda")

native = vramxray._watcher.native
stats = native.stats()
print(stats)
assert stats["events"] <= (1 << 20), "event buffer grew past its cap"

print("drained", len(vramxray.timeline()))
assert native.stats()["events"] == 0
