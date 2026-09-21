"""The memory history should record a peak that is higher than where the run ends."""

import time

import torch
from _common import MiB

import vramxray

vramxray.watch(mode="native", interval_ms=25)

x = torch.randn(1024, 1024, device="cuda")
for _ in range(20):
    (x @ x).sum().item()
    time.sleep(0.01)

hog = [torch.empty(128 * MiB, dtype=torch.uint8, device="cuda") for _ in range(4)]
time.sleep(0.4)
del hog
torch.cuda.empty_cache()
time.sleep(0.4)

h = vramxray.history()
print("rows", len(h))
assert len(h) > 5, "the sampler did not run"

peak = h.peak(0)
now = h.for_device(0)[-1]
print(f"peak {peak.reserved} at t={peak.t:.1f}s, now {now.reserved}")
assert peak.reserved >= 4 * 128 * MiB, "the peak missed the big allocation"
assert peak.reserved > now.reserved, "peak should be above where we ended"
assert peak.largest_free >= 0, "native mode should know the largest free block"

# calling the public helpers twice must keep working, the modules must not shadow them
assert len(vramxray.history()) > 0 and vramxray.report() is not None
print("stalls", {k: round(v["seconds"] * 1000, 1) for k, v in vramxray.stalls().items()})
