"""After handing CUPTI back, torch.profiler must see GPU events again."""

import torch
from torch.profiler import ProfilerActivity, profile

import vramxray

vramxray.watch(mode="native")
x = torch.randn(1024, 1024, device="cuda")
(x @ x).sum().item()

vramxray.release_cupti()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    (x @ x).sum().item()

gpu = [e for e in prof.key_averages() if getattr(e, "self_device_time_total", 0) > 0]
print("profiler GPU events after release:", len(gpu))
assert gpu, "profiler saw no GPU events after release_cupti()"

print("report still works, libs:", dict(vramxray.report().accounting.libs))
