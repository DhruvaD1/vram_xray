"""Native mode with NCCL and cuBLAS loaded, so the report has libraries to name.

Prints markers the test greps for. Run: python tests/repro/native_report.py
"""

import os

import torch
import torch.distributed as dist

import vramxray

w = vramxray.watch(mode="native", stacks="python")
print("CUPTI_RC", w.cupti_rc)

a = torch.randn(2048, 2048, device="cuda")
(a @ a).sum().item()

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29519")
dist.init_process_group("nccl", rank=0, world_size=1, device_id=torch.device("cuda:0"))
t = torch.ones(1024, device="cuda")
dist.all_reduce(t)
torch.cuda.synchronize()

r = vramxray.report()
print(r)
print("LIBS", sorted(r.accounting.libs))

tl = vramxray.timeline()
print("TIMELINE", len(tl), sorted(set(tl.columns["action"])))
with_stack = [s for s in tl.columns["stack"] if s]
print("STACKS", len(with_stack), "e.g.", with_stack[0] if with_stack else "")

dist.destroy_process_group()
