"""Memory captured into a CUDA graph's private pool, which the general pool never gets back.

Run: python tests/repro/graph_pools.py
"""

import torch

import vramxray

w = vramxray.watch(mode="native", stacks="python")

x = torch.randn(1024, 1024, device="cuda")
(x @ x).relu().sum().item()  # warm cuBLAS up before capturing
torch.cuda.synchronize()

print("POOLS_BEFORE", len(w.native.mirror_pools(0)))

graph = torch.cuda.CUDAGraph()
static = torch.randn(1024, 1024, device="cuda")
with torch.cuda.graph(graph):
    out = (static @ static).relu()
torch.cuda.synchronize()

pools = w.native.mirror_pools(0)
print("POOLS_AFTER", len(pools), "BYTES", sum(p["bytes"] for p in pools))
print("REPORT_LINE", "CUDA graph pools" in str(vramxray.report()))
