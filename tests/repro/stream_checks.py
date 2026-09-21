"""Three stream situations: a blocking stream, a clean graph capture, and a broken one.

Only the blocking stream and the broken capture should be reported.
Run: python tests/repro/stream_checks.py
"""

import ctypes
import os
import site

import torch

import vramxray
from vramxray import streams

vramxray.watch(mode="native")
streams.start(channels=True)

# a stream created without the non-blocking flag, the way a C extension usually does it
cudart = ctypes.CDLL(
    os.path.join(site.getsitepackages()[0], "nvidia/cuda_runtime/lib/libcudart.so.12")
)
handle = ctypes.c_void_p()
assert cudart.cudaStreamCreate(ctypes.byref(handle)) == 0

x = torch.randn(256, 256, device="cuda")
side = torch.cuda.Stream()
(x @ x).sin().cos().sum().item()  # warm up cuBLAS before capturing anything
torch.cuda.synchronize()

# a clean capture: the side stream joins through an event, so nothing is stranded
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y = x @ x
    joined = torch.cuda.Event()
    joined.record()
    with torch.cuda.stream(side):
        side.wait_event(joined)
        y.sin()
    torch.cuda.current_stream().wait_stream(side)

stranded = [f for f in streams.findings() if f.kind == "launch_outside_capture"]
print("CLEAN", sum(f.count for f in stranded))

# a broken capture: work on a stream that never joined, so it runs now and is not in the graph
other = torch.cuda.Stream()
broken = torch.cuda.CUDAGraph()
with torch.cuda.graph(broken, capture_error_mode="relaxed"):
    y = x @ x
    with torch.cuda.stream(other):
        x.cos()
torch.cuda.synchronize()

print(streams.report())
print("KINDS", sorted({f.kind for f in streams.findings()}))
