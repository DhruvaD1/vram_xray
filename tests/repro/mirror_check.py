"""Cross check the native allocator mirror against torch's own numbers and a real snapshot.

The mirror is rebuilt from trace events rather than read from torch, so this is the test that
says whether that reconstruction is right. Run: python tests/repro/mirror_check.py
"""

import torch
import torch.nn as nn

import vramxray
from vramxray.frag import explain
from vramxray.snapshot import from_dict

MiB = 1 << 20
w = vramxray.watch(mode="native")
model = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 2048)).cuda()
opt = torch.optim.AdamW(model.parameters())
x = torch.randn(256, 2048, device="cuda")

worst_reserved = 0
worst_hole = 0


def check(tag, keep_alive=None):
    global worst_reserved, worst_hole
    mirror = w.native.mirror_stats(0)
    torch_reserved = torch.cuda.memory_stats()["reserved_bytes.all.current"]
    snap = from_dict(torch.cuda.memory._snapshot(0))
    ex = explain(snap, 0, request=512 * MiB, stream=0, device_free=0)
    worst_reserved = max(worst_reserved, abs(mirror["reserved"] - torch_reserved))
    worst_hole = max(worst_hole, abs(mirror["largest_free"] - ex.largest_hole))
    print(
        f"{tag:18} reserved {mirror['reserved']:>12} vs {torch_reserved:>12} | "
        f"largest_free {mirror['largest_free']:>11} vs {ex.largest_hole:>11}"
    )


check("start")
for _ in range(5):
    model(x).square().mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
torch.cuda.synchronize()
check("after steps")

blocks = [torch.empty(128 * MiB, dtype=torch.uint8, device="cuda") for _ in range(4)]
check("after big blocks")
del blocks[1], blocks[1]
check("with holes")
torch.cuda.empty_cache()
check("after empty_cache")

print("WORST_RESERVED", worst_reserved)
print("WORST_HOLE", worst_hole)
