"""With expandable segments on we must notice, and must not suggest turning them on."""

import os

# torch 2.9 warns that the CUDA name is deprecated but still ignores the new one, so set both
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch  # noqa: E402
from _common import MiB, cap  # noqa: E402

import vramxray  # noqa: E402

cap(2048)
reports = []
vramxray.watch(stacks="python", on_report=reports.append, quiet=True)

filler = []
try:
    while True:
        filler.append(torch.empty(256 * MiB, dtype=torch.uint8, device="cuda"))
except torch.OutOfMemoryError:
    pass
del filler[2], filler[3]  # holes in the middle, neighbours still live
torch.cuda.empty_cache()

r = vramxray.report()
snap = torch.cuda.memory._snapshot()
in_snapshot = any(s.get("is_expandable") for s in snap["segments"])
print("is_expandable in snapshot:", in_snapshot, "| analyzer saw it:", r.explanation.expandable_on)
print("conf var:", r.explanation.alloc_conf_var)
assert in_snapshot and r.explanation.expandable_on
# match the setting, not the word, because the call sites below name this very file
setting = f"{r.explanation.alloc_conf_var}=expandable_segments"
assert not any(setting in s.text for s in r.suggestions), "already on, do not suggest"

try:
    torch.empty(400 * MiB, dtype=torch.uint8, device="cuda")
    print("a request bigger than any one hole succeeded, which is the point of expandable segments")
except torch.OutOfMemoryError:
    print("OOM verdict:", reports[-1].explanation.verdict)
