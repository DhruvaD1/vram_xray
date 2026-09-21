"""watch() called once CUDA is already up. The context is not measurable, and we should say so."""

import torch

import vramxray

x = torch.randn(64, device="cuda")
x.sum().item()

w = vramxray.watch(mode="native")
print("cupti rc", w.cupti_rc, "context known:", w.baseline.get(0, -1) >= 0)
assert w.cupti_rc == 0

text = str(vramxray.report())
print("outside torch labelled:", "outside torch" in text or "unattributed" in text)
assert "outside torch" in text or "unattributed" in text
