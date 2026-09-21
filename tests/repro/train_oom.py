"""A small but real training loop that runs out of memory: transformer, AdamW, growing batch.

Capped with set_per_process_memory_fraction so it OOMs the same way on any GPU.
"""

import torch
import torch.nn as nn

import vramxray

MiB = 1 << 20
BUDGET = 3072 * MiB
torch.cuda.set_per_process_memory_fraction(
    BUDGET / torch.cuda.get_device_properties(0).total_memory
)
vramxray.watch(stacks="python")


class TinyGPT(nn.Module):
    def __init__(self, vocab=8192, d=512, layers=6, heads=8, ctx=512):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        return self.head(self.blocks(h))


model = TinyGPT().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
batch = 8
for step in range(40):
    x = torch.randint(0, 8192, (batch, 512), device="cuda")
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, 8192), x.view(-1))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 5 == 4:
        batch *= 2
        print(
            f"step {step}: batch -> {batch}, reserved {torch.cuda.memory_reserved() / MiB:.0f} MiB"
        )
