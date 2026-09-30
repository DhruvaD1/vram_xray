"""Call site tracking should need no snapshot, no opt in, and no Python on the sampler thread."""

import time

import torch
from _common import MiB

import vramxray

w = vramxray.watch(mode="native", stacks="python", interval_ms=100)

keep = []
for _ in range(20):
    keep.append(torch.empty(32 * MiB, dtype=torch.uint8, device="cuda"))
    time.sleep(0.25)
torch.cuda.synchronize()
time.sleep(0.3)

series = w.history.site_series(0)
print("samples", len(series))
assert len(series) >= 2, "the sampler did not run"

grown = w.history.growth(0)
print("growth", [(g.where, round(g.bytes_per_minute / MiB)) for g in grown][:2])
assert grown, "a site that grew every step should be detected"
assert "native_sites.py:" in grown[0].where

# the same numbers must agree with the snapshot the analyzer would have taken
from vramxray.frag import live_sites  # noqa: E402
from vramxray.snapshot import from_dict  # noqa: E402

snap = live_sites(from_dict(torch.cuda.memory._snapshot(0)).for_device(0), top=1)
native = w.native.mirror_top_sites(0, 1, 512)
print("top site snapshot", snap[0].bytes, "native", native[0]["bytes"])
assert snap[0].bytes == native[0]["bytes"], "native attribution drifted from the snapshot"
