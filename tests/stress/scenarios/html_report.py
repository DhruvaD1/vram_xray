"""The OOM path should leave a page next to the json, and the page must stand alone."""

import glob

import torch
from _common import MiB, cap

import vramxray

cap(1024)
vramxray.watch(stacks="python", quiet=True)
try:
    _keep = [torch.empty(700 * MiB, dtype=torch.uint8, device="cuda") for _ in range(3)]
except torch.OutOfMemoryError:
    pass

pages = glob.glob("vramxray-oom-*.html")
assert pages, "no page was written next to the json"
assert glob.glob("vramxray-oom-*.json"), "the json should still be written"

page = open(pages[0]).read()
print("page bytes", len(page), "| segments", page.count('class="strip"'))
assert "http://" not in page and "https://" not in page, "the page must not fetch anything"
assert 'class="strip"' in page, "the segment map is missing"
assert "verdict" in page
