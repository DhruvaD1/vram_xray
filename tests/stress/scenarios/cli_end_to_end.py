"""Every command against real files: a live OOM report, then analyze, merge and run."""

import glob
import json
import subprocess
import sys

import torch
from _common import MiB, cap

import vramxray

cap(1024)
reports = []
vramxray.watch(stacks="python", on_report=reports.append, quiet=True)
try:
    _keep = [torch.empty(700 * MiB, dtype=torch.uint8, device="cuda") for _ in range(3)]
except torch.OutOfMemoryError:
    pass

written = glob.glob("vramxray-oom-*.json")
assert written, "no report file was written"
doc = json.load(open(written[0]))
assert doc["explanation"]["verdict"] and doc["accounting"]["torch_reserved"] > 0
assert doc["explanation"]["sites"], "live sites missing from the json"
assert doc["explanation"]["alloc_conf_var"].endswith("ALLOC_CONF")

torch.cuda.memory._dump_snapshot("snap.pickle")


def cli(*args):
    p = subprocess.run(
        [sys.executable, "-m", "vramxray.cli", *args], capture_output=True, text=True
    )
    assert p.returncode == 0, p.stderr[-1500:]
    return p.stdout


out = cli("analyze", "snap.pickle", "--json", "an.json")
assert "verdict:" in out and json.load(open("an.json"))
assert "largest hole" in cli("analyze", "snap.pickle", "--request", "1.5G")

for rank in range(3):
    doc["rank"] = rank
    doc["accounting"]["libs"] = {"libnccl.so.2": (4 if rank == 2 else 1) << 30}
    json.dump(doc, open(f"r{rank}.json", "w"))
merged = cli("merge", "r0.json", "r1.json", "r2.json")
assert "NCCL skew" in merged and "rank" in merged

script = "import torch; print('SCRIPT OK', torch.ones(8, device='cuda').sum().item())"
open("script.py", "w").write(script)
assert "SCRIPT OK" in cli("run", "script.py")
print("analyze, run and merge all work on real files")
