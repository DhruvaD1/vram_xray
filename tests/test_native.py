"""Needs a CUDA device, a C++ compiler, and the CUDA headers. Skips cleanly otherwise."""

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

SCRIPT = r"""
import os, sys, torch, vramxray
w = vramxray.watch(mode="native", stacks="python")
print("CUPTI_RC", w.cupti_rc)
a = torch.randn(2048, 2048, device="cuda"); b = a @ a; torch.cuda.synchronize()
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29519")
import torch.distributed as dist
dist.init_process_group("nccl", rank=0, world_size=1, device_id=torch.device("cuda:0"))
t = torch.ones(1024, device="cuda"); dist.all_reduce(t); torch.cuda.synchronize()
r = vramxray.report()
print(r)
print("LIBS", sorted(r.accounting.libs))
tl = vramxray.timeline()
print("TIMELINE", len(tl), sorted(set(tl.columns["action"])))
dist.destroy_process_group()
"""


def _run(code: str, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True, timeout=900
    )


@pytest.mark.gpu
def test_native_report_names_libraries(tmp_path):
    proc = _run(SCRIPT, tmp_path)
    out = proc.stdout + proc.stderr
    if "native build failed" in out or "native mode requested" in out:
        pytest.skip("native extension did not build here")
    assert proc.returncode == 0, out[-2000:]
    assert "CUPTI_RC 0" in out
    assert "libnccl.so.2" in out and "libcublas" in out
    assert (
        "libc10_cuda" not in out.split("LIBS")[1].splitlines()[0]
    )  # torch's own rows are folded into reserved
    # kernel images are an NVML delta around module loads, so they can legitimately be zero here
    assert "TIMELINE" in out and "driver_create" in out and "segment_alloc" in out


@pytest.mark.gpu
def test_native_oom_report_still_prints(tmp_path):
    repro = Path(__file__).parent / "repro" / "oom_live.py"
    code = repro.read_text().replace(
        'vramxray.watch(stacks="python")', 'vramxray.watch(stacks="python", mode="native")'
    )
    proc = _run(code, tmp_path)
    err = proc.stderr
    if "native build failed" in err:
        pytest.skip("native extension did not build here")
    assert "vramxray: OOM on cuda:0" in err and err.index("vramxray: OOM") < err.index("Traceback")
    assert "verdict: fragmentation" in err
