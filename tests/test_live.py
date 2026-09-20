"""Needs a CUDA device. Runs the uncaught-OOM reproducer under the real hook."""

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

REPRO = Path(__file__).parent / "repro" / "oom_live.py"


@pytest.mark.gpu
def test_live_oom_prints_report_before_traceback(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(REPRO)], cwd=tmp_path, capture_output=True, text=True, timeout=300
    )
    err = proc.stderr
    assert proc.returncode != 0 and "OutOfMemoryError" in err
    assert err.index("vramxray: OOM on cuda:0") < err.index("Traceback")
    assert "verdict: fragmentation" in err and "oom_live.py:19" in err
    assert list(tmp_path.glob("vramxray-oom-*.json")), "report json was not written"


@pytest.mark.gpu
def test_report_without_oom():
    import vramxray

    vramxray.watch()
    x = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    r = vramxray.report()
    assert r.accounting is not None and r.accounting.torch_reserved >= x.numel()
    assert "torch reserved" in str(r)
