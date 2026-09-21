"""Needs a CUDA device. Runs the uncaught-OOM reproducer under the real hook."""

import pytest

from .helpers import run_script

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)


@pytest.mark.gpu
def test_live_oom_prints_report_before_traceback(tmp_path):
    proc = run_script("oom_live.py", cwd=tmp_path)
    err = proc.stderr
    assert proc.returncode != 0 and "OutOfMemoryError" in err
    assert err.index("vramxray: OOM on cuda:0") < err.index("Traceback")
    assert "verdict: fragmentation" in err
    assert "oom_live.py:" in err, "the pinning block should be attributed to the script"
    assert list(tmp_path.glob("vramxray-oom-*.json")), "report json was not written"


@pytest.mark.gpu
def test_report_without_oom():
    import vramxray

    vramxray.watch()
    x = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    r = vramxray.report()
    assert r.accounting is not None and r.accounting.torch_reserved >= x.numel()
    assert "torch reserved" in str(r)
