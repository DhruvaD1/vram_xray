"""Needs a CUDA device and the native core."""

import pytest

from .helpers import quiet, run_script

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)


@pytest.mark.gpu
def test_stream_checks(tmp_path):
    proc = run_script("stream_checks.py", cwd=tmp_path)
    out = proc.stdout + proc.stderr
    if "native build failed" in out or "native mode requested" in out:
        pytest.skip("native extension did not build here")
    assert proc.returncode == 0, quiet(out)[-3000:]
    # a capture whose side stream joins through an event is normal and must stay silent
    assert "CLEAN 0" in out, quiet(out)
    assert "blocking_stream" in out and "launch_outside_capture" in out
