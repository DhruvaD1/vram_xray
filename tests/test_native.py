"""Needs a CUDA device, a C++ compiler, and the CUDA headers. Skips cleanly otherwise."""

import pytest

from .helpers import quiet, run_script

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)


def _out(proc) -> str:
    text = proc.stdout + proc.stderr
    if "native build failed" in text or "native mode requested" in text:
        pytest.skip("native extension did not build here")
    return text


@pytest.mark.gpu
def test_native_report_names_libraries(tmp_path):
    proc = run_script("native_report.py", cwd=tmp_path)
    out = _out(proc)
    assert proc.returncode == 0, quiet(out)[-2000:]
    assert "CUPTI_RC 0" in out
    assert "libnccl.so.2" in out and "libcublas" in out
    # torch's own allocations are already counted as reserved, so they must not appear again
    assert "libc10_cuda" not in out.split("LIBS")[1].splitlines()[0]
    assert "TIMELINE" in out and "driver_create" in out and "segment_alloc" in out
    assert int(out.split("STACKS ")[1].split()[0]) > 0, "allocation stacks were not symbolized"


@pytest.mark.gpu
def test_native_oom_report_still_prints(tmp_path):
    proc = run_script("oom_live.py", "native", cwd=tmp_path)
    err = _out(proc)
    assert "vramxray: OOM on cuda:0" in err
    assert err.index("vramxray: OOM") < err.index("Traceback")
    assert "verdict: fragmentation" in err
