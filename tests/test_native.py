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


@pytest.mark.gpu
def test_mirror_matches_torch_and_the_snapshot(tmp_path):
    """The mirror is rebuilt from events, so it has to agree with what torch reports."""
    proc = run_script("mirror_check.py", cwd=tmp_path)
    out = _out(proc)
    assert proc.returncode == 0, quiet(out)[-2000:]
    assert int(out.split("WORST_RESERVED ")[1].split()[0]) == 0, "reserved drifted from torch"
    # a hole can be off by the rounding slack torch leaves inside a block, but no more
    assert int(out.split("WORST_HOLE ")[1].split()[0]) <= (1 << 20), "hole size drifted too far"


@pytest.mark.gpu
def test_graph_private_pools_are_attributed(tmp_path):
    """Memory captured into a CUDA graph pool never returns to the general pool, so say so."""
    proc = run_script("graph_pools.py", cwd=tmp_path)
    out = _out(proc)
    assert proc.returncode == 0, quiet(out)[-2000:]
    assert "POOLS_BEFORE 0" in out, "no graph has been captured yet"
    after = int(out.split("POOLS_AFTER ")[1].split()[0])
    held = int(out.split("BYTES ")[1].split()[0])
    assert after >= 1 and held > 0, "the captured graph should hold memory in its own pool"
    assert "REPORT_LINE True" in out


@pytest.mark.gpu
def test_regions_attribute_memory_to_a_phase(tmp_path):
    proc = run_script("regions.py", cwd=tmp_path)
    out = _out(proc)
    assert proc.returncode == 0, quiet(out)[-2000:]
    found = {
        line.split()[1]: int(line.split()[2])
        for line in out.splitlines()
        if line.startswith("REGION")
    }
    assert {"forward", "backward", "optimizer"} <= set(found), found
    # the optimizer keeps its state between steps, so it should hold the most
    assert found["optimizer"] > found["forward"]
    assert "IN_REPORT True" in out
