from vramxray.accounting import reconcile
from vramxray.frag import explain
from vramxray.nvml import MemoryInfo, ProcessInfo
from vramxray.snapshot import ACTIVE, INACTIVE, GiB, MiB
from vramxray.suggest import suggest

from .test_frag import snap


def _frag():
    s = snap(
        [
            (
                512 * MiB,
                0,
                [(500 * MiB, INACTIVE, []), (4 * MiB, ACTIVE, ["pin.py"]), (8 * MiB, INACTIVE, [])],
            )
        ]
    )
    return explain(s, request=505 * MiB, stream=0, device_free=0)


def test_fragmentation_gets_expandable_and_pin_advice():
    texts = [s.text for s in suggest(_frag(), None)]
    assert any("expandable_segments" in t for t in texts)
    assert any("pin.py" in t and "512.0 MiB hole" in t for t in texts)


def test_process_cap_and_other_processes_are_mentioned():
    nvml = MemoryInfo(total=16 * GiB, used=9 * GiB, free=7 * GiB)
    acc = reconcile(
        nvml,
        2 * GiB,
        1 * GiB,
        0,
        other_processes=[ProcessInfo(31877, 1 * GiB)],
        allowed_max=2 * GiB,
    )
    texts = [s.text for s in suggest(_frag(), acc)]
    assert any("capped at 2.00 GiB" in t for t in texts)
    assert any("pid 31877" in t for t in texts)


def test_unattributed_gap_points_at_native_extra():
    nvml = MemoryInfo(total=16 * GiB, used=9 * GiB, free=7 * GiB)
    acc = reconcile(nvml, 2 * GiB, 1 * GiB, 0)
    assert any("vramxray[native]" in s.text for s in suggest(_frag(), acc))
