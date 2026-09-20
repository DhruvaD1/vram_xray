from vramxray.accounting import reconcile
from vramxray.nvml import MemoryInfo, ProcessInfo
from vramxray.snapshot import GiB, MiB


def test_reconcile_sizes_the_gap():
    nvml = MemoryInfo(total=16 * GiB, used=9 * GiB, free=7 * GiB)
    acc = reconcile(nvml, torch_reserved=2 * GiB, torch_allocated=1 * GiB, baseline=300 * MiB)
    assert acc.unattributed == 9 * GiB - 2 * GiB - 300 * MiB
    assert acc.context_known and not acc.overcommitted


def test_unknown_context_and_unknown_process_sizes_do_not_break_the_sum():
    nvml = MemoryInfo(total=16 * GiB, used=9 * GiB, free=7 * GiB)
    procs = [ProcessInfo(pid=1, used=None), ProcessInfo(pid=2, used=1 * GiB)]
    acc = reconcile(nvml, 2 * GiB, 1 * GiB, baseline=-1, other_processes=procs)
    assert not acc.context_known
    assert acc.other_total == 1 * GiB
    assert acc.unattributed == 6 * GiB


def test_libs_reduce_unattributed_and_overcommit_is_flagged():
    nvml = MemoryInfo(total=16 * GiB, used=20 * GiB, free=0)
    acc = reconcile(nvml, 12 * GiB, 10 * GiB, baseline=0, libs={"libnccl.so.2": 4 * GiB})
    assert acc.unattributed == 4 * GiB and acc.overcommitted


def test_no_nvml_means_no_gap_claims():
    acc = reconcile(None, 2 * GiB, 1 * GiB, baseline=0)
    assert acc.unattributed == 0 and acc.nvml is None
