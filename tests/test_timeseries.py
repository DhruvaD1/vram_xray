"""Growth maths and the warning threshold, with no GPU involved."""

from vramxray.nvml import MemoryInfo
from vramxray.snapshot import GiB, MiB
from vramxray.timeseries import History, Row


class FakeNVML:
    """Stands in for the driver so the threshold logic can be tested on any machine."""

    def __init__(self, total=16 * GiB):
        self.total = total

    def memory(self, device, uuid=None):
        return MemoryInfo(total=self.total, used=0, free=self.total)


def _history(**kw) -> History:
    return History(FakeNVML(), {0: None}, None, **kw)


def test_growth_ranks_sites_by_rate():
    h = _history()
    h.site_rows.append((0.0, 0, {"train.py:10": 100 * MiB, "steady.py:4": 50 * MiB}))
    h.site_rows.append((120.0, 0, {"train.py:10": 500 * MiB, "steady.py:4": 50 * MiB}))
    grown = h.growth(0)
    assert [g.where for g in grown] == ["train.py:10"]
    assert round(grown[0].bytes_per_minute / MiB) == 200
    assert grown[0].bytes_now == 500 * MiB and grown[0].over == "2 min"


def test_growth_needs_two_samples():
    h = _history()
    h.site_rows.append((0.0, 0, {"train.py:10": 100 * MiB}))
    assert h.growth(0) == []


def test_growth_ignores_sites_below_the_rate_floor():
    h = _history()
    h.site_rows.append((0.0, 0, {"slow.py:1": 0}))
    h.site_rows.append((600.0, 0, {"slow.py:1": 1 * MiB}))
    assert h.growth(0) == []


def test_absolute_threshold_fires_once_per_crossing():
    fired = []
    h = _history(warn_at=700 * MiB, on_warn=lambda d, share: fired.append(share))
    h._check_threshold(0, Row(t=0, device=0, reserved=600 * MiB))
    assert fired == []
    h._check_threshold(0, Row(t=1, device=0, reserved=800 * MiB))
    h._check_threshold(0, Row(t=2, device=0, reserved=900 * MiB))
    assert len(fired) == 1, "one warning per crossing, not one per sample"
    h._check_threshold(0, Row(t=3, device=0, reserved=100 * MiB))  # dropped back down
    h._check_threshold(0, Row(t=4, device=0, reserved=800 * MiB))
    assert len(fired) == 2, "crossing again should warn again"


def test_fractional_threshold_is_a_share_of_the_device():
    fired = []
    h = _history(warn_at=0.5, on_warn=lambda d, share: fired.append(share))
    h._check_threshold(0, Row(t=0, device=0, nvml_used=4 * GiB))
    assert fired == []
    h._check_threshold(0, Row(t=1, device=0, nvml_used=9 * GiB))
    assert len(fired) == 1 and 0.5 < fired[0] < 0.6
