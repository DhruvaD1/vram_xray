from pathlib import Path

from vramxray import explain, load
from vramxray.cli import main
from vramxray.snapshot import MiB

DUMP = Path(__file__).parent / "snapshots" / "split_remainder.pickle"


def test_real_dump_is_explained():
    snap = load(DUMP)
    assert snap.devices() == [0]
    ooms = snap.ooms(0)
    assert len(ooms) == 1 and ooms[0].size == 505 * MiB and ooms[0].addr > 0
    ex = explain(snap)
    assert ex.verdict == "fragmentation"
    assert ex.largest_hole == 500 * MiB and ex.free_in_segments == 508 * MiB
    pin = ex.pins[0]
    assert pin.block.size == 4 * MiB and pin.wasted == 508 * MiB
    assert pin.frames[0].filename.endswith("split_remainder.py") and pin.frames[0].line == 19


def test_cli_analyze_runs(capsys, tmp_path):
    out = tmp_path / "r.json"
    assert main(["analyze", str(DUMP), "--top", "2", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "verdict: fragmentation" in text and "expandable_segments" in text and out.exists()
