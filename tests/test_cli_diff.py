import json

from vramxray.cli import main
from vramxray.snapshot import GiB, MiB


def _report(tmp_path, name, sites, libs):
    doc = {
        "accounting": {"torch_reserved": 2 * GiB, "libs": libs},
        "explanation": {
            "verdict": "exhaustion",
            "sites": [{"where": w, "bytes": b, "count": 1} for w, b in sites.items()],
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return str(path)


def test_diff_reports_what_grew(tmp_path, capsys):
    before = _report(
        tmp_path,
        "a.json",
        {"train.py:10": 100 * MiB, "same.py:1": 8 * MiB},
        {"libnccl.so.2": 1 * GiB},
    )
    after = _report(
        tmp_path,
        "b.json",
        {"train.py:10": 900 * MiB, "same.py:1": 8 * MiB},
        {"libnccl.so.2": 3 * GiB},
    )
    assert main(["diff", before, after]) == 0
    out = capsys.readouterr().out
    assert "train.py:10" in out and "+800.0 MiB" in out
    assert "libnccl.so.2" in out and "+2.00 GiB" in out
    assert "same.py:1" not in out


def test_diff_is_quiet_when_nothing_moved(tmp_path, capsys):
    same = {"train.py:10": 100 * MiB}
    before = _report(tmp_path, "a.json", same, {})
    after = _report(tmp_path, "b.json", same, {})
    assert main(["diff", before, after]) == 0
    out = capsys.readouterr().out
    assert "call site changes" not in out and "library changes" not in out
