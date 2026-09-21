import json

from vramxray.cli import main
from vramxray.snapshot import GiB, MiB


def _report(rank, nccl, torch_reserved=2 * GiB, verdict="fragmentation"):
    return {
        "rank": rank,
        "request": 512 * MiB,
        "explanation": {"verdict": verdict},
        "accounting": {
            "torch_reserved": torch_reserved,
            "libs": {"libnccl.so.2": nccl, "libcublas.so.12": 8 * MiB},
            "unattributed": 100 * MiB,
        },
    }


def test_merge_prints_table_and_flags_nccl_skew(tmp_path, capsys):
    paths = []
    for rank, nccl in [(0, 1 * GiB), (1, 1 * GiB), (2, 4 * GiB), (3, 1 * GiB)]:
        p = tmp_path / f"vramxray-oom-rank{rank}.json"
        p.write_text(json.dumps(_report(rank, nccl)))
        paths.append(str(p))
    assert main(["merge", *paths]) == 0
    out = capsys.readouterr().out
    assert out.count("fragmentation") == 4
    assert "NCCL skew: rank(s) 2" in out
