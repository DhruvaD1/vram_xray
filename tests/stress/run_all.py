"""Deeper checks that take longer than the unit suite.

Every file in scenarios/ is one scenario. Each runs as its own process in a fresh temp
directory, so an OOM or a broken CUDA context in one cannot affect the next.

Run: python tests/stress/run_all.py [name ...]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCENARIOS = Path(__file__).parent / "scenarios"
NOISE = ("Warning", "warn", "cpu = _conversion")


def run_one(path: Path) -> bool:
    started = time.time()
    with tempfile.TemporaryDirectory() as work:
        p = subprocess.run(
            [sys.executable, str(path)], cwd=work, capture_output=True, text=True, timeout=1800
        )
    ok = p.returncode == 0
    print(f"[{'PASS' if ok else 'FAIL'}] {path.stem} ({time.time() - started:.0f}s)")
    lines = [
        ln
        for ln in (p.stdout + p.stderr).strip().splitlines()
        if ln and not any(n in ln for n in NOISE)
    ]
    for ln in lines[-4:]:
        print(f"      {ln[:160]}")
    return ok


def main(argv: list[str]) -> int:
    wanted = set(argv)
    files = sorted(f for f in SCENARIOS.glob("*.py") if not f.name.startswith("_"))
    if wanted:
        files = [f for f in files if f.stem in wanted]
        missing = wanted - {f.stem for f in files}
        if missing:
            print(f"no such scenario: {', '.join(sorted(missing))}")
            return 2
    return sum(not run_one(f) for f in files)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
