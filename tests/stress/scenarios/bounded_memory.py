"""Host memory must settle when the tracker is left on, not creep for the life of the run."""

import subprocess
import sys
from pathlib import Path

repro = Path(__file__).resolve().parents[2] / "repro" / "long_run.py"
proc = subprocess.run([sys.executable, str(repro)], capture_output=True, text=True, timeout=1800)
out = proc.stdout + proc.stderr
assert proc.returncode == 0, out[-2000:]


def value(key: str) -> float:
    return float(out.split(key + " ")[1].split()[0])


first, second = value("PHASE1"), value("PHASE2")
print(f"first 30k steps +{first:.0f} MB, next 30k steps +{second:.0f} MB")
assert second < first * 0.6, "host memory is still climbing once the buffers are full"
assert second < 80, f"{second:.0f} MB in a settled phase is too much"
