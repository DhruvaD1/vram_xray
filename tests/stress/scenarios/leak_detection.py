"""A leak that only shows up as a trend: collected outputs are never moved off the GPU."""

import subprocess
import sys
from pathlib import Path

repro = Path(__file__).resolve().parents[2] / "repro" / "leak.py"
proc = subprocess.run([sys.executable, str(repro)], capture_output=True, text=True, timeout=600)
out = proc.stdout + proc.stderr
assert proc.returncode == 0, out[-2000:]

growth = [ln for ln in out.splitlines() if ln.startswith("GROWTH")]
print(growth[0] if growth else "no growth detected")
assert growth, "the leak was not detected"
assert "leak.py:" in growth[0], "the growing site should be named with a line number"
rate = float(growth[0].split()[1])
assert rate > 50, f"expected a clear climb, got {rate} MiB/min"
print("report mentions it:", "growing over the run" in out)
assert "growing over the run" in out
