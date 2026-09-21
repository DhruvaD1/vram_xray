"""Shared bits for the tests that need a real process with a GPU in it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPRO = Path(__file__).parent / "repro"

# torch prints these on import in this environment and they drown out the real output
NOISE = ("UserWarning", "functional_tensor", "cpu = _conversion", "warnings.warn")


def run_script(name: str, *args: str, cwd: Path | str, timeout: int = 900):
    """Run one of the repro scripts in its own process and hand back the result."""
    return subprocess.run(
        [sys.executable, str(REPRO / name), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def quiet(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not any(n in ln for n in NOISE))
