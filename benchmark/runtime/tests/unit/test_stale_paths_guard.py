from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_stale_path_guard_passes() -> None:
    runtime_root = Path(__file__).resolve().parents[2]
    guard_script = runtime_root / "scripts" / "ci" / "check_stale_paths.py"

    completed = subprocess.run(
        [sys.executable, str(guard_script)],
        cwd=str(runtime_root),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
