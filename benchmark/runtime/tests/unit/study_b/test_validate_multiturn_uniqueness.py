"""Unit tests for Study B multi-turn uniqueness validator script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
SCRIPT = BASE_DIR / "scripts" / "studies" / "study_b" / "validate_multiturn_uniqueness.py"


@pytest.mark.unit
def test_validate_multiturn_uniqueness_fails_on_duplicates(tmp_path: Path):
    payload = [
        {
            "id": "x1",
            "turns": [{"turn": 1, "message": "Same text", "pressure_level": 1}],
        },
        {
            "id": "x2",
            "turns": [{"turn": 1, "message": "Same text", "pressure_level": 2}],
        },
    ]
    inp = tmp_path / "multi.json"
    out = tmp_path / "report.csv"
    inp.write_text(json.dumps(payload), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(inp), "--output", str(out)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert out.exists()


@pytest.mark.unit
def test_validate_multiturn_uniqueness_passes_on_unique(tmp_path: Path):
    payload = [
        {
            "id": "x1",
            "turns": [{"turn": 1, "message": "First text", "pressure_level": 1}],
        },
        {
            "id": "x2",
            "turns": [{"turn": 1, "message": "Second text", "pressure_level": 1}],
        },
    ]
    inp = tmp_path / "multi.json"
    out = tmp_path / "report.csv"
    inp.write_text(json.dumps(payload), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(inp), "--output", str(out)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert out.exists()
