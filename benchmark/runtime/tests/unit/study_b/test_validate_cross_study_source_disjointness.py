"""Unit tests for cross-study source disjointness validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
SCRIPT = BASE_DIR / "scripts" / "studies" / "validate_cross_study_source_disjointness.py"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
def test_cross_study_source_disjointness_fails_on_overlap(tmp_path: Path):
    root = tmp_path / "data"
    _write_json(
        root / "openr1_psy_splits" / "study_a_test.json",
        {
            "samples": [
                {"id": "a_1", "metadata": {"source_split": "test", "source_openr1_ids": [7]}},
            ]
        },
    )
    _write_json(
        root / "openr1_psy_splits" / "study_b_test.json",
        [
            {"id": "b_1", "metadata": {"source_split": "generated", "source_openr1_ids": []}},
        ],
    )
    _write_json(
        root / "openr1_psy_splits" / "study_b_multi_turn_test.json",
        [
            {
                "id": "b_mt_1",
                "turns": [{"turn": 1, "message": "Same text", "pressure_level": 1}],
                "metadata": {"source_split": "test", "source_openr1_ids": [7]},
            }
        ],
    )
    _write_json(
        root / "openr1_psy_splits" / "study_c_test.json",
        {"cases": [{"id": "c_1", "metadata": {"source_split": "train", "source_openr1_ids": [9]}}]},
    )
    _write_json(
        root / "adversarial_bias" / "biased_vignettes.json",
        {"cases": [{"id": "abias_1", "pair_group_id": "pg_1", "metadata": {"source_openr1_split": "test", "source_openr1_id": 7}}]},
    )

    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--data-root", str(root), "--output", str(out)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert out.exists()


@pytest.mark.unit
def test_cross_study_source_disjointness_passes_when_clean(tmp_path: Path):
    root = tmp_path / "data"
    _write_json(
        root / "openr1_psy_splits" / "study_a_test.json",
        {
            "samples": [
                {"id": "a_1", "metadata": {"source_split": "test", "source_openr1_ids": [7]}},
            ]
        },
    )
    _write_json(
        root / "openr1_psy_splits" / "study_b_test.json",
        [
            {"id": "b_1", "metadata": {"source_split": "generated", "source_openr1_ids": []}},
        ],
    )
    _write_json(
        root / "openr1_psy_splits" / "study_b_multi_turn_test.json",
        [
            {
                "id": "b_mt_1",
                "turns": [{"turn": 1, "message": "Unique text", "pressure_level": 1}],
                "metadata": {"source_split": "train", "source_openr1_ids": [17]},
            }
        ],
    )
    _write_json(
        root / "openr1_psy_splits" / "study_c_test.json",
        {"cases": [{"id": "c_1", "metadata": {"source_split": "train", "source_openr1_ids": [9]}}]},
    )
    _write_json(
        root / "adversarial_bias" / "biased_vignettes.json",
        {"cases": [{"id": "abias_1", "pair_group_id": "pg_1", "metadata": {"source_openr1_split": "test", "source_openr1_id": 8}}]},
    )

    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--data-root", str(root), "--output", str(out)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert out.exists()


@pytest.mark.unit
def test_cross_study_detects_turn_level_donor_overlap(tmp_path: Path):
    """Validator must catch overlap when metadata looks clean but turn-level IDs overlap."""
    root = tmp_path / "data"
    _write_json(
        root / "openr1_psy_splits" / "study_a_test.json",
        {"samples": [{"id": "a_1", "metadata": {"source_split": "test", "source_openr1_ids": [1]}}]},
    )
    _write_json(
        root / "openr1_psy_splits" / "study_b_test.json",
        [{"id": "b_1", "metadata": {"source_split": "generated", "source_openr1_ids": []}}],
    )
    # B MT case: metadata says ID 99, but turn-level has donor ID 42
    _write_json(
        root / "openr1_psy_splits" / "study_b_multi_turn_test.json",
        [{
            "id": "bmt_1",
            "turns": [
                {"turn": 1, "message": "text", "pressure_level": 0,
                 "source_openr1_split": "test", "source_openr1_id": 42},
            ],
            "metadata": {"source_split": "test", "source_openr1_ids": [99]},
        }],
    )
    # Study C case: metadata says ID 88, but turn-level also has donor ID 42
    _write_json(
        root / "openr1_psy_splits" / "study_c_test.json",
        {"cases": [{
            "id": "c_1",
            "turns": [
                {"turn": 1, "message": "text",
                 "source_openr1_split": "test", "source_openr1_id": 42},
            ],
            "metadata": {"source_split": "test", "source_openr1_ids": [88]},
        }]},
    )
    _write_json(
        root / "adversarial_bias" / "biased_vignettes.json",
        {"cases": [{"id": "abias_1", "pair_group_id": "pg_1",
                     "metadata": {"source_openr1_split": "test", "source_openr1_id": 2}}]},
    )

    out = tmp_path / "report.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--data-root", str(root), "--output", str(out)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, f"Should detect turn-level overlap but got rc=0\nstdout: {proc.stdout}"
