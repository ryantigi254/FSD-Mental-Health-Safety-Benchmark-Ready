"""Checks for generated `v6` review and provenance outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
VERIFY_ROOT = BASE_DIR / "data" / "verification" / "v6"


def _read_json(path: Path):
    assert path.exists(), f"Missing file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_ssv(path: Path) -> list[dict[str, str]]:
    assert path.exists(), f"Missing file: {path}"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


@pytest.mark.unit
def test_v6_cross_study_review_summary_is_clean() -> None:
    payload = _read_json(VERIFY_ROOT / "run_metadata.json")
    summary = payload.get("summary", {})
    assert summary["study_a"]["acceptable"] == 2000
    assert summary["study_b_single"]["acceptable"] == 2000
    assert summary["study_b_multi"]["acceptable"] == 120
    assert summary["study_c"]["acceptable"] == 100
    assert summary["study_a"]["reject"] == 0
    assert summary["study_b_single"]["reject"] == 0
    assert summary["study_b_multi"]["reject"] == 0
    assert summary["study_c"]["reject"] == 0


@pytest.mark.unit
def test_v6_verdict_files_have_expected_row_counts() -> None:
    assert len(_read_ssv(VERIFY_ROOT / "study_a_reference_verdicts.ssv")) == 2000
    assert len(_read_ssv(VERIFY_ROOT / "study_b_single_verdicts.ssv")) == 2000
    assert len(_read_ssv(VERIFY_ROOT / "study_b_multi_verdicts.ssv")) == 120
    assert len(_read_ssv(VERIFY_ROOT / "study_c_verdicts.ssv")) == 100


@pytest.mark.unit
def test_v6_source_backed_verification_is_clean() -> None:
    payload = _read_json(VERIFY_ROOT / "source_backed_snapshot_verification.json")
    assert payload["ok"] is True
    assert payload["error_count"] == 0
    assert payload["study_b_multi_turn_provenance_counts"]["source"] > 0
    assert payload["study_b_multi_turn_provenance_counts"]["retrieved_composed"] > 0
    assert payload["study_c_provenance_counts"]["source"] > 0
    assert payload["study_c_provenance_counts"]["retrieved_composed"] > 0
