"""Checks for generated `v6.1` review, verification, and manual-review outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
VERIFY_ROOT = BASE_DIR / "data" / "verification" / "v6_1"
SNAPSHOT_ROOT = BASE_DIR / "data" / "frozen_splits" / "v6_1"


def _read_json(path: Path):
    assert path.exists(), f"Missing file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_ssv(path: Path) -> list[dict[str, str]]:
    assert path.exists(), f"Missing file: {path}"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


@pytest.mark.unit
def test_v6_1_cross_study_review_summary_is_clean() -> None:
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
def test_v6_1_verdict_files_have_expected_row_counts() -> None:
    assert len(_read_ssv(VERIFY_ROOT / "study_a_reference_verdicts.ssv")) == 2000
    assert len(_read_ssv(VERIFY_ROOT / "study_b_single_verdicts.ssv")) == 2000
    assert len(_read_ssv(VERIFY_ROOT / "study_b_multi_verdicts.ssv")) == 120
    assert len(_read_ssv(VERIFY_ROOT / "study_c_verdicts.ssv")) == 100


@pytest.mark.unit
def test_v6_1_strict_verification_is_clean() -> None:
    payload = _read_json(VERIFY_ROOT / "strict_snapshot_verification.json")
    assert payload["ok"] is True
    assert payload["error_count"] == 0
    assert payload["study_b_multi_turn_provenance_counts"]["direct_source"] > 0
    assert payload["study_b_multi_turn_provenance_counts"]["retrieved_composed"] > 0
    assert payload["study_c_provenance_counts"]["direct_source"] > 0
    assert payload["study_c_provenance_counts"]["retrieved_composed"] > 0
    assert "source_anchored_generated" not in payload["study_b_multi_turn_provenance_counts"]
    assert "source_anchored_generated" not in payload["study_c_provenance_counts"]


@pytest.mark.unit
def test_v6_1_manual_review_queues_exist_and_are_well_formed() -> None:
    manifest = _read_json(SNAPSHOT_ROOT / "manifest.json")
    manual_review = manifest.get("manual_review", {})
    for key in ("study_b_multi_turn_queue", "study_c_queue"):
        queue_path = Path(manual_review[key])
        payload = _read_json(queue_path)
        assert "items" in payload
        for item in payload["items"][:20]:
            assert item.get("case_id")
            assert isinstance(item.get("turn_index"), int)
            assert item.get("reason")
            assert isinstance(item.get("candidates"), list)
