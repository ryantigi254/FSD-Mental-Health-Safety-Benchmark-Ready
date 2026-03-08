"""Post-run contract checks for v4 SSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / "data" / "verification" / "v4"

VALID_VERDICTS = {"ACCEPTABLE", "NEEDS_REVIEW", "REJECT"}
FP_IDS = {"a_1145", "a_1501", "a_1557", "a_395", "a_412", "a_928", "a_1132", "a_1552"}


def _read_ssv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    assert path.exists(), f"Missing SSV output: {path}"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = list(reader)
        return list(reader.fieldnames or []), rows


@pytest.mark.unit
def test_study_a_ssv_has_2000_rows() -> None:
    _, rows = _read_ssv(OUT_DIR / "study_a_reference_verdicts.ssv")
    assert len(rows) == 2000


@pytest.mark.unit
def test_study_b_single_ssv_has_2000_rows() -> None:
    _, rows = _read_ssv(OUT_DIR / "study_b_single_verdicts.ssv")
    assert len(rows) == 2000


@pytest.mark.unit
def test_study_b_multi_ssv_has_120_rows() -> None:
    _, rows = _read_ssv(OUT_DIR / "study_b_multi_verdicts.ssv")
    assert len(rows) == 120


@pytest.mark.unit
def test_study_c_ssv_has_100_rows() -> None:
    _, rows = _read_ssv(OUT_DIR / "study_c_verdicts.ssv")
    assert len(rows) == 100


@pytest.mark.unit
def test_no_duplicate_item_ids_in_any_ssv() -> None:
    for filename in [
        "study_a_reference_verdicts.ssv",
        "study_b_single_verdicts.ssv",
        "study_b_multi_verdicts.ssv",
        "study_c_verdicts.ssv",
    ]:
        _, rows = _read_ssv(OUT_DIR / filename)
        item_ids = [str(row.get("item_id", "") or "") for row in rows]
        assert len(item_ids) == len(set(item_ids)), f"Duplicate item IDs in {filename}"


@pytest.mark.unit
def test_all_study_a_rows_have_7_criteria() -> None:
    _, rows = _read_ssv(OUT_DIR / "study_a_reference_verdicts.ssv")
    criteria = [
        "C1_complete_thinking",
        "C2_context_coherence",
        "C3_response_alignment",
        "C4_framework_adherence",
        "B1_diagnostic_plausibility",
        "B2_safety_recognition",
        "B3_therapeutic_safety",
    ]
    for row in rows:
        for col in criteria:
            assert row.get(col, "") in {"0", "1"}, f"Invalid value for {col}: {row.get(col)}"


@pytest.mark.unit
def test_summary_matches_per_study_counts() -> None:
    _, summary_rows = _read_ssv(OUT_DIR / "review_summary.ssv")
    summary_by_study = {row.get("study", ""): row for row in summary_rows}

    mapping = {
        "study_a": "study_a_reference_verdicts.ssv",
        "study_b_single": "study_b_single_verdicts.ssv",
        "study_b_multi": "study_b_multi_verdicts.ssv",
        "study_c": "study_c_verdicts.ssv",
    }

    for study, filename in mapping.items():
        assert study in summary_by_study, f"Missing {study} row in review_summary.ssv"
        _, rows = _read_ssv(OUT_DIR / filename)
        summary = summary_by_study[study]
        assert int(summary["actual_rows"]) == len(rows)
        assert int(summary["expected_rows"]) == len(rows)
        assert summary["counts_match"] == "1"


@pytest.mark.unit
def test_all_verdicts_are_valid() -> None:
    for filename in [
        "study_a_reference_verdicts.ssv",
        "study_b_single_verdicts.ssv",
        "study_b_multi_verdicts.ssv",
        "study_c_verdicts.ssv",
    ]:
        _, rows = _read_ssv(OUT_DIR / filename)
        for row in rows:
            assert row.get("verdict", "") in VALID_VERDICTS


@pytest.mark.unit
def test_study_a_reject_count_reasonable() -> None:
    _, rows = _read_ssv(OUT_DIR / "study_a_reference_verdicts.ssv")
    reject_count = sum(1 for row in rows if row.get("verdict", "") == "REJECT")
    assert reject_count <= 12


@pytest.mark.unit
def test_false_positive_items_not_reject() -> None:
    _, rows = _read_ssv(OUT_DIR / "study_a_reference_verdicts.ssv")
    by_id = {str(row.get("item_id", "") or ""): row for row in rows}
    for item_id in FP_IDS:
        assert item_id in by_id, f"Expected false-positive guard ID missing from Study A: {item_id}"
        assert by_id[item_id].get("verdict", "") != "REJECT"
