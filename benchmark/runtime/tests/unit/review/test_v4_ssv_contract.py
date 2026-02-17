"""Contract checks for v4 cross-study SSV outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
OUT_DIR = BASE_DIR / "data" / "verification" / "v4"


def _read_ssv(path: Path) -> list[dict[str, str]]:
    assert path.exists(), f"Missing SSV output: {path}"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader)


@pytest.mark.unit
def test_v4_per_study_files_exist_and_have_expected_counts():
    expected = {
        "study_a_reference_verdicts.ssv": 2000,
        "study_b_single_verdicts.ssv": 2000,
        "study_b_multi_verdicts.ssv": 120,
        "study_c_verdicts.ssv": 100,
    }

    for filename, count in expected.items():
        rows = _read_ssv(OUT_DIR / filename)
        assert rows, f"Empty SSV output: {filename}"
        assert len(rows) == count, f"Unexpected row count for {filename}"


@pytest.mark.unit
def test_v4_item_ids_are_unique_within_each_study_file():
    filenames = [
        "study_a_reference_verdicts.ssv",
        "study_b_single_verdicts.ssv",
        "study_b_multi_verdicts.ssv",
        "study_c_verdicts.ssv",
    ]

    for filename in filenames:
        rows = _read_ssv(OUT_DIR / filename)
        ids = [row["item_id"] for row in rows]
        assert len(ids) == len(set(ids)), f"Duplicate item_id rows in {filename}"


@pytest.mark.unit
def test_v4_study_a_contains_required_rubric_columns_and_values():
    rows = _read_ssv(OUT_DIR / "study_a_reference_verdicts.ssv")

    required_cols = [
        "C1_complete_thinking",
        "C2_context_coherence",
        "C3_response_alignment",
        "C4_framework_adherence",
        "B1_diagnostic_plausibility",
        "B2_safety_recognition",
        "B3_therapeutic_safety",
        "verdict",
        "gold_diagnosis_label",
        "source_openr1_ids",
    ]

    for row in rows:
        for col in required_cols:
            assert col in row, f"Missing column in Study A SSV: {col}"
        for col in required_cols[:7]:
            assert row[col] in {"0", "1"}, f"Invalid Study A rubric value for {col}: {row[col]}"
        assert row["verdict"] in {"ACCEPTABLE", "NEEDS_REVIEW", "REJECT"}


@pytest.mark.unit
def test_v4_summary_matches_per_study_totals():
    summary_rows = _read_ssv(OUT_DIR / "review_summary.ssv")
    assert summary_rows, "Summary SSV is empty"

    row_by_study = {row["study"]: row for row in summary_rows}
    for key in ("study_a", "study_b_single", "study_b_multi", "study_c"):
        assert key in row_by_study, f"Missing summary row: {key}"

    expected_from_files = {
        "study_a": len(_read_ssv(OUT_DIR / "study_a_reference_verdicts.ssv")),
        "study_b_single": len(_read_ssv(OUT_DIR / "study_b_single_verdicts.ssv")),
        "study_b_multi": len(_read_ssv(OUT_DIR / "study_b_multi_verdicts.ssv")),
        "study_c": len(_read_ssv(OUT_DIR / "study_c_verdicts.ssv")),
    }

    for study, expected_count in expected_from_files.items():
        summary_row = row_by_study[study]
        assert int(summary_row["actual_rows"]) == expected_count
        assert int(summary_row["expected_rows"]) == expected_count
        assert summary_row["counts_match"] == "1"
        assert int(summary_row["duplicate_item_ids"]) == 0


@pytest.mark.unit
def test_v4_replacement_candidates_file_exists():
    path = OUT_DIR / "replacement_candidates_study_a.ssv"
    rows = _read_ssv(path)
    # Header-only is acceptable when no replacements are required.
    assert isinstance(rows, list)
