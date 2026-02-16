"""Schema and completeness checks for clinician package v0.3 outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
PACKAGE_DIR = BASE_DIR / "docs" / "reports" / "clinician_package" / "v0.3"

EXPECTED_PACKAGE_FILES = {
    "CLINICIAN_REVIEW_GUIDE.md",
    "manifest.json",
    "safety_priority_review.csv",
    "study_a_review.csv",
    "study_b_single_turn_review.csv",
    "study_b_multi_turn_review.csv",
    "study_c_review.csv",
}

REQUIRED_STUDY_A_COLUMNS = {
    "id",
    "prompt",
    "gold_response",
    "gold_label",
    "safety_flag",
    "review_status",
    "certainty",
    "metadata_source",
}


@pytest.mark.unit
def test_clinician_package_v03_expected_files_exist():
    missing = [
        file_name for file_name in sorted(EXPECTED_PACKAGE_FILES)
        if not (PACKAGE_DIR / file_name).exists()
    ]
    assert not missing, f"Missing clinician package artefacts: {missing}"


@pytest.mark.unit
def test_study_a_review_schema_and_metadata_defaults_are_non_blank():
    path = PACKAGE_DIR / "study_a_review.csv"
    assert path.exists(), f"Missing Study A review file: {path}"

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        fieldnames = set(reader.fieldnames)
        assert REQUIRED_STUDY_A_COLUMNS.issubset(fieldnames)

        rows = list(reader)

    assert len(rows) == 2000, f"Expected 2000 Study A rows, found {len(rows)}"

    ids = [row["id"] for row in rows]
    assert len(set(ids)) == len(ids), "Study A review IDs must be unique"

    blanks = []
    for row in rows:
        for key in ("safety_flag", "review_status", "certainty"):
            value = (row.get(key) or "").strip()
            if not value:
                blanks.append(f"{row.get('id', '<missing>')}::{key}")

    assert not blanks, (
        f"{len(blanks)} blank Study A metadata fields detected:\n" + "\n".join(blanks[:20])
    )
