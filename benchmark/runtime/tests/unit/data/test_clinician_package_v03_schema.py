"""Schema and completeness checks for clinician package v0.3 outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from reliable_clinical_benchmark.data.release_paths import CLINICIAN_READINESS_RELEASE_ID


BASE_DIR = Path(__file__).resolve().parents[3]
PACKAGE_DIR = BASE_DIR / "docs" / "reports" / "clinician_package" / "v0.3"
DATA_DIR = BASE_DIR / "data"
STUDY_A_GOLD_DIR = DATA_DIR / "releases" / CLINICIAN_READINESS_RELEASE_ID / "study_a_gold"

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

REQUIRED_STUDY_B_SINGLE_COLUMNS = {
    "id",
    "prompt",
    "gold_answer",
    "incorrect_opinion",
    "persona_id",
}

REQUIRED_STUDY_B_MULTI_COLUMNS = {
    "id",
    "gold_answer",
    "incorrect_opinion",
    "pressure_style",
    "pressure_schedule",
    "turns_text",
    "persona_id",
}

REQUIRED_STUDY_C_COLUMNS = {
    "id",
    "persona_id",
    "patient_summary",
    "critical_entities",
    "target_plan",
}


@pytest.mark.unit
def test_clinician_package_v03_expected_files_exist():
    missing = [
        file_name for file_name in sorted(EXPECTED_PACKAGE_FILES)
        if not (PACKAGE_DIR / file_name).exists()
    ]
    assert not missing, f"Missing clinician package artefacts: {missing}"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    assert path.exists(), f"Missing CSV file: {path}"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        rows = list(reader)
    return list(reader.fieldnames), rows


def _study_a_alias_map() -> dict[str, str]:
    canonical_path = STUDY_A_GOLD_DIR / "label_canonical_map.json"
    payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    aliases = payload.get("aliases", {}) if isinstance(payload, dict) else {}
    return aliases if isinstance(aliases, dict) else {}


def _canonicalise(label: str, alias_map: dict[str, str]) -> str:
    return alias_map.get(label, label)


def _repo_derived_ed_ids() -> set[str]:
    labels_path = STUDY_A_GOLD_DIR / "gold_diagnosis_labels.json"
    labels_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = labels_payload.get("labels", {}) if isinstance(labels_payload, dict) else {}
    alias_map = _study_a_alias_map()
    return {
        str(sample_id)
        for sample_id, raw_label in labels.items()
        if _canonicalise(str(raw_label), alias_map) == "Eating Disorder"
    }


@pytest.mark.unit
def test_study_a_review_schema_and_metadata_defaults_are_non_blank():
    path = PACKAGE_DIR / "study_a_review.csv"
    fieldnames, rows = _read_csv(path)
    assert REQUIRED_STUDY_A_COLUMNS.issubset(set(fieldnames))

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


@pytest.mark.unit
def test_study_a_review_exports_canonical_label_for_a1538():
    path = PACKAGE_DIR / "study_a_review.csv"
    _, rows = _read_csv(path)
    by_id = {row["id"]: row for row in rows}
    assert "a_1538" in by_id, "Expected a_1538 in Study A review export"
    assert (
        by_id["a_1538"]["gold_label"] == "Attention Deficit Hyperactivity Disorder"
    ), by_id["a_1538"]["gold_label"]


@pytest.mark.unit
def test_study_b_single_turn_required_columns_present():
    path = PACKAGE_DIR / "study_b_single_turn_review.csv"
    fieldnames, rows = _read_csv(path)
    assert REQUIRED_STUDY_B_SINGLE_COLUMNS.issubset(set(fieldnames))
    assert len(rows) == 2000, f"Expected 2000 Study B single-turn rows, found {len(rows)}"


@pytest.mark.unit
def test_study_b_multi_turn_required_columns_present():
    path = PACKAGE_DIR / "study_b_multi_turn_review.csv"
    fieldnames, rows = _read_csv(path)
    assert REQUIRED_STUDY_B_MULTI_COLUMNS.issubset(set(fieldnames))
    assert len(rows) == 120, f"Expected 120 Study B multi-turn rows, found {len(rows)}"

    missing_turns_text = [row["id"] for row in rows if not (row.get("turns_text") or "").strip()]
    assert not missing_turns_text, (
        f"{len(missing_turns_text)} rows have blank turns_text: {missing_turns_text[:10]}"
    )


@pytest.mark.unit
def test_study_c_required_columns_present():
    path = PACKAGE_DIR / "study_c_review.csv"
    fieldnames, rows = _read_csv(path)
    assert REQUIRED_STUDY_C_COLUMNS.issubset(set(fieldnames))
    assert len(rows) == 100, f"Expected 100 Study C rows, found {len(rows)}"

    missing_persona = [row["id"] for row in rows if not (row.get("persona_id") or "").strip()]
    assert not missing_persona, (
        f"{len(missing_persona)} rows have blank persona_id: {missing_persona[:10]}"
    )


@pytest.mark.unit
def test_safety_sheet_includes_metadata_and_ed_union():
    path = PACKAGE_DIR / "safety_priority_review.csv"
    _, rows = _read_csv(path)
    metadata_path = STUDY_A_GOLD_DIR / "gold_diagnosis_metadata.json"
    metadata_map = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_ids = set(metadata_map.keys())
    ed_ids = _repo_derived_ed_ids()
    expected_union = metadata_ids | ed_ids

    # v0.3 fixed expectation: 27 metadata + 5 ED-derived rows.
    assert len(expected_union) == 32, f"Expected v0.3 metadata∪ED size 32, found {len(expected_union)}"
    assert len(rows) == len(expected_union), (
        f"Expected {len(expected_union)} safety-priority rows, found {len(rows)}"
    )

    row_by_id = {row["id"]: row for row in rows}
    missing_union_ids = sorted(expected_union - set(row_by_id))
    assert not missing_union_ids, f"Missing safety-sheet IDs: {missing_union_ids[:10]}"

    for sample_id in sorted(ed_ids):
        row = row_by_id[sample_id]
        assert row.get("review_status") == "requires_clinician"
        assert row.get("certainty") == "low"
        if sample_id in metadata_ids:
            assert row.get("metadata_source") == "explicit"
        else:
            assert row.get("metadata_source") == "derived_rule"
            assert row.get("safety_flag") == "eating_disorder_needs_clinician_review"
