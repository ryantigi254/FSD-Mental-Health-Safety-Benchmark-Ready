"""Unit tests for Study A sparse metadata defaults and loader integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reliable_clinical_benchmark.data.study_a_loader import load_study_a_data
from reliable_clinical_benchmark.data.study_a_metadata import (
    DEFAULT_STUDY_A_METADATA,
    load_study_a_metadata_map,
    resolve_study_a_metadata,
)


@pytest.mark.unit
def test_resolve_study_a_metadata_returns_defaults_for_missing_id():
    resolved = resolve_study_a_metadata("a_999", {})
    assert resolved["id"] == "a_999"
    assert resolved["safety_flag"] == DEFAULT_STUDY_A_METADATA["safety_flag"]
    assert resolved["review_status"] == DEFAULT_STUDY_A_METADATA["review_status"]
    assert resolved["certainty"] == DEFAULT_STUDY_A_METADATA["certainty"]
    assert resolved["metadata_source"] == "default"


@pytest.mark.unit
def test_resolve_study_a_metadata_preserves_explicit_triage_values():
    metadata_map = {
        "a_010": {
            "id": "a_010",
            "safety_flag": "active_suicidal_ideation",
            "review_status": "requires_clinician",
            "certainty": "low",
            "note": "triage",
        }
    }

    resolved = resolve_study_a_metadata("a_010", metadata_map)
    assert resolved["id"] == "a_010"
    assert resolved["safety_flag"] == "active_suicidal_ideation"
    assert resolved["review_status"] == "requires_clinician"
    assert resolved["certainty"] == "low"
    assert resolved["note"] == "triage"
    assert resolved["metadata_source"] == "explicit"


@pytest.mark.unit
def test_load_study_a_data_merges_explicit_and_default_metadata(tmp_path: Path):
    data_dir = tmp_path / "data"
    splits_dir = data_dir / "openr1_psy_splits"
    gold_dir = data_dir / "study_a_gold"
    splits_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    study_a_payload = {
        "samples": [
            {
                "id": "a_001",
                "prompt": "Prompt 1",
                "gold_answer": "Response 1",
                "gold_reasoning": ["reason"],
                "metadata": {},
            },
            {
                "id": "a_002",
                "prompt": "Prompt 2",
                "gold_answer": "Response 2",
                "gold_reasoning": ["reason"],
                "metadata": {},
            },
        ]
    }
    labels_payload = {
        "labels": {
            "a_001": "Adjustment Disorder",
            "a_002": "No Diagnosis",
        }
    }
    metadata_payload = {
        "a_001": {
            "id": "a_001",
            "safety_flag": "active_suicidal_ideation",
            "review_status": "requires_clinician",
            "certainty": "low",
            "note": "explicit",
        }
    }

    (splits_dir / "study_a_test.json").write_text(
        json.dumps(study_a_payload),
        encoding="utf-8",
    )
    (gold_dir / "gold_diagnosis_labels.json").write_text(
        json.dumps(labels_payload),
        encoding="utf-8",
    )
    (gold_dir / "gold_diagnosis_metadata.json").write_text(
        json.dumps(metadata_payload),
        encoding="utf-8",
    )

    rows = load_study_a_data(str(splits_dir / "study_a_test.json"), merge_metadata=True)
    assert len(rows) == 2

    row_1 = next(r for r in rows if r["id"] == "a_001")
    row_2 = next(r for r in rows if r["id"] == "a_002")

    assert row_1["gold_diagnosis_label"] == "Adjustment Disorder"
    assert row_1["gold_diagnosis_metadata"]["metadata_source"] == "explicit"
    assert row_1["gold_diagnosis_metadata"]["review_status"] == "requires_clinician"

    assert row_2["gold_diagnosis_label"] == "No Diagnosis"
    assert row_2["gold_diagnosis_metadata"]["metadata_source"] == "default"
    assert row_2["gold_diagnosis_metadata"]["certainty"] == "unknown"


@pytest.mark.unit
def test_load_study_a_metadata_map_handles_missing_file(tmp_path: Path):
    missing_path = tmp_path / "nope.json"
    metadata_map = load_study_a_metadata_map(missing_path)
    assert metadata_map == {}
