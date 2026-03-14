from __future__ import annotations

import json
from pathlib import Path

import pytest

from reliable_clinical_benchmark.invariance import (
    build_invariance_manifest,
    materialize_invariance_split_root,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
def test_materialize_invariance_split_root_builds_frozen_layout(tmp_path: Path):
    source_root = tmp_path / "v5"
    manifest_dir = tmp_path / "manifests"
    output_root = tmp_path / "materialized"

    _write_json(
        source_root / "study_a_test.json",
        {"samples": [{"id": "a_001", "prompt": "P1", "gold_answer": "Major Depressive Disorder", "gold_reasoning": []}]},
    )
    _write_json(source_root / "study_a" / "gold_diagnosis_labels.json", {"labels": {"a_001": "Major Depressive Disorder"}})
    _write_json(source_root / "study_a" / "gold_diagnosis_metadata.json", {"a_001": {"safety_flag": "none"}})
    _write_json(source_root / "study_a" / "gold_labels_mapping.json", {"mapping": {"a_001": {"gold_label": "Major Depressive Disorder"}}})

    _write_json(
        source_root / "study_b_test.json",
        [
            {
                "id": "b_0001",
                "prompt": "B1",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "metadata": {"persona_id": "rowan", "age": 23},
            }
        ],
    )
    _write_json(
        source_root / "study_b_multi_turn_test.json",
        [
            {
                "id": "b_mt_0001",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "pressure_style": "self_doubt",
                "pressure_schedule": "gradual",
                "turns": [{"turn": 1, "message": "x", "pressure_level": 1}],
                "metadata": {"persona_id": "rowan", "age": 23},
            }
        ],
    )
    _write_json(
        source_root / "study_c_test.json",
        {
            "cases": [
                {
                    "id": "c_001",
                    "patient_summary": "A 24-year-old patient with major depressive disorder.",
                    "critical_entities": ["major depressive disorder", "24"],
                    "turns": [{"turn": 1, "message": "x"}],
                    "metadata": {"persona_id": "aisha", "source_openr1_ids": [1], "source_split": "test"},
                }
            ]
        },
    )
    _write_json(source_root / "study_c" / "study_c_target_plans.json", {"meta": {}, "plans": {"c_001": {"plan": "Monitor mood."}}})
    _write_json(source_root / "study_c" / "entity_evidence_map.json", {"meta": {}, "global_synonyms": {}, "case_evidence": {"c_001": {"24": "24"}}})

    for study_name, sample_size in (
        ("study_a", 1),
        ("study_b", 1),
        ("study_b_multi_turn", 1),
        ("study_c", 1),
    ):
        manifest = build_invariance_manifest(
            study_name,
            root=source_root,
            sample_size=sample_size,
            seed=42,
            min_high_risk=0,
        )
        _write_json(manifest_dir / f"{study_name}_manifest.json", manifest)

    summary = materialize_invariance_split_root(
        source_root=source_root,
        output_root=output_root,
        manifest_dir=manifest_dir,
    )

    assert summary["layout"] == "frozen_snapshot"
    assert (output_root / "study_a_test.json").exists()
    assert (output_root / "study_b_test.json").exists()
    assert (output_root / "study_b_multi_turn_test.json").exists()
    assert (output_root / "study_b_multi_turn.json").exists()
    assert (output_root / "study_c_test.json").exists()
    assert (output_root / "study_a" / "gold_diagnosis_labels.json").exists()
    assert (output_root / "study_c" / "target_plans.json").exists()
    assert (output_root / "study_c" / "study_c_target_plans.json").exists()


@pytest.mark.unit
def test_materialize_invariance_split_root_accepts_controllability_layout(tmp_path: Path):
    source_root = tmp_path / "ctrl"
    manifest_dir = tmp_path / "ctrl_manifests"
    output_root = tmp_path / "ctrl_materialized"

    _write_json(
        source_root / "study_a_controllability_test.json",
        {
            "samples": [
                {
                    "id": "ctrl_a_001",
                    "prompt": "P1",
                    "gold_answer": "Major Depressive Disorder",
                    "gold_reasoning": [],
                    "metadata": {"source_split": "train", "source_openr1_ids": [1]},
                }
            ]
        },
    )
    _write_json(source_root / "ctrl_gold_diagnosis_labels.json", {"meta": {}, "labels": {"ctrl_a_001": "Major Depressive Disorder"}})

    _write_json(
        source_root / "study_b_controllability_test.json",
        [
            {
                "id": "ctrl_b_0001",
                "prompt": "B1",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "metadata": {"persona_id": "rowan", "age": 23, "source_openr1_ids": [2], "source_split": "train"},
            }
        ],
    )
    _write_json(
        source_root / "study_b_multi_turn_controllability_test.json",
        [
            {
                "id": "ctrl_b_mt_0001",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "pressure_style": "self_doubt",
                "pressure_schedule": "gradual",
                "turns": [{"turn": 1, "message": "x", "pressure_level": 1}],
                "metadata": {"persona_id": "rowan", "age": 23, "source_openr1_ids": [3], "source_split": "train"},
            }
        ],
    )
    _write_json(
        source_root / "study_c_controllability_test.json",
        {
            "cases": [
                {
                    "id": "ctrl_c_001",
                    "patient_summary": "A 24-year-old patient with major depressive disorder.",
                    "critical_entities": ["major depressive disorder", "24"],
                    "turns": [{"turn": 1, "message": "x"}],
                    "metadata": {"persona_id": "aisha", "source_openr1_ids": [4], "source_split": "train"},
                }
            ]
        },
    )
    _write_json(source_root / "ctrl_target_plans.json", {"meta": {}, "plans": {"ctrl_c_001": {"plan": "Monitor mood."}}})

    for study_name, sample_size in (
        ("study_a", 1),
        ("study_b", 1),
        ("study_b_multi_turn", 1),
        ("study_c", 1),
    ):
        manifest = build_invariance_manifest(
            study_name,
            root=source_root,
            sample_size=sample_size,
            seed=42,
            min_high_risk=0,
        )
        _write_json(manifest_dir / f"{study_name}_manifest.json", manifest)

    summary = materialize_invariance_split_root(
        source_root=source_root,
        output_root=output_root,
        manifest_dir=manifest_dir,
    )

    assert summary["source_profile"] == "controllability"
    assert (output_root / "study_a_test.json").exists()
    assert (output_root / "study_b_test.json").exists()
    assert (output_root / "study_b_multi_turn_test.json").exists()
    assert (output_root / "study_c_test.json").exists()
    assert (output_root / "study_a" / "gold_diagnosis_labels.json").exists()
    assert (output_root / "study_c" / "target_plans.json").exists()
