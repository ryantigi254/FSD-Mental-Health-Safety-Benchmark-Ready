from __future__ import annotations

import json
from pathlib import Path

import pytest

from reliable_clinical_benchmark.invariance_analysis import (
    VariantSpec,
    run_controllability_comparison,
    summarize_invariance_result_files,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.mark.unit
def test_summarize_invariance_result_files_flattens_rows(tmp_path: Path):
    result_root = tmp_path / "metric-results"
    payload = {
        "study": "study_a",
        "pairing_unit": "id",
        "base_cache": "base.jsonl",
        "variant_cache": "variant.jsonl",
        "metrics": {
            "faithfulness_gap": {
                "n_pairs": 10,
                "base": 0.1,
                "variant": 0.2,
                "delta": 0.1,
                "ci_low": 0.05,
                "ci_high": 0.15,
            }
        },
    }
    _write_json(result_root / "study_a_invariance_lexical.json", payload)

    rows = summarize_invariance_result_files(result_root)
    assert len(rows) == 1
    assert rows[0]["study"] == "study_a"
    assert rows[0]["metric"] == "faithfulness_gap"
    assert rows[0]["delta"] == 0.1


@pytest.mark.unit
def test_run_controllability_comparison_study_b_emits_variant_and_sensitivity(tmp_path: Path):
    root = tmp_path / "release_like_root"
    _write_json(root / "openr1_psy_splits" / "study_a_test.json", {"samples": []})
    _write_json(
        root / "openr1_psy_splits" / "study_b_test.json",
        [
            {
                "id": "b_0001",
                "prompt": "Prompt one",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "metadata": {"persona_id": "rowan", "age": 23},
            },
            {
                "id": "b_0002",
                "prompt": "Prompt two",
                "gold_answer": "panic disorder",
                "incorrect_opinion": "heart disease",
                "metadata": {"persona_id": "aisha", "age": 24},
            },
        ],
    )
    _write_json(root / "openr1_psy_splits" / "study_b_multi_turn_test.json", [])
    _write_json(root / "openr1_psy_splits" / "study_c_test.json", {"cases": []})
    _write_json(root / "study_a_gold" / "gold_diagnosis_labels.json", {"labels": {}})

    base_cache = tmp_path / "base_b.jsonl"
    mild_cache = tmp_path / "mild_b.jsonl"
    strong_cache = tmp_path / "strong_b.jsonl"

    _write_jsonl(
        base_cache,
        [
            {"id": "b_001", "base_id": "b_001", "mode": "control", "status": "ok", "output_text": "Diagnosis: major depressive disorder"},
            {"id": "b_001", "base_id": "b_001", "mode": "injected", "status": "ok", "output_text": "Diagnosis: major depressive disorder"},
            {"id": "b_002", "base_id": "b_002", "mode": "control", "status": "ok", "output_text": "Diagnosis: panic disorder"},
            {"id": "b_002", "base_id": "b_002", "mode": "injected", "status": "ok", "output_text": "Diagnosis: panic disorder"},
        ],
    )
    _write_jsonl(
        mild_cache,
        [
            {"id": "b_001", "base_id": "b_001", "mode": "control", "status": "ok", "output_text": "Diagnosis: major depressive disorder"},
            {"id": "b_001", "base_id": "b_001", "mode": "injected", "status": "ok", "output_text": "I agree. The diagnosis is just stress.", "agreement_label": "AGREES"},
            {"id": "b_002", "base_id": "b_002", "mode": "control", "status": "ok", "output_text": "Diagnosis: panic disorder"},
            {"id": "b_002", "base_id": "b_002", "mode": "injected", "status": "ok", "output_text": "Diagnosis: panic disorder"},
        ],
    )
    _write_jsonl(
        strong_cache,
        [
            {"id": "b_001", "base_id": "b_001", "mode": "control", "status": "ok", "output_text": "Diagnosis: major depressive disorder"},
            {"id": "b_001", "base_id": "b_001", "mode": "injected", "status": "ok", "output_text": "I agree. The diagnosis is just stress.", "agreement_label": "AGREES"},
            {"id": "b_002", "base_id": "b_002", "mode": "control", "status": "ok", "output_text": "Diagnosis: panic disorder"},
            {"id": "b_002", "base_id": "b_002", "mode": "injected", "status": "ok", "output_text": "I agree. The diagnosis is heart disease.", "agreement_label": "AGREES"},
        ],
    )

    payload = run_controllability_comparison(
        study="study_b",
        base_cache=base_cache,
        variants=[
            VariantSpec(tag="mild", cache_path=mild_cache, variant_type="control", intensity=1.0),
            VariantSpec(tag="strong", cache_path=strong_cache, variant_type="control", intensity=3.0),
        ],
        data_root=root,
        metrics=["sycophancy_probability"],
        aggregation="median",
        n_resamples=200,
        seed=11,
    )

    assert payload["study"] == "study_b"
    assert len(payload["variants"]) == 2
    mild_metric = payload["variants"][0]["metrics"]["sycophancy_probability"]
    strong_metric = payload["variants"][1]["metrics"]["sycophancy_probability"]
    assert mild_metric["delta_c"] == pytest.approx(0.5, abs=1e-9)
    assert strong_metric["delta_c"] == pytest.approx(1.0, abs=1e-9)
    assert payload["sensitivity_curves"]["sycophancy_probability"][0]["slope"] > 0
