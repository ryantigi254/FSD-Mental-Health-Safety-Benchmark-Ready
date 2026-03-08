"""Unit tests for v4.1 Study A deterministic resampling helpers."""

from __future__ import annotations

import pytest

import reliable_clinical_benchmark.review.v4_reference_review as review_mod


@pytest.mark.unit
def test_extract_diagnosis_pattern_preferred() -> None:
    diagnosis = review_mod.extract_diagnosis_from_text(
        [
            "The presentation suggests Major Depressive Disorder with persistent low mood.",
            "We should evaluate safety and functioning impact.",
        ],
        "I have struggled for months.",
    )
    assert diagnosis == "Major Depressive Disorder"


@pytest.mark.unit
def test_extract_diagnosis_keyword_fallback() -> None:
    diagnosis = review_mod.extract_diagnosis_from_text(
        ["No explicit diagnosis is named."],
        "I keep having panic attacks and fear leaving home.",
    )
    assert diagnosis == "Panic Disorder"


@pytest.mark.unit
def test_iter_openr1_candidates_is_deterministic_and_skips_used(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_rows = {
        "train": [
            {
                "conversation": [
                    {
                        "patient": "I feel very low and hopeless.",
                        "counselor_content": "It sounds exhausting and painful.",
                        "counselor_think": "Major depressive symptoms are present. Assess risk factors.",
                    }
                ]
            },
            {
                "conversation": [
                    {
                        "patient": "This row should be skipped because its id is used.",
                        "counselor_content": "skip",
                        "counselor_think": "skip",
                    }
                ]
            },
        ],
        "test": [
            {
                "conversation": [
                    {
                        "patient": "I am constantly anxious at work.",
                        "counselor_content": "Let's explore this anxiety pattern.",
                        "counselor_think": "General anxiety signs and excessive worry are present.",
                    }
                ]
            }
        ],
    }

    def _fake_load_dataset(dataset_id: str, split: str):  # type: ignore[no-untyped-def]
        assert dataset_id == "fake/ds"
        return fake_rows[split]

    monkeypatch.setattr(review_mod, "load_dataset", _fake_load_dataset)

    used_ids = {1}
    first_run = list(review_mod.iter_openr1_candidates(used_source_ids=used_ids, dataset_id="fake/ds"))
    second_run = list(review_mod.iter_openr1_candidates(used_source_ids=used_ids, dataset_id="fake/ds"))

    assert [(row["split"], row["openr1_id"]) for row in first_run] == [("train", 0), ("test", 0)]
    assert first_run == second_run
    assert all(str(row.get("diagnosis_label", "")).strip() for row in first_run)
