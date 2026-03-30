"""Unit tests for MedGemma LM Studio runner helpers (no LM Studio required)."""

import pytest

from reliable_clinical_benchmark.models.lmstudio_medgemma import (
    _resolve_lmstudio_model_name,
    extract_answer_and_reasoning,
)


@pytest.mark.unit
def test_extract_answer_and_reasoning_from_think_block():
    raw = "<think>Clinical reasoning here.</think>\nFinal response."
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert "Clinical reasoning here." in reasoning
    assert "Final response." in answer


@pytest.mark.unit
def test_resolve_lmstudio_model_name_from_hf_repo_id(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"id": "google.medgemma-27b-text-it"},
                    {"id": "some-other-model"},
                ]
            }

    def fake_get(*_args, **_kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "reliable_clinical_benchmark.models.lmstudio_medgemma.requests.get",
        fake_get,
    )

    resolved = _resolve_lmstudio_model_name(
        "http://127.0.0.1:1234/v1",
        "google/medgemma-27b-it",
    )
    assert resolved == "google.medgemma-27b-text-it"
