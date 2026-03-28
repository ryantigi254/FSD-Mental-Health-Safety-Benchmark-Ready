"""Unit tests for the Qwen 3.5 distilled LM Studio runner."""

import pytest

from reliable_clinical_benchmark.models.lmstudio_qwen3_5_distilled import (
    LEGACY_MODEL_NAME,
    PRIMARY_MODEL_NAME,
    Qwen35DistilledLMStudioRunner,
    extract_answer_and_reasoning,
)


@pytest.mark.unit
def test_extract_answer_and_reasoning_from_think_block():
    raw = "<think>Reasoning here.</think>\nFinal diagnosis."
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert reasoning == "Reasoning here."
    assert "Final diagnosis" in answer


@pytest.mark.unit
def test_active_model_name_prefers_loaded_primary(monkeypatch):
    runner = Qwen35DistilledLMStudioRunner(api_base="http://127.0.0.1:1234/v1")

    monkeypatch.setattr(
        "reliable_clinical_benchmark.models.lmstudio_qwen3_5_distilled.is_model_loaded",
        lambda api_base, model: model == PRIMARY_MODEL_NAME,
    )

    assert runner._active_model_name() == PRIMARY_MODEL_NAME


@pytest.mark.unit
def test_active_model_name_falls_back_to_loaded_legacy(monkeypatch):
    runner = Qwen35DistilledLMStudioRunner(api_base="http://127.0.0.1:1234/v1")

    monkeypatch.setattr(
        "reliable_clinical_benchmark.models.lmstudio_qwen3_5_distilled.is_model_loaded",
        lambda api_base, model: model == LEGACY_MODEL_NAME,
    )

    assert runner._active_model_name() == LEGACY_MODEL_NAME


@pytest.mark.unit
def test_active_model_name_uses_primary_when_nothing_is_loaded(monkeypatch):
    runner = Qwen35DistilledLMStudioRunner(api_base="http://127.0.0.1:1234/v1")

    monkeypatch.setattr(
        "reliable_clinical_benchmark.models.lmstudio_qwen3_5_distilled.is_model_loaded",
        lambda api_base, model: False,
    )

    assert runner._active_model_name() == PRIMARY_MODEL_NAME
