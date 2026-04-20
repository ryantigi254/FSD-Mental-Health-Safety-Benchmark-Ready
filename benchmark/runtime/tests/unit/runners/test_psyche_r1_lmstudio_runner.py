"""Unit tests for Psyche-R1 LM Studio runner parsing helpers."""

import pytest

from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models.lmstudio_psyche_r1 import extract_answer_and_reasoning
from reliable_clinical_benchmark.models.lmstudio_psyche_r1 import PsycheR1LMStudioRunner


@pytest.mark.unit
def test_extract_answer_and_reasoning_from_think_block():
    raw = "<think>Reasoning here.</think>\nFinal answer."
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert reasoning == "Reasoning here."
    assert "Final answer" in answer


@pytest.mark.unit
def test_extract_answer_and_reasoning_from_sections():
    raw = "Some reasoning...\nDiagnosis: Adjustment disorder"
    answer, reasoning = extract_answer_and_reasoning(raw)
    assert "Adjustment disorder" in answer
    assert "Some reasoning" in reasoning


@pytest.mark.unit
def test_factory_returns_psyche_r1_lmstudio_runner_for_alias() -> None:
    runner = get_model_runner("psyche_r1_lmstudio")
    assert isinstance(runner, PsycheR1LMStudioRunner)
