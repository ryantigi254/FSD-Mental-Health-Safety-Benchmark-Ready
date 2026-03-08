"""Unit tests for OllamaCloudRunner wiring and factory aliases."""

from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models.ollama_cloud import OllamaCloudRunner


def test_factory_returns_ollama_runner_for_minimax_alias():
    runner = get_model_runner("ollama_minimax_m2_5_cloud")
    assert isinstance(runner, OllamaCloudRunner)
    assert runner.model_name == "minimax-m2.5:cloud"


def test_ollama_runner_normalises_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11434")
    runner = get_model_runner("minimax-m2.5-cloud")
    assert isinstance(runner, OllamaCloudRunner)
    assert runner.api_base == "http://localhost:11434/v1"
