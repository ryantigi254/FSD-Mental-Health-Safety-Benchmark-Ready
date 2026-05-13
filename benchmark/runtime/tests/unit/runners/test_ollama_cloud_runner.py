"""Unit tests for OllamaCloudRunner wiring and factory aliases."""

from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.models.factory import get_model_runner
from reliable_clinical_benchmark.models import ollama_cloud
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


def test_ollama_runner_forwards_chat_request(monkeypatch):
    captured = {}

    def fake_chat_completion(**kwargs):
        captured.update(kwargs)
        return "assistant response"

    monkeypatch.setenv("OLLAMA_API_BASE", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-key")
    monkeypatch.setattr(ollama_cloud, "chat_completion", fake_chat_completion)

    runner = OllamaCloudRunner(
        model_name="minimax-m2.5:cloud",
        config=GenerationConfig(temperature=0.2, top_p=0.8, max_tokens=123),
    )
    messages = [
        {"role": "system", "content": "Clinical assistant"},
        {"role": "user", "content": "First turn"},
        {"role": "assistant", "content": "First response"},
        {"role": "user", "content": "Second turn"},
    ]

    response = runner.chat(messages)

    assert response == "assistant response"
    assert captured == {
        "api_base": "http://localhost:11434/v1",
        "model": "minimax-m2.5:cloud",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 123,
        "top_p": 0.8,
        "timeout": None,
        "api_key": "test-key",
    }
