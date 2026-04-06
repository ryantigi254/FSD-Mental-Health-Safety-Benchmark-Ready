"""Unit tests for worker runtime helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from reliable_clinical_benchmark.models.base import GenerationConfig, ModelRunner
from reliable_clinical_benchmark.models.vllm_runner import VLLMRunner
from reliable_clinical_benchmark.utils.worker_runtime import (
    append_jsonl_with_retry,
    is_lmstudio_runner,
    is_vllm_runner,
    iter_threaded_results,
    resolve_worker_count,
    supports_parallel_workers,
)


class _DummyRunner(ModelRunner):
    def __init__(self, model_name: str = "dummy", config: Optional[GenerationConfig] = None):
        super().__init__(model_name=model_name, config=config or GenerationConfig())

    def generate(self, prompt: str, mode: str = "default") -> str:
        return f"{mode}:{prompt}"

    def generate_with_reasoning(self, prompt: str):
        return self.generate(prompt), "reasoning"


class _DummyLMStudioRunner(_DummyRunner):
    def __init__(self, model_name: str = "dummy-lm", config: Optional[GenerationConfig] = None):
        super().__init__(model_name=model_name, config=config)
        self.api_base = "http://localhost:1234/v1"


@pytest.mark.unit
def test_worker_count_fallback_for_non_lm_runner() -> None:
    runner = _DummyRunner()
    assert not is_lmstudio_runner(runner)
    assert resolve_worker_count(4, runner) == 1


@pytest.mark.unit
def test_worker_count_allows_parallel_for_lm_runner() -> None:
    runner = _DummyLMStudioRunner()
    assert is_lmstudio_runner(runner)
    assert supports_parallel_workers(runner)
    assert resolve_worker_count(4, runner) == 4
    assert resolve_worker_count(None, runner) == 4


@pytest.mark.unit
def test_worker_count_caps_gpt_oss_lmstudio_parallelism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LMSTUDIO_GPT_OSS_MAX_WORKERS", raising=False)
    from reliable_clinical_benchmark.models.lmstudio_gpt_oss import GPTOSSLMStudioRunner

    with patch("reliable_clinical_benchmark.models.lmstudio_gpt_oss.requests.get") as mock_get:
        mock_resp = mock_get.return_value
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {"data": [{"id": "gpt-oss-20b"}]}
        runner = GPTOSSLMStudioRunner()

    assert resolve_worker_count(4, runner) == 1

    monkeypatch.setenv("LMSTUDIO_GPT_OSS_MAX_WORKERS", "2")
    assert resolve_worker_count(4, runner) == 2


@pytest.mark.unit
def test_worker_count_disables_parallel_for_vllm_runner() -> None:
    runner = VLLMRunner(model_name="GMLHUHE/PsyLLM-8B", port=8101, config=GenerationConfig(max_tokens=64))
    assert is_lmstudio_runner(runner)
    assert is_vllm_runner(runner)
    assert not supports_parallel_workers(runner)
    assert resolve_worker_count(4, runner) == 1
    assert resolve_worker_count(None, runner) == 1


@pytest.mark.unit
def test_append_jsonl_with_retry_writes_entry(tmp_path: Path) -> None:
    out_path = tmp_path / "cache" / "rows.jsonl"
    payload = {"id": "row_1", "status": "ok"}
    assert append_jsonl_with_retry(out_path, payload)
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows == [payload]


@pytest.mark.unit
def test_iter_threaded_results_returns_all_jobs() -> None:
    jobs = [1, 2, 3, 4]
    seen = []

    def _worker(job: int) -> int:
        return job * 10

    for job, result in iter_threaded_results(jobs, worker_count=2, worker_fn=_worker):
        seen.append((job, result))

    assert sorted(seen) == [(1, 10), (2, 20), (3, 30), (4, 40)]
