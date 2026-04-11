"""Unit tests for GPT-OSS LM Studio runner defaults."""

from __future__ import annotations

import pytest

from reliable_clinical_benchmark.models.lmstudio_gpt_oss import GPTOSSLMStudioRunner


@pytest.mark.unit
def test_gpt_oss_runner_defaults_to_unbounded_max_tokens(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "reliable_clinical_benchmark.models.lmstudio_gpt_oss._resolve_lmstudio_model_name",
        lambda api_base, model_name: model_name or "gpt-oss-20b",
    )
    runner = GPTOSSLMStudioRunner()
    assert runner.model_name == "gpt-oss-20b"
    assert runner.config.max_tokens is None
