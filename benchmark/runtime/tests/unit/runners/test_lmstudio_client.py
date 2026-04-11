"""Unit tests for LM Studio client retry and sizing behaviour."""

from __future__ import annotations

import requests
import pytest

from reliable_clinical_benchmark.models import lmstudio_client


class _Response:
    def __init__(self, payload=None, text: str = "", status_code: int = 200):
        self._payload = payload or {}
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Client Error",
                response=self,
            )

    def json(self):
        return self._payload


@pytest.mark.unit
def test_chat_completion_reduces_max_tokens_after_context_overflow(monkeypatch: pytest.MonkeyPatch):
    post_payloads = []
    responses = [
        _Response(text="Context size has been exceeded.", status_code=400),
        _Response(
            payload={"choices": [{"message": {"content": "Recovered answer"}}]},
            status_code=200,
        ),
    ]

    def fake_post(endpoint, json=None, headers=None, timeout=None):
        post_payloads.append(dict(json))
        response = responses.pop(0)
        return response

    monkeypatch.setattr(lmstudio_client.requests, "post", fake_post)
    monkeypatch.setattr(
        lmstudio_client,
        "get_loaded_model_runtime_limits",
        lambda api_base, model: {},
    )
    monkeypatch.setattr(
        lmstudio_client,
        "get_model_load_state",
        lambda api_base, model: (True, model),
    )
    monkeypatch.setattr(lmstudio_client.time, "sleep", lambda _: None)

    result = lmstudio_client.chat_completion(
        api_base="http://127.0.0.1:1234/v1",
        model="qwen3-8b",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.7,
        max_tokens=2048,
        top_p=0.9,
    )

    assert result == "Recovered answer"
    assert post_payloads[0]["max_tokens"] == 2048
    assert post_payloads[1]["max_tokens"] == 1024


@pytest.mark.unit
def test_chat_completion_downsizes_requested_max_tokens_to_fit_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
):
    post_payloads = []

    def fake_post(endpoint, json=None, headers=None, timeout=None):
        post_payloads.append(dict(json))
        return _Response(payload={"choices": [{"message": {"content": "ok"}}]}, status_code=200)

    monkeypatch.setattr(lmstudio_client.requests, "post", fake_post)
    monkeypatch.setattr(
        lmstudio_client,
        "get_loaded_model_runtime_limits",
        lambda api_base, model: {"context_length": 4096},
    )
    monkeypatch.setattr(
        lmstudio_client,
        "get_model_load_state",
        lambda api_base, model: (True, model),
    )

    lmstudio_client.chat_completion(
        api_base="http://127.0.0.1:1234/v1",
        model="qwen3-8b",
        messages=[{"role": "user", "content": "x" * 8000}],
        temperature=0.7,
        max_tokens=4096,
        top_p=0.9,
    )

    assert post_payloads
    assert post_payloads[0]["max_tokens"] < 4096
    assert post_payloads[0]["max_tokens"] >= 256
