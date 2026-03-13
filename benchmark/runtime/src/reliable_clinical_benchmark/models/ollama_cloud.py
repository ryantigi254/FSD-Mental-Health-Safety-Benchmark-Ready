"""OpenAI-compatible runner for the Ollama-hosted Minimax cloud alias."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

from .base import GenerationConfig, ModelRunner
from .lmstudio_client import chat_completion


def _normalise_api_base(raw_base: str | None) -> str:
    base = (raw_base or "http://localhost:11434").rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


class OllamaCloudRunner(ModelRunner):
    """Use Ollama's OpenAI-compatible API for the Minimax cloud alias."""

    def __init__(
        self,
        model_name: str = "minimax-m2.5:cloud",
        api_base: str | None = None,
        config: GenerationConfig | None = None,
    ):
        super().__init__(
            model_name,
            config
            or GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=None,
            ),
        )
        self.api_base = _normalise_api_base(api_base or os.getenv("OLLAMA_API_BASE"))

    def generate(self, prompt: str, mode: str = "default") -> str:
        formatted_prompt = self._format_prompt(prompt, mode)
        messages = [{"role": "user", "content": formatted_prompt}]
        return chat_completion(
            api_base=self.api_base,
            model=self.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=None,
        )

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        return full_response, full_response

    def chat(self, messages: List[Dict[str, str]], mode: str = "default") -> str:
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user" and msg == messages[-1] and mode != "default":
                content = self._format_prompt(content, mode)
            formatted_messages.append({"role": role, "content": content})
        return chat_completion(
            api_base=self.api_base,
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=None,
        )
