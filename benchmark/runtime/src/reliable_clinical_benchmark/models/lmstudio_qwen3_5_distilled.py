"""Local LM Studio runner for the Qwen 3.5 27B distilled model.

This runner connects to LM Studio's OpenAI-compatible endpoint
(``/v1/chat/completions``) for the Qwen 3.5 27B distilled reasoning model.
It accepts both the newer `@q8_0` LM Studio loaded-model label and the older
`-v2` label used in earlier local setups.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Sequence, Tuple

from .base import ModelRunner, GenerationConfig
from .lmstudio_client import chat_completion, is_model_loaded


PRIMARY_MODEL_NAME = "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0"
LEGACY_MODEL_NAME = "mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2"
DEFAULT_MODEL_CANDIDATES = (PRIMARY_MODEL_NAME, LEGACY_MODEL_NAME)


def _resolve_loaded_model_name(api_base: str, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if is_model_loaded(api_base, candidate):
            return candidate
    return candidates[0]


def extract_answer_and_reasoning(full_response: str) -> Tuple[str, str]:
    """Extract (answer, reasoning) from <think> blocks when present."""
    text = (full_response or "").strip()
    think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
    think_match = re.search(think_pattern, text, re.DOTALL)

    if think_match:
        reasoning = think_match.group(1).strip()
        end_tag_pos = text.find("</", think_match.end())
        if end_tag_pos != -1:
            closing_tag_end = text.find(">", end_tag_pos)
            if closing_tag_end != -1:
                answer = text[closing_tag_end + 1 :].strip()
            else:
                answer = text
        else:
            answer = text
        return answer, reasoning

    # Fallback: no explicit think blocks detected.
    return text, text


class Qwen35DistilledLMStudioRunner(ModelRunner):
    """Local LM Studio inference for the Qwen 3.5 27B distilled model."""

    def __init__(
        self,
        model_name: str | None = None,
        api_base: str = "http://127.0.0.1:1234/v1",
        config: GenerationConfig | None = None,
    ):
        env_model_name = os.environ.get("LM_STUDIO_QWEN35_DISTILLED_MODEL", "").strip()
        if model_name:
            model_candidates = (model_name,)
        elif env_model_name:
            model_candidates = (env_model_name,) + tuple(
                candidate for candidate in DEFAULT_MODEL_CANDIDATES if candidate != env_model_name
            )
        else:
            model_candidates = DEFAULT_MODEL_CANDIDATES

        super().__init__(
            model_candidates[0],
            config
            or GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=None,
            ),
        )
        self.api_base = api_base
        self.model_candidates = model_candidates

    def _active_model_name(self) -> str:
        model_name = _resolve_loaded_model_name(self.api_base, self.model_candidates)
        self.model_name = model_name
        return model_name

    def generate(self, prompt: str, mode: str = "default") -> str:
        formatted_prompt = self._format_prompt(prompt, mode)
        messages = [{"role": "user", "content": formatted_prompt}]
        return chat_completion(
            api_base=self.api_base,
            model=self._active_model_name(),
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=None,
        )

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        return extract_answer_and_reasoning(full_response)

    def chat(self, messages: List[Dict[str, str]], mode: str = "default") -> str:
        formatted_messages: List[Dict[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted_messages.append({"role": "system", "content": content})
            elif role == "user":
                if msg == messages[-1] and mode != "default":
                    formatted_content = self._format_prompt(content, mode)
                    formatted_messages.append({"role": "user", "content": formatted_content})
                else:
                    formatted_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})

        return chat_completion(
            api_base=self.api_base,
            model=self._active_model_name(),
            messages=formatted_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=None,
        )
