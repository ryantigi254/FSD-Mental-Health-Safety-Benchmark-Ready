"""
Local LM Studio runner for Piaget-8B.

This runner uses LM Studio's OpenAI-compatible /v1/chat/completions API.
It assumes the model is loaded in LM Studio with API identifier `piaget-8b`
unless overridden via `LMSTUDIO_PIAGET_MODEL`.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

from .base import GenerationConfig, ModelRunner
from .lmstudio_client import chat_completion


def extract_answer_and_reasoning(full_response: str) -> Tuple[str, str]:
    """Parse a response into answer/reasoning parts without rewriting content."""
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

    parts = re.split(r"\n(?:Diagnosis|Answer|Conclusion):\s*", text)
    if len(parts) >= 2:
        return parts[1].strip(), parts[0].strip()

    return text, text


class PiagetLMStudioRunner(ModelRunner):
    """Local LM Studio inference for Piaget-8B."""

    def __init__(
        self,
        model_name: str | None = None,
        api_base: str = "http://127.0.0.1:1234/v1",
        config: GenerationConfig = None,
    ):
        resolved_model_name = model_name or os.getenv("LMSTUDIO_PIAGET_MODEL", "piaget-8b")
        super().__init__(
            resolved_model_name,
            config or GenerationConfig(temperature=0.7, top_p=0.9, max_tokens=None),
        )
        self.api_base = api_base

    def generate(self, prompt: str, mode: str = "default") -> str:
        formatted_prompt = self._format_prompt(prompt, mode)
        return chat_completion(
            api_base=self.api_base,
            model=self.model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=None,
        )

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        return extract_answer_and_reasoning(full_response)

    def chat(self, messages: List[Dict[str, str]], mode: str = "default") -> str:
        formatted_messages = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                formatted_messages.append({"role": "system", "content": content})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})
            else:
                if message == messages[-1] and mode != "default":
                    content = self._format_prompt(content, mode)
                formatted_messages.append({"role": "user", "content": content})

        return chat_completion(
            api_base=self.api_base,
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=None,
        )
