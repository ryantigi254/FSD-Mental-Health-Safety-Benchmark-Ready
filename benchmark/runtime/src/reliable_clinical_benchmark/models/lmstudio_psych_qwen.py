"""Local LM Studio runner for Psych_Qwen_32B MLX models.

This runner connects to LM Studio's OpenAI-compatible endpoint
(``/v1/chat/completions``) and uses the model name configured in LM Studio.
"""

from __future__ import annotations

import re
from typing import Tuple, List, Dict

from .base import ModelRunner, GenerationConfig
from .lmstudio_client import chat_completion


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


class PsychQwen32bLMStudioRunner(ModelRunner):
    """Local LM Studio inference for Psych_Qwen_32B MLX via /v1/chat/completions."""

    def __init__(
        self,
        model_name: str = "psych_qwen_32b-mlx",
        api_base: str = "http://127.0.0.1:1234/v1",
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
        self.api_base = api_base

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
            timeout=None,  # allow long generations
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
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            timeout=None,  # allow long generations
        )

