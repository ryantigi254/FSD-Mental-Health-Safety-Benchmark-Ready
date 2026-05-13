"""Ollama OpenAI-compatible runner for cloud-hosted models."""

import os
import re
from typing import Dict, List, Optional, Tuple

from .base import GenerationConfig, ModelRunner
from .lmstudio_client import chat_completion


DEFAULT_OLLAMA_API_BASE = "http://127.0.0.1:11434/v1"
DEFAULT_OLLAMA_MODEL = "minimax-m2.5:cloud"


def _normalise_openai_base_url(api_base: str) -> str:
    base_url = (api_base or DEFAULT_OLLAMA_API_BASE).rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def _extract_answer_and_reasoning(full_response: str) -> Tuple[str, str]:
    text = (full_response or "").strip()
    think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
    think_match = re.search(think_pattern, text, re.DOTALL)

    if think_match:
        reasoning = think_match.group(1).strip()
        answer = text[think_match.end() :].strip()
        return answer, reasoning

    parts = re.split(r"\n(?:Diagnosis|Answer|Conclusion):\s*", text)
    if len(parts) >= 2:
        return parts[1].strip(), parts[0].strip()

    return text, text


class OllamaCloudRunner(ModelRunner):
    """OpenAI-compatible client for Ollama-hosted cloud models."""

    def __init__(
        self,
        model_name: str = DEFAULT_OLLAMA_MODEL,
        api_base: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ):
        super().__init__(
            model_name,
            config
            or GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=2048,
            ),
        )
        self.api_base = _normalise_openai_base_url(
            api_base or os.getenv("OLLAMA_API_BASE", DEFAULT_OLLAMA_API_BASE)
        )
        self.api_key = os.getenv("OLLAMA_API_KEY")

    def generate(self, prompt: str, mode: str = "default") -> str:
        formatted_prompt = self._format_prompt(prompt, mode)
        return self.chat([{"role": "user", "content": formatted_prompt}])

    def chat(self, messages: List[Dict[str, str]], mode: str = "default") -> str:
        formatted_messages: List[Dict[str, str]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "user" and message == messages[-1] and mode != "default":
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
            api_key=self.api_key,
        )

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        return _extract_answer_and_reasoning(full_response)
