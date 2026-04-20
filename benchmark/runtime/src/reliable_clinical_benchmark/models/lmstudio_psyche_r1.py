"""
Local LM Studio runner for Psyche-R1 GGUF variants.

This runner talks to LM Studio's OpenAI-compatible endpoint
(/v1/chat/completions). LM Studio is responsible for applying the GGUF
chat template and exposing the loaded model through the configured API
identifier.
"""

import os
import re
from typing import Dict, List, Tuple

from .base import GenerationConfig, ModelRunner
from .lmstudio_client import chat_completion


def extract_answer_and_reasoning(full_response: str) -> Tuple[str, str]:
    """
    Parse a response into (answer, reasoning) without rewriting model content.
    """
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


class PsycheR1LMStudioRunner(ModelRunner):
    """
    Local LM Studio inference for Psyche-R1 GGUF variants.
    """

    def __init__(
        self,
        model_name: str = "psyche-r1@f16",
        api_base: str = "http://127.0.0.1:1234/v1",
        config: GenerationConfig = None,
    ):
        resolved_model_name = os.getenv("LMSTUDIO_PSYCHE_R1_MODEL", model_name)
        super().__init__(
            resolved_model_name,
            config
            or GenerationConfig(
                temperature=0.2,
                top_p=0.8,
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
            timeout=None,
        )

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        full_response = self.generate(prompt, mode="cot")
        return extract_answer_and_reasoning(full_response)

    def chat(self, messages: List[Dict[str, str]], mode: str = "default") -> str:
        """
        Generate a response from chat history using LM Studio chat completions.
        """
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted_messages.append({"role": "system", "content": content})
            elif role == "user":
                if msg == messages[-1] and mode != "default":
                    formatted_messages.append(
                        {"role": "user", "content": self._format_prompt(content, mode)}
                    )
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
            timeout=None,
        )
