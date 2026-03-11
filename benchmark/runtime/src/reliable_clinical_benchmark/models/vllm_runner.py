"""
Generic vLLM runner for locally-served HF models.

vLLM exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint, so we
re-use the shared ``lmstudio_client.chat_completion`` helper.  Each model is
distinguished only by its HF repo name (used as the ``model`` field in the
request) and the port the vLLM server is listening on.

Default port assignments (one model per GPU at a time):
    PsyLLM-8B        → 8101
    Piaget-8B         → 8102
    Psyche-R1         → 8103
    Psych_Qwen_32B    → 8104
"""

import os
import re
from typing import Dict, List, Optional, Tuple

from .base import GenerationConfig, ModelRunner
from .lmstudio_client import chat_completion


# ── Shared helpers ──────────────────────────────────────────────────────────

def _extract_answer_and_reasoning(full_response: str) -> Tuple[str, str]:
    """
    Parse a response into (answer, reasoning) without altering content.
    Handles <think>/<redacted_reasoning> blocks, then falls back to heuristics.
    """
    text = (full_response or "").strip()
    think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
    think_match = re.search(think_pattern, text, re.DOTALL)

    if think_match:
        reasoning = think_match.group(1).strip()
        end_tag_pos = text.find("</", think_match.end())
        if end_tag_pos != -1:
            closing_tag_end = text.find(">", end_tag_pos)
            if closing_tag_end != -1:
                answer = text[closing_tag_end + 1:].strip()
            else:
                answer = text
        else:
            answer = text
        return answer, reasoning

    parts = re.split(r"\n(?:Diagnosis|Answer|Conclusion):\s*", text)
    if len(parts) >= 2:
        return parts[1].strip(), parts[0].strip()

    return text, text


# ── Runner ──────────────────────────────────────────────────────────────────

# vLLM validates requests against max_model_len server-side.
# If max_tokens is omitted, vLLM validates only the prompt length against the
# server's configured max_model_len. We keep the legacy caps as optional
# env-driven explicit limits rather than forcing them on every request.
#
# Env overrides (precedence high -> low):
# 1) VLLM_MAX_COMPLETION_TOKENS               (global override for all models)
# 2) VLLM_MAX_COMPLETION_TOKENS_PSYCH_QWEN    (Psych_Qwen only)
# 3) VLLM_MAX_COMPLETION_TOKENS_DEFAULT       (all non-Psych_Qwen models)
# 4) no client-side cap by default
_DEFAULT_NON_PSYCH_QWEN_CAP = os.environ.get("VLLM_MAX_COMPLETION_TOKENS_DEFAULT")
_DEFAULT_PSYCH_QWEN_CAP = os.environ.get("VLLM_MAX_COMPLETION_TOKENS_PSYCH_QWEN")


def _resolve_model_max_completion_tokens(model_name: str) -> Optional[int]:
    global_override = os.environ.get("VLLM_MAX_COMPLETION_TOKENS")
    if global_override:
        return int(global_override)
    name = (model_name or "").lower()
    if "psych_qwen" in name or "compumacy/psych_qwen_32b" in name:
        return int(_DEFAULT_PSYCH_QWEN_CAP) if _DEFAULT_PSYCH_QWEN_CAP else None
    return int(_DEFAULT_NON_PSYCH_QWEN_CAP) if _DEFAULT_NON_PSYCH_QWEN_CAP else None


class VLLMRunner(ModelRunner):
    """
    OpenAI-compatible client that talks to a running vLLM API server.

    Instantiate with the HF repo name used when launching the server
    (``--model`` flag) and the port it listens on.
    """

    def __init__(
        self,
        model_name: str,
        port: int = 8101,
        host: str = "127.0.0.1",
        config: Optional[GenerationConfig] = None,
    ):
        model_max_tokens = _resolve_model_max_completion_tokens(model_name)
        if (
            config is not None
            and config.max_tokens is not None
            and model_max_tokens is not None
            and config.max_tokens > model_max_tokens
        ):
            config = GenerationConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=model_max_tokens,
            )
        super().__init__(
            model_name,
            config
            or GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_tokens=None,
            ),
        )
        self.api_base = f"http://{host}:{port}/v1"

    # ── core generation ─────────────────────────────────────────────────

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
        return _extract_answer_and_reasoning(full_response)

    # ── multi-turn chat ─────────────────────────────────────────────────

    def chat(self, messages: List[Dict[str, str]], mode: str = "default") -> str:
        """
        Generate a response from chat history via the vLLM server.
        """
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
            timeout=None,
        )
