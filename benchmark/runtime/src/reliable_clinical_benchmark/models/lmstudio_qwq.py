"""
Local LM Studio runner for QwQ.

Uses the shared lmstudio_client to talk to LM Studio's /v1/chat/completions.
The runner resolves the concrete LM Studio model ID from /v1/models so it can
work with provider-prefixed IDs such as ``qwen/qwq-32b``.
"""

import os
import re
from typing import Tuple, List, Dict
import logging
import requests

from .base import ModelRunner, GenerationConfig
from .lmstudio_client import chat_completion, get_loaded_model_runtime_limits

logger = logging.getLogger(__name__)

DEFAULT_QWQ_MODEL_ID = "qwq-32b"


def _resolve_lmstudio_model_name(api_base: str, configured_model_name: str) -> str:
    """
    Resolve the actual loaded QwQ model ID from LM Studio.

    Prefer the currently loaded instance ID from the native LM Studio models API
    so requests target the exact in-memory model (for example ``qwen/qwq-32b``)
    rather than a shorter alias like ``qwq-32b`` that may trigger an unnecessary
    reload path.
    """
    candidate_name = (configured_model_name or DEFAULT_QWQ_MODEL_ID).strip()
    if not candidate_name:
        candidate_name = DEFAULT_QWQ_MODEL_ID

    loaded_runtime = get_loaded_model_runtime_limits(api_base, candidate_name, timeout=5)
    loaded_id = str(loaded_runtime.get("loaded_id") or "").strip()
    if loaded_id:
        logger.info(
            "Resolved LM Studio QwQ model id '%s' -> '%s' via loaded instance.",
            candidate_name,
            loaded_id,
        )
        return loaded_id

    endpoint = f"{api_base.rstrip('/')}/models"
    try:
        response = requests.get(endpoint, timeout=(5, 20))
        response.raise_for_status()
        payload = response.json()
        model_entries = payload.get("data", []) if isinstance(payload, dict) else []
        available_model_ids = [
            entry.get("id", "").strip()
            for entry in model_entries
            if isinstance(entry, dict) and entry.get("id")
        ]
    except Exception as request_error:
        logger.warning(
            "Could not resolve LM Studio QwQ model id from %s (%s). Using configured id '%s'.",
            endpoint,
            request_error,
            candidate_name,
        )
        return candidate_name

    if not available_model_ids:
        logger.warning(
            "LM Studio returned no models from %s. Using configured id '%s'.",
            endpoint,
            candidate_name,
        )
        return candidate_name

    candidate_name_lower = candidate_name.lower()
    exact_prefixed_matches = [
        model_id
        for model_id in available_model_ids
        if model_id.lower() == candidate_name_lower and "/" in model_id
    ]
    if exact_prefixed_matches:
        resolved_model_id = exact_prefixed_matches[0]
        logger.info(
            "Resolved LM Studio model id '%s' -> '%s' via exact provider-prefixed match.",
            candidate_name,
            resolved_model_id,
        )
        return resolved_model_id

    suffix_matches = [
        model_id
        for model_id in available_model_ids
        if model_id.lower().endswith(candidate_name_lower)
    ]
    if suffix_matches:
        resolved_model_id = suffix_matches[0]
        logger.info(
            "Resolved LM Studio model id '%s' -> '%s' via suffix match.",
            candidate_name,
            resolved_model_id,
        )
        return resolved_model_id

    if candidate_name in available_model_ids:
        return candidate_name

    qwq_matches = [
        model_id
        for model_id in available_model_ids
        if "qwq" in model_id.lower()
    ]
    if qwq_matches:
        resolved_model_id = qwq_matches[0]
        logger.info(
            "Resolved LM Studio model id '%s' -> '%s' via QwQ fallback.",
            candidate_name,
            resolved_model_id,
        )
        return resolved_model_id

    logger.warning(
        "Configured model id '%s' not found in LM Studio models %s. Using configured id as-is.",
        candidate_name,
        available_model_ids,
    )
    return candidate_name


def extract_answer_and_reasoning(full_response: str) -> Tuple[str, str]:
    """
    Parse a response into (answer, reasoning) without rewriting model content.
    We only *extract* when <think> blocks exist; otherwise we fall back to heuristics.
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


class QwQLMStudioRunner(ModelRunner):
    """
    Local LM Studio inference for QwQ.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QWQ_MODEL_ID,
        api_base: str = "http://127.0.0.1:1234/v1",
        config: GenerationConfig = None,
    ):
        # Allow overriding the LM Studio API Identifier without editing code.
        # Example:
        #   $Env:LMSTUDIO_QWQ_MODEL="qwq-32b@q4_k_m"
        model_name = os.getenv("LMSTUDIO_QWQ_MODEL", model_name)
        resolved_model_name = _resolve_lmstudio_model_name(api_base, model_name)
        super().__init__(
            resolved_model_name,
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
            timeout=None,  # No timeout - allow long generations
        )

    def chat(self, messages: List[Dict[str, str]], mode: str = "default") -> str:
        """
        Generate response from chat history using LM Studio chat completion API.
        
        Properly handles multi-turn conversations with rolling context.
        """
        formatted_messages = []
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

    def generate_with_reasoning(self, prompt: str) -> Tuple[str, str]:
        """
        Extract reasoning from <think> blocks when available.
        """
        full_response = self.generate(prompt, mode="cot")
        return extract_answer_and_reasoning(full_response)

