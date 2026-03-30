"""
Local LM Studio runner for MedGemma 27B Instruct.

This runner uses LM Studio's OpenAI-compatible /v1/chat/completions API.
It resolves the configured model identifier against /v1/models so both
Hugging Face-style names (for example "google/medgemma-27b-it") and LM Studio
API identifiers (for example "google.medgemma-27b-text-it") work.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

import requests

from .base import GenerationConfig, ModelRunner
from .lmstudio_client import chat_completion

logger = logging.getLogger(__name__)

DEFAULT_MEDGEMMA_MODEL_ID = "google/medgemma-27b-it"


def extract_answer_and_reasoning(full_response: str) -> Tuple[str, str]:
    """Parse a response into answer/reasoning parts without rewriting content."""
    text = (full_response or "").strip()
    think_pattern = r"<(?:redacted_reasoning|think)>(.*?)</(?:redacted_reasoning|think)>"
    think_match = re.search(think_pattern, text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        answer = re.sub(think_pattern, "", text, count=1, flags=re.DOTALL).strip()
        return answer, reasoning

    parts = re.split(r"\n(?:Diagnosis|Answer|Conclusion):\s*", text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip(), parts[0].strip()

    return text, text


def _tokenize_model_id(model_id: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", model_id.lower()) if token]


def _resolve_lmstudio_model_name(api_base: str, configured_model_name: str) -> str:
    """
    Resolve a MedGemma model ID from LM Studio /v1/models.

    LM Studio may expose model identifiers that differ from the Hugging Face
    repo ID. We try exact, suffix, token-subset, and MedGemma-family fallbacks.
    """
    candidate_name = (configured_model_name or DEFAULT_MEDGEMMA_MODEL_ID).strip()
    if not candidate_name:
        candidate_name = DEFAULT_MEDGEMMA_MODEL_ID

    endpoint = f"{api_base.rstrip('/')}/models"
    try:
        response = requests.get(endpoint, timeout=(5, 20))
        response.raise_for_status()
        payload = response.json()
        available_model_ids = [
            entry.get("id", "").strip()
            for entry in payload.get("data", [])
            if isinstance(entry, dict) and entry.get("id")
        ]
    except Exception as request_error:
        logger.warning(
            "Could not resolve LM Studio MedGemma model id from %s (%s). Using configured id '%s'.",
            endpoint,
            request_error,
            candidate_name,
        )
        return candidate_name

    if candidate_name in available_model_ids:
        return candidate_name

    candidate_lower = candidate_name.lower()
    suffix_matches = [
        model_id for model_id in available_model_ids if model_id.lower().endswith(candidate_lower)
    ]
    if suffix_matches:
        return suffix_matches[0]

    candidate_tokens = _tokenize_model_id(candidate_name)
    token_matches = []
    for model_id in available_model_ids:
        model_tokens = set(_tokenize_model_id(model_id))
        if candidate_tokens and all(token in model_tokens for token in candidate_tokens):
            token_matches.append(model_id)
    if token_matches:
        return token_matches[0]

    medgemma_fallbacks = [
        model_id
        for model_id in available_model_ids
        if "medgemma" in model_id.lower() and "27b" in model_id.lower()
    ]
    if medgemma_fallbacks:
        return medgemma_fallbacks[0]

    logger.warning(
        "Configured model id '%s' not found in LM Studio models %s. Using configured id as-is.",
        candidate_name,
        available_model_ids,
    )
    return candidate_name


class MedGemmaLMStudioRunner(ModelRunner):
    """Local LM Studio inference for MedGemma 27B."""

    def __init__(
        self,
        model_name: str = DEFAULT_MEDGEMMA_MODEL_ID,
        api_base: str = "http://127.0.0.1:1234/v1",
        config: GenerationConfig = None,
    ):
        resolved_model_name = _resolve_lmstudio_model_name(api_base, model_name)
        super().__init__(
            resolved_model_name,
            config or GenerationConfig(temperature=0.2, top_p=0.9, max_tokens=2048),
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
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted_messages.append({"role": "system", "content": content})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})
            else:
                if msg == messages[-1] and mode != "default":
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
