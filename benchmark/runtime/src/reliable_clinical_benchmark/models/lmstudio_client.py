"""
Shared LM Studio HTTP client for all locally-hosted models.

This module provides a single, unified interface for communicating with
LM Studio's local server API. All models running via LM Studio (PsyLLM,
and potentially others like QwQ-32B, DeepSeek-R1-14B if loaded locally)
should use this client to ensure consistent error handling and timeout management.

For this dissertation, PsyLLM is the primary model evaluated, running
locally via LM Studio. Other models (QwQ, DeepSeek-R1, etc.) are configured
as remote API runners for spec completeness, but the actual evaluation
focuses on local PsyLLM inference.
"""

import json
import time
import requests
from typing import Dict, Any, List, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


def is_model_loaded(api_base: str, model: str, timeout: int = 10) -> bool:
    """Check whether *model* is already loaded in LM Studio."""
    try:
        resp = requests.get(
            f"{api_base}/models",
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        loaded_ids = {m.get("id", "").lower() for m in data}
        return model.lower() in loaded_ids
    except Exception:
        return False


def _wait_for_model(
    api_base: str,
    model: str,
    poll_interval: float = 3.0,
    max_wait: float = 120.0,
) -> bool:
    """Poll ``/v1/models`` until *model* appears or *max_wait* elapses."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        if is_model_loaded(api_base, model):
            return True
        time.sleep(poll_interval)
    return False


def _flatten_content(content: Any) -> str:
    """
    Normalise LM Studio / OpenAI-style mixed content:
    - str -> as-is
    - list of blocks -> join text/content fields
    - fallback -> repr for debugging
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(str(block["content"]))
                else:
                    parts.append(json.dumps(block))
            else:
                parts.append(str(block))
        return "".join(parts)
    return repr(content)


def _split_content_and_reasoning(content: Any) -> Tuple[str, str]:
    """
    Split mixed content into (text, reasoning) to preserve think traces.
    Handles LM Studio/OpenAI-style blocks where reasoning may appear as its own type.
    """
    text_parts: List[str] = []
    reasoning_parts: List[str] = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "reasoning":
                    if "text" in block:
                        reasoning_parts.append(str(block["text"]))
                    elif "content" in block:
                        reasoning_parts.append(str(block["content"]))
                    else:
                        reasoning_parts.append(json.dumps(block))
                    continue

                if "text" in block:
                    text_parts.append(str(block["text"]))
                elif "content" in block:
                    text_parts.append(str(block["content"]))
                else:
                    text_parts.append(json.dumps(block))
            else:
                text_parts.append(str(block))
    else:
        text_parts.append(repr(content))

    return "".join(text_parts), "".join(reasoning_parts)


def _do_chat_request(
    endpoint: str,
    payload: Dict[str, Any],
    request_timeout: Union[Tuple[int, Optional[int]], None],
    model: str,
    timeout_raw: Optional[Union[int, Tuple[int, Optional[int]]]],
    api_key: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    """Execute a single chat completion HTTP request (no retry logic)."""
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)

    response = requests.post(
        endpoint,
        json=payload,
        headers=headers,
        timeout=request_timeout,
    )
    response.raise_for_status()
    result = response.json()

    choice = result["choices"][0]["message"]
    content_text, content_reasoning = _split_content_and_reasoning(
        choice.get("content", "")
    )

    # Preserve reasoning if LM Studio returns it alongside content
    reasoning_field = choice.get("reasoning")
    if not reasoning_field:
        # LM Studio commonly emits reasoning as `reasoning_content`.
        reasoning_field = choice.get("reasoning_content")
    reasoning_parts: List[str] = []
    if isinstance(reasoning_field, str) and reasoning_field.strip():
        reasoning_parts.append(reasoning_field.strip())
    if content_reasoning.strip():
        reasoning_parts.append(content_reasoning.strip())

    combined_reasoning = "\n".join(reasoning_parts).strip()
    content = content_text if content_text is not None else _flatten_content(
        choice.get("content", "")
    )
    if combined_reasoning:
        content = f"<think>{combined_reasoning}</think>\n{content}"

    return content


def chat_completion(
    api_base: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    top_p: float,
    timeout: Optional[Union[int, Tuple[int, Optional[int]]]] = None,
    api_key: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    max_retries: int = 3,
) -> str:
    """
    Single shared helper for LM Studio /v1/chat/completions endpoint.

    Before making a request the function verifies the model is loaded via
    ``/v1/models``.  On transient errors (HTTP 4xx/5xx, connection resets)
    it re-checks model availability and retries up to *max_retries* times
    instead of blindly re-sending — this prevents LM Studio from spawning
    duplicate model-load operations.

    Args:
        api_base: Base URL for LM Studio API (e.g., "http://localhost:1234/v1")
        model: Model name/identifier as recognised by LM Studio
        messages: List of message dicts with "role" and "content" keys
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate. If None, omit the field and
            let LM Studio apply its own server/model default.
        top_p: Nucleus sampling parameter
        timeout: Request timeout in seconds. If None, no timeout (default: None).
                 Can be a tuple (connect_timeout, read_timeout) for fine-grained control.
                 For long-running generations, use None or a large read_timeout.
        max_retries: Maximum number of retry attempts after transient errors.

    Returns:
        Generated text content from the model

    Raises:
        TimeoutError: If request exceeds timeout
        requests.exceptions.RequestException: For other HTTP/connection errors
    """
    endpoint = f"{api_base}/chat/completions"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "tool_choice": "none",
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    # Use tuple timeout: (connect_timeout, read_timeout)
    # Connect timeout: 30s to fail fast if server is down
    # Read timeout: None (no limit) to allow long generations
    if timeout is None:
        request_timeout = (30, None)  # 30s connect, no read timeout
    elif isinstance(timeout, tuple):
        request_timeout = timeout
    else:
        # If single int provided, use it for both (backward compatibility)
        # But prefer no read timeout for long generations
        request_timeout = (30, timeout) if timeout > 60 else (timeout, timeout)

    # ── Pre-flight: verify model is loaded ──────────────────────────
    if not is_model_loaded(api_base, model):
        logger.info(
            "Model %s not yet visible in /v1/models — waiting up to 120 s",
            model,
        )
        if not _wait_for_model(api_base, model, max_wait=120.0):
            raise RuntimeError(
                f"Model {model!r} never appeared in LM Studio /v1/models. "
                "Load it manually in LM Studio before running generations."
            )

    # ── Request with retry ──────────────────────────────────────────
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return _do_chat_request(
                endpoint, payload, request_timeout, model, timeout,
                api_key=api_key, extra_headers=extra_headers,
            )

        except requests.exceptions.Timeout:
            timeout_str = f"{timeout}s" if timeout else "no timeout set"
            logger.error(
                "LM Studio request timed out (%s) for model %s", timeout_str, model,
            )
            raise TimeoutError(f"Generation timed out ({timeout_str})")

        except (requests.exceptions.HTTPError, requests.exceptions.RequestException) as exc:
            last_exc = exc
            logger.warning(
                "LM Studio error on attempt %d/%d for %s: %s",
                attempt, max_retries, model, exc,
            )
            if attempt >= max_retries:
                break

            # Check model is still loaded before retrying — avoids
            # triggering a redundant model-load in LM Studio.
            if not is_model_loaded(api_base, model):
                logger.info(
                    "Model %s disappeared from /v1/models — waiting for it to reload",
                    model,
                )
                if not _wait_for_model(api_base, model, max_wait=120.0):
                    raise RuntimeError(
                        f"Model {model!r} not available after error; "
                        "check LM Studio."
                    ) from exc
            else:
                # Model is loaded; brief back-off before retry.
                time.sleep(2 * attempt)

        except (KeyError, IndexError) as exc:
            logger.error(
                "LM Studio response parsing error for model %s: %s", model, exc,
            )
            raise ValueError(f"Unexpected response format from LM Studio: {exc}") from exc

    # All retries exhausted — raise the last captured exception.
    raise last_exc  # type: ignore[misc]
