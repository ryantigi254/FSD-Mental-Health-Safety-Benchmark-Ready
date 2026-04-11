"""
Shared LM Studio HTTP client for locally-hosted models.

This module centralizes LM Studio request handling so all LM Studio-backed
models get consistent load checks, retry logic, and response parsing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests


logger = logging.getLogger(__name__)
DEFAULT_LMSTUDIO_READ_TIMEOUT_SECONDS = 600
# Tokens reserved for template overhead, logits, and estimator error when fitting prompts.
LMSTUDIO_CONTEXT_SAFETY_MARGIN = 2048


def _default_read_timeout_seconds() -> int:
    raw_value = os.getenv("LMSTUDIO_READ_TIMEOUT_SECONDS", str(DEFAULT_LMSTUDIO_READ_TIMEOUT_SECONDS)).strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid LMSTUDIO_READ_TIMEOUT_SECONDS=%r; using default %d",
            raw_value,
            DEFAULT_LMSTUDIO_READ_TIMEOUT_SECONDS,
        )
        return DEFAULT_LMSTUDIO_READ_TIMEOUT_SECONDS
    return max(1, parsed)


def _normalise_model_id(model: str) -> str:
    return (model or "").strip().lower()


def _model_ids_match(requested_model: str, loaded_model: str) -> bool:
    requested = _normalise_model_id(requested_model)
    loaded = _normalise_model_id(loaded_model)
    if not requested or not loaded:
        return False
    if requested == loaded:
        return True
    return (
        loaded.endswith(f"/{requested}")
        or requested.endswith(f"/{loaded}")
        or loaded.endswith(f"@{requested}")
        or requested.endswith(f"@{loaded}")
    )


def _native_api_base(api_base: str) -> str:
    base = (api_base or "").rstrip("/")
    if base.endswith("/api/v1"):
        return base
    if base.endswith("/v1"):
        return f"{base[:-3]}/api/v1"
    return f"{base}/api/v1"


def _autoload_allowed() -> bool:
    value = os.getenv("LMSTUDIO_ALLOW_AUTOLOAD", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _estimate_message_tokens(messages: List[Dict[str, str]]) -> int:
    """
    Heuristic upper bound on prompt tokens for chat payload sizing.

    Uses a conservative chars-per-token ratio so we undershoot less often than
    a naive /4 estimate (clinical text, markup, and some tokenisers run denser).
    """
    total_chars = 0
    for message in messages:
        role = str(message.get("role", "") or "")
        content = str(message.get("content", "") or "")
        total_chars += len(role) + len(content) + 8
    return max(1, math.ceil(total_chars / 3.0))


def _prune_messages_to_prompt_budget(
    messages: List[Dict[str, str]],
    max_prompt_tokens: int,
) -> List[Dict[str, str]]:
    """
    Drop oldest user/assistant turns (after an optional system message), then trim
    long user contents so the estimated prompt size stays under max_prompt_tokens.
    Preserves the last user message where possible so the active turn remains meaningful.
    """
    out: List[Dict[str, str]] = [dict(m) for m in messages]

    def est() -> int:
        return _estimate_message_tokens(out)

    while est() > max_prompt_tokens and len(out) > 2:
        if out[0].get("role") == "system" and len(out) >= 3:
            if out[1].get("role") == "user" and out[2].get("role") == "assistant":
                del out[1:3]
                continue
        if out[0].get("role") == "system" and len(out) > 1:
            del out[1]
            continue
        del out[0]

    if est() <= max_prompt_tokens:
        return out

    last_user_idx = max((i for i, m in enumerate(out) if m.get("role") == "user"), default=-1)
    for i, msg in enumerate(out):
        if msg.get("role") != "user":
            continue
        if i == last_user_idx:
            continue
        content = str(msg.get("content") or "")
        if len(content) > 800:
            msg["content"] = "[Earlier turn truncated for context budget]\n" + content[-800:]

    while est() > max_prompt_tokens and last_user_idx >= 0:
        msg = out[last_user_idx]
        content = str(msg.get("content") or "")
        if len(content) <= 400:
            break
        msg["content"] = "[Turn truncated for context budget]\n" + content[len(content) // 3 :]

    return out


def _extract_loaded_model_details(
    api_base: str,
    model: str,
    timeout: int = 10,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return loaded-instance details for *model* from LM Studio /api/v1/models."""
    endpoint = f"{_native_api_base(api_base)}/models"
    last_error = ""
    for attempt in range(1, 4):
        try:
            resp = requests.get(endpoint, timeout=timeout)
            resp.raise_for_status()
            data = resp.json().get("models", [])
            for entry in data if isinstance(data, list) else []:
                if not isinstance(entry, dict):
                    continue
                model_key = str(entry.get("key", "")).strip()
                loaded_instances = entry.get("loaded_instances", [])
                if not isinstance(loaded_instances, list):
                    continue

                for instance in loaded_instances:
                    if not isinstance(instance, dict):
                        continue
                    loaded_id = str(instance.get("id", "")).strip()
                    if (
                        (loaded_id and _model_ids_match(model, loaded_id))
                        or (model_key and _model_ids_match(model, model_key))
                    ):
                        config = instance.get("config", {})
                        if not isinstance(config, dict):
                            config = {}
                        return {
                            "loaded_id": loaded_id or model_key,
                            "model_key": model_key,
                            "context_length": config.get("context_length"),
                            "parallel": config.get("parallel"),
                        }, ""
        except Exception as exc:
            last_error = str(exc)

        if attempt < 3:
            time.sleep(2)

    return None, last_error


def get_loaded_model_runtime_limits(
    api_base: str,
    model: str,
    timeout: int = 10,
) -> Dict[str, Any]:
    """Best-effort runtime metadata for a loaded LM Studio model."""
    details, _ = _extract_loaded_model_details(api_base, model, timeout=timeout)
    return details or {}


def get_model_load_state(
    api_base: str,
    model: str,
    timeout: int = 10,
) -> Tuple[Optional[bool], str]:
    """
    Check whether *model* is already loaded in LM Studio.

    Returns:
        (True, loaded_id) when the model has a loaded instance in ``/api/v1/models``.
        (False, "") when the check succeeded and the model is not present.
        (None, error_message) when the check itself failed.
    """
    details, last_error = _extract_loaded_model_details(api_base, model, timeout=timeout)
    if details:
        return True, str(details.get("loaded_id") or details.get("model_key") or "")

    endpoint = f"{_native_api_base(api_base)}/models"
    try:
        resp = requests.get(endpoint, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("models", [])
        for entry in data if isinstance(data, list) else []:
            if not isinstance(entry, dict):
                continue
            model_key = str(entry.get("key", "")).strip()
            if model_key and _model_ids_match(model, model_key):
                return False, ""
        return False, ""
    except Exception as exc:
        if not last_error:
            last_error = str(exc)
        return None, last_error


def is_model_loaded(api_base: str, model: str, timeout: int = 10) -> bool:
    """Backward-compatible loaded-model predicate."""
    is_loaded, _ = get_model_load_state(api_base, model, timeout=timeout)
    return is_loaded is True


def _wait_for_model(
    api_base: str,
    model: str,
    poll_interval: float = 3.0,
    max_wait: float = 120.0,
) -> bool:
    """Poll ``/api/v1/models`` until *model* appears or *max_wait* elapses."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        is_loaded, _ = get_model_load_state(api_base, model)
        if is_loaded is True:
            return True
        time.sleep(poll_interval)
    return False


def _flatten_content(content: Any) -> str:
    """
    Normalize mixed LM Studio/OpenAI content.
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
    Split mixed content into (text, reasoning).
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


def _format_http_error(exc: requests.exceptions.HTTPError) -> str:
    """Include status code and a short response body when available."""
    response = exc.response
    if response is None:
        return str(exc)

    try:
        body = (response.text or "").strip()
    except Exception:
        body = ""

    if body:
        if len(body) > 400:
            body = body[:400].rstrip() + "...<truncated>"
        return f"{exc} | response={body}"
    return str(exc)


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
    content_text, content_reasoning = _split_content_and_reasoning(choice.get("content", ""))

    reasoning_field = choice.get("reasoning")
    if not reasoning_field:
        reasoning_field = choice.get("reasoning_content")
    reasoning_parts: List[str] = []
    if isinstance(reasoning_field, str) and reasoning_field.strip():
        reasoning_parts.append(reasoning_field.strip())
    if content_reasoning.strip():
        reasoning_parts.append(content_reasoning.strip())

    combined_reasoning = "\n".join(reasoning_parts).strip()
    content = content_text if content_text is not None else _flatten_content(choice.get("content", ""))
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
    Shared helper for LM Studio ``/v1/chat/completions``.

    Before making a request this checks ``/api/v1/models`` for a loaded instance.
    By default it refuses to send a request when loaded state cannot be confirmed,
    which avoids accidentally triggering LM Studio JIT autoload after a failure.
    Set ``LMSTUDIO_ALLOW_AUTOLOAD=1`` to opt back into LM Studio lazy-loading.
    """
    endpoint = f"{api_base}/chat/completions"
    allow_autoload = _autoload_allowed()
    runtime_limits = get_loaded_model_runtime_limits(api_base, model)
    context_length_opt = runtime_limits.get("context_length")
    working_messages: List[Dict[str, str]] = [dict(m) for m in messages]
    requested_max_tokens = max_tokens

    if context_length_opt is not None:
        context_length = int(context_length_opt)
        estimated_prompt_tokens = _estimate_message_tokens(working_messages)

        if requested_max_tokens is None:
            requested_max_tokens = max(
                256,
                min(8192, context_length - estimated_prompt_tokens - LMSTUDIO_CONTEXT_SAFETY_MARGIN),
            )
            logger.info(
                "LM Studio max_tokens unset; using %d for %s (context_length=%d, est_prompt=%d).",
                requested_max_tokens,
                model,
                context_length,
                estimated_prompt_tokens,
            )
        else:
            safe_budget = max(
                256,
                context_length - estimated_prompt_tokens - LMSTUDIO_CONTEXT_SAFETY_MARGIN,
            )
            if requested_max_tokens > safe_budget:
                logger.info(
                    "Reducing LM Studio max_tokens for %s from %d to %d based on loaded context_length=%d "
                    "and estimated prompt tokens=%d.",
                    model,
                    requested_max_tokens,
                    safe_budget,
                    context_length,
                    estimated_prompt_tokens,
                )
                requested_max_tokens = safe_budget

        max_prompt_tokens = max(
            256,
            context_length
            - int(requested_max_tokens or 256)
            - LMSTUDIO_CONTEXT_SAFETY_MARGIN,
        )
        est_before = _estimate_message_tokens(working_messages)
        working_messages = _prune_messages_to_prompt_budget(working_messages, max_prompt_tokens)
        est_after = _estimate_message_tokens(working_messages)
        if est_after < est_before:
            logger.info(
                "Pruned LM Studio chat messages for %s to respect context_length=%d (est_prompt %d -> %d).",
                model,
                context_length,
                est_before,
                est_after,
            )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": working_messages,
        "temperature": temperature,
        "top_p": top_p,
        "tool_choice": "none",
    }
    if requested_max_tokens is not None:
        payload["max_tokens"] = requested_max_tokens

    if timeout is None:
        request_timeout = (30, _default_read_timeout_seconds())
    elif isinstance(timeout, tuple):
        request_timeout = timeout
    else:
        request_timeout = (30, timeout) if timeout > 60 else (timeout, timeout)

    preflight_loaded, preflight_detail = get_model_load_state(api_base, model)
    preflight_confirmed_loaded = preflight_loaded is True
    if preflight_loaded is False:
        if not allow_autoload:
            raise RuntimeError(
                f"Model {model!r} is not currently loaded in LM Studio /api/v1/models. "
                "Load it once in LM Studio before running generations."
            )
        logger.info(
            "Model %s not currently loaded; autoload is enabled, so continuing.",
            model,
        )
    elif preflight_loaded is None:
        if not allow_autoload:
            raise RuntimeError(
                f"Could not verify whether model {model!r} is already loaded in LM Studio "
                f"({preflight_detail}). Refusing to send a request because that may trigger "
                "an unintended autoload."
            )
        logger.warning(
            "Could not verify whether model %s is loaded (%s). "
            "Proceeding because LMSTUDIO_ALLOW_AUTOLOAD is enabled.",
            model,
            preflight_detail,
        )

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return _do_chat_request(
                endpoint,
                payload,
                request_timeout,
                model,
                timeout,
                api_key=api_key,
                extra_headers=extra_headers,
            )

        except requests.exceptions.Timeout:
            read_timeout = request_timeout[1] if isinstance(request_timeout, tuple) else request_timeout
            timeout_str = f"{read_timeout}s" if read_timeout is not None else "no timeout set"
            logger.error(
                "LM Studio request timed out (%s) for model %s",
                timeout_str,
                model,
            )
            raise TimeoutError(f"Generation timed out ({timeout_str})")

        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            error_text = _format_http_error(exc)
            logger.warning(
                "LM Studio error on attempt %d/%d for %s: %s",
                attempt,
                max_retries,
                model,
                error_text,
            )
            if attempt >= max_retries:
                break

            if "context size has been exceeded" in error_text.lower():
                current_max_tokens = payload.get("max_tokens")
                if isinstance(current_max_tokens, int) and current_max_tokens > 256:
                    reduced_tokens = max(256, current_max_tokens // 2)
                    if reduced_tokens < current_max_tokens:
                        logger.info(
                            "Retrying LM Studio request for %s with reduced max_tokens %d -> %d after context overflow.",
                            model,
                            current_max_tokens,
                            reduced_tokens,
                        )
                        payload["max_tokens"] = reduced_tokens
                ctx_raw = runtime_limits.get("context_length")
                if ctx_raw:
                    ctx = int(ctx_raw)
                    mt = int(payload.get("max_tokens") or 256)
                    max_prompt = max(256, ctx - mt - LMSTUDIO_CONTEXT_SAFETY_MARGIN)
                    est_before = _estimate_message_tokens(payload["messages"])
                    payload["messages"] = _prune_messages_to_prompt_budget(
                        [dict(m) for m in payload["messages"]],
                        max_prompt,
                    )
                    est_after = _estimate_message_tokens(payload["messages"])
                    if est_after < est_before:
                        logger.info(
                            "Pruned messages after LM Studio context overflow for %s (est_prompt %d -> %d, max_prompt=%d).",
                            model,
                            est_before,
                            est_after,
                            max_prompt,
                        )
                else:
                    msgs = payload["messages"]
                    if (
                        len(msgs) >= 3
                        and msgs[0].get("role") == "system"
                        and msgs[1].get("role") == "user"
                        and msgs[2].get("role") == "assistant"
                    ):
                        payload["messages"] = [dict(msgs[0])] + [dict(m) for m in msgs[3:]]
                        logger.info(
                            "Dropped oldest user/assistant pair after LM Studio context overflow (no context_length metadata)."
                        )
                time.sleep(1.0)
                continue

            model_loaded, load_detail = get_model_load_state(api_base, model)
            if model_loaded is False:
                if not allow_autoload:
                    raise RuntimeError(
                        f"Model {model!r} is no longer loaded after an LM Studio error. "
                        "Aborting retries to avoid triggering an unintended reload."
                    ) from exc
                logger.info(
                    "Model %s disappeared from LM Studio; waiting for it to reappear",
                    model,
                )
                if not _wait_for_model(api_base, model, max_wait=120.0):
                    raise RuntimeError(
                        f"Model {model!r} not available after error; check LM Studio."
                    ) from exc
            elif model_loaded is None:
                if preflight_confirmed_loaded and not allow_autoload:
                    logger.warning(
                        "Could not verify model load state after LM Studio error for %s (%s). "
                        "Using cached confirmation that the model was loaded earlier and backing off.",
                        model,
                        load_detail,
                    )
                    time.sleep(2 * attempt)
                elif not allow_autoload:
                    raise RuntimeError(
                        f"Could not verify whether model {model!r} remained loaded after an LM "
                        f"Studio error ({load_detail}). Aborting retries to avoid an unintended "
                        "autoload."
                    ) from exc
                logger.warning(
                    "Could not verify model load state after LM Studio error for %s (%s). "
                    "Backing off because autoload is enabled.",
                    model,
                    load_detail,
                )
                time.sleep(2 * attempt)
            else:
                time.sleep(2 * attempt)

        except requests.exceptions.RequestException as exc:
            last_exc = exc
            logger.warning(
                "LM Studio transport error on attempt %d/%d for %s: %s",
                attempt,
                max_retries,
                model,
                exc,
            )
            if attempt >= max_retries:
                break

            model_loaded, load_detail = get_model_load_state(api_base, model)
            if model_loaded is False:
                if not allow_autoload:
                    raise RuntimeError(
                        f"Model {model!r} is no longer loaded after a transport error. "
                        "Aborting retries to avoid triggering an unintended reload."
                    ) from exc
                logger.info(
                    "Model %s disappeared from LM Studio; waiting for it to reappear",
                    model,
                )
                if not _wait_for_model(api_base, model, max_wait=120.0):
                    raise RuntimeError(
                        f"Model {model!r} not available after error; check LM Studio."
                    ) from exc
            elif model_loaded is None:
                if preflight_confirmed_loaded and not allow_autoload:
                    logger.warning(
                        "Could not verify model load state after LM Studio transport error for %s (%s). "
                        "Using cached confirmation that the model was loaded earlier and backing off.",
                        model,
                        load_detail,
                    )
                    time.sleep(2 * attempt)
                elif not allow_autoload:
                    raise RuntimeError(
                        f"Could not verify whether model {model!r} remained loaded after a "
                        f"transport error ({load_detail}). Aborting retries to avoid an "
                        "unintended autoload."
                    ) from exc
                logger.warning(
                    "Could not verify model load state after LM Studio transport error for %s (%s). "
                    "Backing off because autoload is enabled.",
                    model,
                    load_detail,
                )
                time.sleep(2 * attempt)
            else:
                time.sleep(2 * attempt)

        except (KeyError, IndexError) as exc:
            logger.error(
                "LM Studio response parsing error for model %s: %s",
                model,
                exc,
            )
            raise ValueError(f"Unexpected response format from LM Studio: {exc}") from exc

    raise last_exc  # type: ignore[misc]
