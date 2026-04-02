"""
Shared LM Studio HTTP client for locally-hosted models.

This module centralizes LM Studio request handling so all LM Studio-backed
models get consistent load checks, retry logic, and response parsing.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests


logger = logging.getLogger(__name__)


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
    try:
        endpoint = f"{_native_api_base(api_base)}/models"
        resp = requests.get(endpoint, timeout=timeout)
        resp.raise_for_status()
        data = resp.json().get("models", [])
        model_known = False
        for entry in data if isinstance(data, list) else []:
            if not isinstance(entry, dict):
                continue
            model_key = str(entry.get("key", "")).strip()
            if model_key and _model_ids_match(model, model_key):
                model_known = True

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
                    return True, loaded_id or model_key

        if model_known:
            return False, ""
        return False, ""
    except Exception as exc:
        return None, str(exc)


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

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "tool_choice": "none",
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if timeout is None:
        request_timeout = (30, None)
    elif isinstance(timeout, tuple):
        request_timeout = timeout
    else:
        request_timeout = (30, timeout) if timeout > 60 else (timeout, timeout)

    preflight_loaded, preflight_detail = get_model_load_state(api_base, model)
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
            timeout_str = f"{timeout}s" if timeout else "no timeout set"
            logger.error(
                "LM Studio request timed out (%s) for model %s",
                timeout_str,
                model,
            )
            raise TimeoutError(f"Generation timed out ({timeout_str})")

        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            logger.warning(
                "LM Studio error on attempt %d/%d for %s: %s",
                attempt,
                max_retries,
                model,
                _format_http_error(exc),
            )
            if attempt >= max_retries:
                break

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
                if not allow_autoload:
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
                if not allow_autoload:
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
