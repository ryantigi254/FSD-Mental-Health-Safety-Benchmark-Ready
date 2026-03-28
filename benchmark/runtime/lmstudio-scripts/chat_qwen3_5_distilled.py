import argparse
import json
import os

import requests

DEFAULT_BASE = "http://127.0.0.1:1234"
DEFAULT_URL = f"{DEFAULT_BASE}/v1/chat/completions"
MODELS_URL = f"{DEFAULT_BASE}/v1/models"
DEFAULT_SYSTEM = (
    "You are a clinical reasoning assistant powered by "
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0. "
    "Respond with concise, safe, clinically-grounded reasoning."
)
PRIMARY_MODEL_ID = "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0"
LEGACY_MODEL_ID = "mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2"


def resolve_model_id() -> str:
    override = os.environ.get("LM_STUDIO_QWEN35_DISTILLED_MODEL")
    if override:
        return override

    models_url = os.environ.get("LM_STUDIO_MODELS_URL", MODELS_URL)
    try:
        response = requests.get(models_url, timeout=10)
        response.raise_for_status()
        loaded = {
            str(model.get("id", "")).strip()
            for model in response.json().get("data", [])
            if isinstance(model, dict) and model.get("id")
        }
        if PRIMARY_MODEL_ID in loaded:
            return PRIMARY_MODEL_ID
        if LEGACY_MODEL_ID in loaded:
            return LEGACY_MODEL_ID
    except requests.RequestException:
        pass

    return PRIMARY_MODEL_ID


def call_lmstudio(
    prompt: str, system_prompt: str, temperature: float, max_tokens: int | None = None
) -> str:
    url = os.environ.get("LM_STUDIO_URL", DEFAULT_URL)
    api_key = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    model_id = resolve_model_id()

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response = requests.post(url, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Unexpected response: {json.dumps(data, indent=2)}")

    return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query LM Studio with the Qwen 3.5 27B distilled local model."
    )
    parser.add_argument(
        "prompt",
        help="User prompt to send to the model. Wrap in quotes for multi-line text.",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="Override the default system prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature passed to LM Studio.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate. Omit to use LM Studio server default.",
    )

    args = parser.parse_args()
    model_id = resolve_model_id()

    print(f"Calling LM Studio at {os.environ.get('LM_STUDIO_URL', DEFAULT_URL)}")
    print(f"Model id: {model_id}\n")

    reply = call_lmstudio(
        prompt=args.prompt,
        system_prompt=args.system,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("===== MODEL REPLY =====")
    print(reply)


if __name__ == "__main__":
    main()
