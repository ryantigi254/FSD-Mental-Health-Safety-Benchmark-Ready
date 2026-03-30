import argparse
import json
import os
import requests

DEFAULT_BASE = "http://127.0.0.1:1234"
DEFAULT_URL = f"{DEFAULT_BASE}/v1/chat/completions"
DEFAULT_SYSTEM = (
    "You are Google MedGemma 27B running through LM Studio. "
    "Be clinically cautious, concise, and avoid presenting yourself as a licensed clinician."
)
MODEL_ID = "google/medgemma-27b-it"


def call_lmstudio(prompt: str, system_prompt: str, temperature: float, max_tokens: int) -> str:
    url = os.environ.get("LM_STUDIO_URL", DEFAULT_URL)
    api_key = os.environ.get("LM_STUDIO_API_KEY", "lm-studio")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": os.environ.get("LM_STUDIO_MODEL", MODEL_ID),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(f"Unexpected response: {json.dumps(data, indent=2)}")

    return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query LM Studio with the MedGemma 27B model.")
    parser.add_argument("prompt", help="User prompt to send to the model.")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="Override the default system prompt.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature passed to LM Studio.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum new tokens to generate.")
    args = parser.parse_args()

    print(f"Calling LM Studio at {os.environ.get('LM_STUDIO_URL', DEFAULT_URL)}")
    print(f"Model id: {os.environ.get('LM_STUDIO_MODEL', MODEL_ID)}\n")
    print("===== MODEL REPLY =====")
    print(
        call_lmstudio(
            prompt=args.prompt,
            system_prompt=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
