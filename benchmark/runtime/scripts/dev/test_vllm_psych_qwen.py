#!/usr/bin/env python3
"""Quick vLLM sanity check for Psych_Qwen_32B on port 8104."""
import time
import requests

url = "http://127.0.0.1:8104/v1/chat/completions"
payload = {
    "model": "Compumacy/Psych_Qwen_32B",
    "messages": [{"role": "user", "content": "Give a concise differential diagnosis for chest pain."}],
    "max_tokens": 512,
    "temperature": 0.0,
}

t0 = time.time()
r = requests.post(url, json=payload, timeout=600)
dt = time.time() - t0

# vLLM returns JSON even on some errors; handle both cleanly
try:
    data = r.json()
except Exception:
    print("Non-JSON response:", r.status_code, r.text[:500])
    raise

if r.status_code != 200:
    print("HTTP", r.status_code)
    print("Error payload:", data)
    raise SystemExit(1)

usage = data.get("usage", {}) or {}
out_tok = usage.get("completion_tokens")
in_tok = usage.get("prompt_tokens")

print("seconds:", round(dt, 3))
print("prompt_tokens:", in_tok, "completion_tokens:", out_tok)

if out_tok:
    print("tok/s:", round(out_tok / dt, 3))
else:
    # fallback: approximate from returned text length if usage missing
    text = ""
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        pass
    approx_tok = max(1, len(text) // 4)  # rough chars->tokens
    print("tok/s (approx):", round(approx_tok / dt, 3))
