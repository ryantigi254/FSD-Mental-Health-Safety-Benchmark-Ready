#!/usr/bin/env python3
"""
Smoke test: load each of the 4 local HF models and run 2 short prompts.

Usage (from runtime/):
    conda activate mh-llm-local-env
    python scripts/dev/smoke_test_local_models.py
"""

import sys
import time
from pathlib import Path

# ── bootstrap imports ──────────────────────────────────────────────────────
RUNTIME_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.models.factory import get_model_runner

# ── config ─────────────────────────────────────────────────────────────────
MODEL_IDS = [
    "piaget_local",
    "psyllm_gml_local",
    "psyche_r1_local",
    "psych_qwen_local",
]

PROMPTS = [
    "A 28-year-old female presents with persistent low mood, anhedonia, and insomnia for 6 weeks. Provide the most likely diagnosis.",
    "A 45-year-old male reports hearing voices commenting on his actions for the past 3 months, with no substance use history. What is the most likely diagnosis?",
]

MAX_TOKENS = 256  # Short for smoke test


def smoke_test_model(model_id: str) -> bool:
    """Load one model, run both prompts, return True if all OK."""
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_id}")
    print(f"{'='*70}")

    try:
        config = GenerationConfig(max_tokens=MAX_TOKENS)
        t0 = time.time()
        runner = get_model_runner(model_id, config)
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s")
    except Exception as exc:
        print(f"  LOAD FAILED: {exc}")
        return False

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n  --- Prompt {i} ---")
        print(f"  {prompt[:80]}...")
        try:
            t0 = time.time()
            output = runner.generate(prompt)
            gen_time = time.time() - t0
            preview = output[:300].replace("\n", " ")
            print(f"  Response ({gen_time:.1f}s, {len(output)} chars): {preview}")
        except Exception as exc:
            print(f"  GENERATE FAILED: {exc}")
            return False

    # Free GPU memory before next model
    try:
        import torch
        if torch.cuda.is_available():
            del runner
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  GPU cache cleared.")
    except Exception:
        pass

    return True


def main() -> int:
    print(f"Runtime root: {RUNTIME_ROOT}")
    print(f"Models to test: {len(MODEL_IDS)}")
    print(f"Prompts per model: {len(PROMPTS)}")
    print(f"Max tokens: {MAX_TOKENS}")

    results = {}
    for model_id in MODEL_IDS:
        results[model_id] = smoke_test_model(model_id)

    print(f"\n{'='*70}")
    print("  SMOKE TEST RESULTS")
    print(f"{'='*70}")
    for model_id, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {model_id}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
