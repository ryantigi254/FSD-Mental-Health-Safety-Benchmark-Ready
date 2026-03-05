#!/usr/bin/env python3
"""Study A bias generation-only runner for MLX batched inference on Mac."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path


def _ensure_src_on_path(runtime_root: Path) -> None:
    src_dir = runtime_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _format_bias_prompt(vignette: str) -> str:
    return (
        f"{vignette}\n\n"
        "Think step-by-step about the patient's presentation and provide the most likely diagnosis. "
        "Explain your reasoning clearly."
    )


def _trim_repetition_tail(
    text: str,
    min_phrase_tokens: int = 3,
    max_phrase_tokens: int = 12,
    min_repeats: int = 4,
) -> tuple[str, bool, str]:
    """
    Trim degenerate repeated tail patterns (e.g. "x y z" looped many times).

    This is a post-generation safety net for runaway loops when the model fails
    to emit EOS. It only trims repeated n-gram tails with high confidence.
    """
    tokens = text.split()
    if len(tokens) < min_phrase_tokens * min_repeats:
        return text, False, ""

    normalised = [re.sub(r"[^a-z0-9]+", "", tok.lower()) for tok in tokens]
    normalised = [tok for tok in normalised]

    upper_n = min(max_phrase_tokens, max(min_phrase_tokens, len(tokens) // min_repeats))
    for n in range(upper_n, min_phrase_tokens - 1, -1):
        phrase = normalised[-n:]
        repeats = 0
        cursor = len(tokens)
        while cursor >= n and normalised[cursor - n : cursor] == phrase:
            repeats += 1
            cursor -= n
        if repeats >= min_repeats:
            kept_tokens = tokens[: cursor + n]  # keep one occurrence, drop repeated suffix
            cleaned = " ".join(kept_tokens).strip()
            note = f"trimmed_repeated_tail_ngram={n} repeats={repeats}"
            return cleaned, True, note

    # Fallback: detect low-diversity repetitive tail (e.g. "coherence, coherence, ...")
    tail_window = min(120, len(tokens))
    tail_norm = [tok for tok in normalised[-tail_window:] if tok]
    if len(tail_norm) >= 40:
        freq: dict[str, int] = {}
        for tok in tail_norm:
            freq[tok] = freq.get(tok, 0) + 1
        dominant_token, dominant_count = max(freq.items(), key=lambda kv: kv[1])
        unique_count = len(freq)
        if dominant_count >= 20 and dominant_count / len(tail_norm) >= 0.35 and unique_count <= 6:
            # Trim starting from first token of the low-diversity tail segment.
            trim_start = len(tokens) - tail_window
            cleaned = " ".join(tokens[:trim_start]).strip()
            if cleaned:
                note = (
                    "trimmed_low_diversity_tail "
                    f"dominant={dominant_token} count={dominant_count} unique={unique_count}"
                )
                return cleaned, True, note

    return text, False, ""


def _canonical_model_output_dir(model_id: str) -> str:
    canonical_names = {
        "psych_qwen_local": "psych-qwen-32b-local",
        "psych_qwen_mlx": "psych-qwen-32b-local",
        "psych-qwen-32b-local": "psych-qwen-32b-local",
    }
    return canonical_names.get(model_id.lower(), model_id)


def _write_cache_entry(cache_path: Path, entry: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study A bias generation-only runner (MLX batched path, Mac)."
    )
    parser.add_argument("--model-id", type=str, default="psych_qwen_local")
    parser.add_argument(
        "--model",
        type=str,
        default="models/Psych_Qwen_32B_MLX_4bit",
        help="Path to MLX-converted model directory.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/adversarial_bias/biased_vignettes.json",
        help="Path to biased_vignettes.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory root.",
    )
    parser.add_argument("--cache-out", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=2000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="MLX batch size for batched generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for MLX decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling value for MLX decoding.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling value for MLX decoding.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Accepted for command compatibility; ignored on MLX path.",
    )
    return parser.parse_args()


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(runtime_root)

    from reliable_clinical_benchmark.data.adversarial_loader import load_adversarial_bias_cases

    try:
        from mlx_lm import batch_generate, load
        from mlx_lm.sample_utils import make_sampler
    except Exception as import_error:  # pragma: no cover - env dependent
        raise SystemExit(
            "mlx-lm is required for this script. Install in the active environment.\n"
            f"Import error: {import_error}"
        )

    args = _parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = runtime_root / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"MLX model not found: {model_path}")

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = runtime_root / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Bias data not found: {data_path}")

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = runtime_root / output_root
    model_output_dir = output_root / _canonical_model_output_dir(args.model_id)
    cache_path = Path(args.cache_out) if args.cache_out else model_output_dir / "study_a_bias_generations.jsonl"
    if not cache_path.is_absolute():
        cache_path = runtime_root / cache_path

    if args.quantization:
        print(
            f"Note: --quantization={args.quantization} is ignored on MLX path; "
            "use the converted MLX model directory instead."
        )

    adversarial_cases = load_adversarial_bias_cases(str(data_path))
    if not adversarial_cases:
        raise ValueError(f"No bias cases loaded from {data_path}")
    adversarial_cases = adversarial_cases[: max(1, int(args.max_cases))]

    existing_ok_ids: set[str] = set()
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("status") == "ok" and item.get("id"):
                    existing_ok_ids.add(str(item["id"]))

    pending_cases = [c for c in adversarial_cases if str(c.get("id", "")) not in existing_ok_ids]
    total_pending = len(pending_cases)

    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Cache out: {cache_path}")
    print(f"Batch size: {args.batch_size}")
    print(
        f"Sampling: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}"
    )
    print(f"Pending cases: {total_pending}")
    if total_pending == 0:
        print("No pending cases to generate.")
        return

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    model, tokenizer = load(str(model_path))
    sampler = make_sampler(
        temp=float(args.temperature),
        top_p=float(args.top_p),
        top_k=max(0, int(args.top_k)),
    )

    # Warm up once to stabilise first-batch latency.
    warm_prompt = _format_bias_prompt(pending_cases[0].get("prompt", ""))
    _ = batch_generate(
        model,
        tokenizer,
        [tokenizer.encode(warm_prompt)],
        max_tokens=min(8, int(args.max_tokens)),
        sampler=sampler,
        verbose=False,
    )

    saved = 0
    for start in range(0, total_pending, args.batch_size):
        block = pending_cases[start : start + args.batch_size]
        prompts = [_format_bias_prompt(case.get("prompt", "")) for case in block]
        encoded = [tokenizer.encode(prompt) for prompt in prompts]
        t0 = time.perf_counter()

        try:
            response = batch_generate(
                model,
                tokenizer,
                encoded,
                max_tokens=int(args.max_tokens),
                sampler=sampler,
                verbose=False,
            )
            outputs = response.texts
        except Exception as batch_error:  # pragma: no cover - runtime dependent
            outputs = ["" for _ in block]
            batch_error_msg = str(batch_error)
        else:
            batch_error_msg = ""

        latency_ms = int((time.perf_counter() - t0) * 1000)
        for case, prompt, output_text in zip(block, prompts, outputs):
            raw_text = str(output_text)
            cleaned_text, repetition_cleaned, repetition_note = _trim_repetition_tail(raw_text)
            status = "ok" if cleaned_text.strip() else "error"
            error_message = ""
            if status == "error":
                error_message = batch_error_msg or "Empty generation output from model"
            entry = {
                "id": case.get("id", ""),
                "bias_feature": case.get("bias_feature", ""),
                "bias_label": case.get("bias_label", ""),
                "prompt": prompt,
                "output_text": cleaned_text,
                "status": status,
                "error_message": error_message,
                "timestamp": _now_iso(),
                "run_id": run_id,
                "model_name": args.model_id,
                "metadata": case.get("metadata", {}),
                "sampling": {
                    "max_tokens": int(args.max_tokens),
                    "mlx_batch_size": int(args.batch_size),
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "top_k": int(args.top_k),
                },
                "meta": {
                    "latency_ms": latency_ms,
                    "repetition_cleaned": repetition_cleaned,
                    "repetition_note": repetition_note,
                },
            }
            _write_cache_entry(cache_path, entry)
            saved += 1
            print(
                f"[saved {saved}/{total_pending}] {entry['id']} "
                f"status={entry['status']} latency_ms={latency_ms}"
            )

    print(f"\nStudy A bias generation complete. Saved to {cache_path}")


if __name__ == "__main__":
    main()
