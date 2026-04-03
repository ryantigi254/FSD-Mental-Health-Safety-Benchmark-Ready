#!/usr/bin/env python3
"""Cross-platform launcher for study generation scripts."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


BASE_MODEL_IDS = {
    "qwen3_lmstudio",
    "medgemma_lmstudio",
    "qwq",
    "deepseek_r1_lmstudio",
    "gpt_oss",
    "ollama_minimax_m2_5_cloud",
    "psyllm_gml_local",
    "piaget_local",
    "psyche_r1_local",
    "psych_qwen_local",
    "psyllm",
    "psyllm_gml_vllm",
    "piaget_vllm",
    "psyche_r1_vllm",
    "psych_qwen_vllm",
}
INVARIANCE_STUDIES = {
    "study_a_invariance",
    "study_b_invariance",
    "study_b_multi_turn_invariance",
    "study_c_invariance",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run study generation scripts with automatic runtime root resolution."
    )
    parser.add_argument(
        "--study",
        required=True,
        choices=[
            "study_a",
            "study_a_bias",
            "study_a_bias_invariance",
            "study_b",
            "study_b_multi_turn",
            "study_c",
            "study_a_invariance",
            "study_b_invariance",
            "study_b_multi_turn_invariance",
            "study_c_invariance",
        ],
        help="Study generation target.",
    )
    parser.add_argument("--model-id", required=True, help="Model ID for the target study script.")
    parser.add_argument(
        "--env",
        default=None,
        help="Optional conda env name. If omitted, uses current Python interpreter.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate path/script/model wiring only. Does not execute generation.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def _consume_flag_value(tokens: list[str], index: int, default: str) -> tuple[str, int]:
    next_index = index + 1
    if next_index < len(tokens):
        candidate = tokens[next_index]
        if candidate and not candidate.startswith("-"):
            return candidate, next_index + 1
    return default, next_index


def main() -> int:
    args, passthrough = parse_args()
    runtime_root = Path(__file__).resolve().parents[2]

    study_script_map = {
        "study_a": runtime_root / "hf-local-scripts" / "run_study_a_generate_only.py",
        "study_a_bias": runtime_root / "hf-local-scripts" / "run_study_a_bias_generate_only.py",
        "study_a_bias_invariance": runtime_root / "hf-local-scripts" / "run_study_a_bias_generate_only.py",
        "study_b": runtime_root / "hf-local-scripts" / "run_study_b_generate_only.py",
        "study_b_multi_turn": runtime_root / "hf-local-scripts" / "run_study_b_multi_turn_generate_only.py",
        "study_c": runtime_root / "hf-local-scripts" / "run_study_c_generate_only.py",
        "study_a_invariance": runtime_root / "hf-local-scripts" / "run_invariance_generate_only.py",
        "study_b_invariance": runtime_root / "hf-local-scripts" / "run_invariance_generate_only.py",
        "study_b_multi_turn_invariance": runtime_root / "hf-local-scripts" / "run_invariance_generate_only.py",
        "study_c_invariance": runtime_root / "hf-local-scripts" / "run_invariance_generate_only.py",
    }
    allowed_model_ids_by_study = {
        "study_a": BASE_MODEL_IDS - {"psyllm"},
        "study_a_bias": BASE_MODEL_IDS | {"gpt_oss_lmstudio"},
        "study_a_bias_invariance": BASE_MODEL_IDS | {"gpt_oss_lmstudio"},
        "study_b": BASE_MODEL_IDS,
        "study_b_multi_turn": BASE_MODEL_IDS,
        "study_c": BASE_MODEL_IDS,
        "study_a_invariance": (BASE_MODEL_IDS - {"psyllm"}) | {"gpt_oss_lmstudio"},
        "study_b_invariance": BASE_MODEL_IDS | {"gpt_oss_lmstudio"},
        "study_b_multi_turn_invariance": BASE_MODEL_IDS | {"gpt_oss_lmstudio"},
        "study_c_invariance": BASE_MODEL_IDS | {"gpt_oss_lmstudio"},
    }

    if args.study in {"study_a_bias", "study_a_bias_invariance"}:
        if args.study == "study_a_bias_invariance":
            default_bias_data = "data/frozen_splits/v5/adversarial_bias/biased_vignettes.json"
            default_output_dir = "results_invariance"
            default_study_name = "study_a_bias_invariance"
        else:
            default_bias_data = "data/frozen_splits/v4_1_resampled/adversarial_bias/biased_vignettes.json"
            default_output_dir = "results_invariance"
            default_study_name = "study_a_bias"

        out: list[str] = []
        i = 0
        while i < len(passthrough):
            if passthrough[i] == "--data-path":
                out.append("--data-path")
                value, i = _consume_flag_value(passthrough, i, default_bias_data)
                out.append(value)
                continue
            if passthrough[i] == "--output-dir":
                out.append("--output-dir")
                value, i = _consume_flag_value(passthrough, i, default_output_dir)
                out.append(value)
                continue
            out.append(passthrough[i])
            i += 1
        if "--data-path" not in out:
            out = ["--data-path", default_bias_data, *out]
        if "--output-dir" not in out:
            out = [*out, "--output-dir", default_output_dir]
        if "--study-name" not in out:
            out = ["--study-name", default_study_name, *out]
        passthrough = out

    target_script = study_script_map[args.study]
    if not target_script.exists():
        print(f"Script not found: {target_script}", file=sys.stderr)
        return 2
    if args.model_id not in allowed_model_ids_by_study[args.study]:
        allowed_values = ", ".join(sorted(allowed_model_ids_by_study[args.study]))
        print(
            f"Model ID '{args.model_id}' is not allowed for {args.study}. Allowed: {allowed_values}",
            file=sys.stderr,
        )
        return 2

    command = ["conda", "run", "--no-capture-output", "-n", args.env, "python"] if args.env else [sys.executable]
    command.append(str(target_script))
    if args.study in INVARIANCE_STUDIES:
        command.extend(["--study", args.study])
    command.extend(["--model-id", args.model_id])
    command.extend(passthrough)

    print(f"runtime root: {runtime_root}")
    print(f"Executing: {shlex.join(command)}")
    if args.check_only:
        print("Check-only mode: validation passed; no generation executed.")
        return 0

    completed = subprocess.run(command, cwd=str(runtime_root))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
