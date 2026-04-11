#!/usr/bin/env python3
"""Cross-platform launcher for study generation scripts."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_MODEL_IDS = {
    "qwen3_lmstudio",
    "piaget_lmstudio",
    "medgemma_lmstudio",
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0",
    "qwq",
    "deepseek_r1_lmstudio",
    "gpt_oss",
    "psych_qwen_32b",
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
    "gpt-oss-120b-runpod",
    "gpt_oss_120b_runpod",
    "glm-4.7-flash-runpod",
    "glm47_flash_runpod",
}
INVARIANCE_STUDIES = {
    "study_a_invariance",
    "study_b_invariance",
    "study_b_multi_turn_invariance",
    "study_c_invariance",
}
CTRL_STUDIES = {
    "ctrl_study_a",
    "ctrl_study_a_bias",
    "ctrl_study_b",
    "ctrl_study_b_multi_turn",
    "ctrl_study_c",
}

LMSTUDIO_MODEL_PREFLIGHT = {
    "qwq": {
        "env_var": "LMSTUDIO_QWQ_MODEL",
        "default": "qwq-32b",
        "aliases": ("qwen/qwq-32b", "qwq-32b"),
    },
    "qwen3_lmstudio": {
        "env_var": "LMSTUDIO_QWEN3_MODEL",
        "default": "qwen3-8b",
        "aliases": ("qwen/qwen3-8b", "qwen3-8b"),
    },
    "piaget_lmstudio": {
        "env_var": "LMSTUDIO_PIAGET_MODEL",
        "default": "piaget-8b",
        "aliases": ("piaget-8b",),
    },
    "deepseek_r1_lmstudio": {
        "env_var": "LMSTUDIO_DEEPSEEK_R1_MODEL",
        "default": "deepseek-r1-distill-qwen-14b",
        "aliases": ("deepseek-r1-distill-qwen-14b",),
    },
    "medgemma_lmstudio": {
        "env_var": "LMSTUDIO_MEDGEMMA_MODEL",
        "default": "google/medgemma-27b-it",
        "aliases": ("google.medgemma-27b-text-it", "google/medgemma-27b-it"),
    },
    "psych_qwen_32b": {
        "env_var": "LMSTUDIO_PSYCH_QWEN_MODEL",
        "default": "psych_qwen_32b",
        "aliases": ("psych_qwen_32b",),
    },
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0": {
        "env_var": "LMSTUDIO_QWEN35_DISTILLED_MODEL",
        "default": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0",
        "aliases": (
            "mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2",
            "qwen3.5-distilled",
            "qwen3.5-27b-distilled",
            "qwen3_5_distilled_lmstudio",
            "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0",
        ),
    },
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
            "ctrl_study_a",
            "ctrl_study_a_bias",
            "ctrl_study_b",
            "ctrl_study_b_multi_turn",
            "ctrl_study_c",
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
    parser.add_argument(
        "--allow-lmstudio-autoload",
        action="store_true",
        help=(
            "Allow LM Studio to lazy-load models by skipping /v1/models preflight "
            "for LM Studio model IDs."
        ),
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


def _normalise_model_id(model_id: str) -> str:
    return (model_id or "").strip().lower()


def _matches_loaded_model(loaded_model_id: str, candidate: str) -> bool:
    loaded = _normalise_model_id(loaded_model_id)
    expected = _normalise_model_id(candidate)
    if not loaded or not expected:
        return False
    if loaded == expected:
        return True
    return loaded.endswith(f"/{expected}") or loaded.endswith(f"@{expected}")


def _lmstudio_native_models_endpoint(api_base: str) -> str:
    base = (api_base or "").rstrip("/")
    if base.endswith("/api/v1"):
        return f"{base}/models"
    if base.endswith("/v1"):
        return f"{base[:-3]}/api/v1/models"
    return f"{base}/api/v1/models"


def _check_lmstudio_model_loaded(model_id: str) -> tuple[bool, str]:
    model_key = _normalise_model_id(model_id)
    preflight_cfg = LMSTUDIO_MODEL_PREFLIGHT.get(model_key)
    if not preflight_cfg:
        return True, ""

    resolved_model = os.getenv(preflight_cfg["env_var"], preflight_cfg["default"]).strip()
    aliases = [resolved_model, *preflight_cfg["aliases"]]
    api_base = os.getenv("LMSTUDIO_API_BASE", "http://127.0.0.1:1234/v1").rstrip("/")
    endpoint = _lmstudio_native_models_endpoint(api_base)
    last_failure = ""
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as request_error:
            last_failure = (
                f"Failed strict LM Studio loaded-instance preflight against {endpoint}: {request_error}."
            )
        except json.JSONDecodeError as parse_error:
            last_failure = (
                f"LM Studio native models endpoint returned non-JSON payload from {endpoint}: {parse_error}."
            )
        else:
            loaded_instance_ids = []
            matched_but_not_loaded = []
            for entry in payload.get("models", []) if isinstance(payload, dict) else []:
                if not isinstance(entry, dict):
                    continue
                entry_key = str(entry.get("key", "")).strip()
                key_matches = any(_matches_loaded_model(entry_key, alias) for alias in aliases)
                loaded_instances = entry.get("loaded_instances", [])
                if not isinstance(loaded_instances, list):
                    loaded_instances = []

                for instance in loaded_instances:
                    if not isinstance(instance, dict):
                        continue
                    loaded_model_id = str(instance.get("id", "")).strip()
                    if loaded_model_id:
                        loaded_instance_ids.append(loaded_model_id)
                    if any(
                        _matches_loaded_model(loaded_model_id, alias) or _matches_loaded_model(entry_key, alias)
                        for alias in aliases
                    ):
                        return True, loaded_model_id or entry_key

                if key_matches:
                    matched_but_not_loaded.append(entry_key or "<unknown>")

            if matched_but_not_loaded:
                matched_msg = ", ".join(dict.fromkeys(matched_but_not_loaded))
                last_failure = (
                    "LM Studio loaded-instance preflight failed. "
                    f"Found installed model entries [{matched_msg}] but they currently have no loaded_instances."
                )
            else:
                alias_msg = ", ".join(dict.fromkeys(aliases))
                available_msg = ", ".join(loaded_instance_ids) if loaded_instance_ids else "<none>"
                last_failure = (
                    "LM Studio loaded-instance preflight failed. "
                    f"Requested model-id '{model_id}' expects one of [{alias_msg}] to already be loaded, "
                    f"but /api/v1/models returned loaded instances [{available_msg}]."
                )

        if attempt < 3:
            time.sleep(2)

    return (
        False,
        f"{last_failure} Open the model once in LM Studio before running generations.",
    )


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
        "ctrl_study_a": runtime_root / "hf-local-scripts" / "run_ctrl_generate_only.py",
        "ctrl_study_a_bias": runtime_root / "hf-local-scripts" / "run_ctrl_generate_only.py",
        "ctrl_study_b": runtime_root / "hf-local-scripts" / "run_ctrl_generate_only.py",
        "ctrl_study_b_multi_turn": runtime_root / "hf-local-scripts" / "run_ctrl_generate_only.py",
        "ctrl_study_c": runtime_root / "hf-local-scripts" / "run_ctrl_generate_only.py",
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
        "ctrl_study_a": BASE_MODEL_IDS - {"psyllm"},
        "ctrl_study_a_bias": BASE_MODEL_IDS | {"gpt_oss_lmstudio"},
        "ctrl_study_b": BASE_MODEL_IDS,
        "ctrl_study_b_multi_turn": BASE_MODEL_IDS,
        "ctrl_study_c": BASE_MODEL_IDS,
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

        # Bias runner only accepts --data-path (to biased_vignettes.json). Map a split
        # root directory to the standard layout, and drop stray --data-dir when
        # --data-path is already set.
        has_data_path = any(t == "--data-path" for t in passthrough)
        normalised: list[str] = []
        i = 0
        while i < len(passthrough):
            if passthrough[i] == "--data-dir":
                dir_val, i = _consume_flag_value(passthrough, i, "")
                if not has_data_path:
                    rel = Path(dir_val) / "adversarial_bias" / "biased_vignettes.json"
                    normalised.extend(["--data-path", rel.as_posix()])
                    has_data_path = True
                continue
            normalised.append(passthrough[i])
            i += 1
        passthrough = normalised

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

    if args.study == "ctrl_study_a_bias":
        default_bias_data = "data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json"
        default_output_dir = "results_ctrl_invariance"

        out = []
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
    if args.study in (INVARIANCE_STUDIES | CTRL_STUDIES):
        command.extend(["--study", args.study])
    command.extend(["--model-id", args.model_id])
    command.extend(passthrough)

    print(f"runtime root: {runtime_root}")
    print(f"Executing: {shlex.join(command)}")
    if not args.allow_lmstudio_autoload:
        is_ready, detail = _check_lmstudio_model_loaded(args.model_id)
        if not is_ready:
            print(
                f"{detail} Load the model in LM Studio first, or pass "
                "--allow-lmstudio-autoload to bypass this guard.",
                file=sys.stderr,
            )
            return 2
        if detail:
            print(f"LM Studio preflight matched loaded model: {detail}")
    if args.check_only:
        print("Check-only mode: validation passed; no generation executed.")
        return 0

    child_env = os.environ.copy()
    if args.allow_lmstudio_autoload:
        child_env["LMSTUDIO_ALLOW_AUTOLOAD"] = "1"

    completed = subprocess.run(command, cwd=str(runtime_root), env=child_env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
