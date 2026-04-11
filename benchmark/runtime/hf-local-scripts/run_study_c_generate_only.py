import argparse
import logging
import os
import sys
from pathlib import Path


# Set PyTorch CUDA allocator config to reduce memory fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

STUDY_C_LMSTUDIO_MAX_TOKENS = 4096
STUDY_C_NON_LM_MAX_TOKENS = 16384
STUDY_C_LMSTUDIO_MODEL_IDS = {
    "gpt_oss",
    "gpt_oss_lmstudio",
    "gpt-oss-lmstudio",
    "gpt-oss-20b",
    "qwen3_lmstudio",
    "qwen3-lmstudio",
    "qwen3-8b-lmstudio",
    "piaget_lmstudio",
    "piaget-lmstudio",
    "piaget-8b-lmstudio",
    "mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2",
    "qwen3.5-distilled",
    "qwen3.5-27b-distilled",
    "qwen3_5_distilled_lmstudio",
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0",
}


def _ensure_src_on_path(runtime_root: Path) -> None:
    src_dir = runtime_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Study C generation-only runner (no evaluation.py).")
    p.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model ID understood by reliable_clinical_benchmark.models.factory.get_model_runner",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing study_c_test.json (defaults to runtime/data/openr1_psy_splits).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Results directory (defaults to runtime/results).",
    )
    p.add_argument("--max-cases", type=int, default=None, help="Limit Study C cases.")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Optional max new tokens per generation. If omitted, Study C uses a safer "
            "default: 4096 for LM Studio models and 16384 for non-LM Studio models."
        ),
    )
    p.add_argument(
        "--cache-out",
        type=str,
        default=None,
        help="Explicit cache path (defaults to results/<model-id>/study_c_generations.jsonl).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of parallel generation workers. "
            "Default is auto: 4 for LM Studio runners, 1 for non-LM Studio runners."
        ),
    )
    p.add_argument(
        "--progress-interval-seconds",
        type=int,
        default=10,
        help="Heartbeat interval for progress logging while waiting for workers.",
    )
    return p.parse_args()


def _normalize_model_id_for_path(model_id: str, output_dir: Path) -> str:
    """Normalize model_id to match existing folder names in results directory."""
    alias_map = {
        "gpt_oss": "gpt-oss-20b",
        "gpt-oss-120b": "gpt-oss-120b",
        "gpt-oss-120b-runpod": "gpt-oss-120b",
        "gpt_oss_120b_runpod": "gpt-oss-120b",
        "gpt_oss_remote": "gpt-oss-120b",
        "gpt_oss_120b_remote": "gpt-oss-120b",
        "deepseek_r1_lmstudio": "deepseek-r1-lmstudio",
        "piaget_lmstudio": "piaget-8b-local",
        "piaget_local": "piaget-8b-local",
        "psych_qwen_local": "psych-qwen-32b-local",
        "psyllm_gml_local": "psyllm-gml-local",
        "psyche_r1_local": "psyche-r1-local",
        "qwen3_lmstudio": "qwen3-lmstudio",
        "medgemma_lmstudio": "medgemma-lmstudio",
        "ollama_minimax_m2_5_cloud": "minimax-m2.5-cloud",
        "minimax_m2_5_cloud": "minimax-m2.5-cloud",
        "mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0",
        "qwen3.5-distilled": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0",
        "qwen3.5-27b-distilled": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0",
        "qwen3_5_distilled_lmstudio": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0",
        "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0",
    }

    alias_target = alias_map.get(model_id.lower())
    if alias_target:
        alias_path = output_dir / alias_target
        if alias_path.exists() and alias_path.is_dir():
            return alias_target
        return alias_target

    candidates = [
        model_id,
        model_id.replace("_", "-"),
        model_id.replace("-", "_"),
        model_id.lower(),
        model_id.lower().replace("_", "-"),
        model_id.lower().replace("-", "_"),
    ]
    for candidate in candidates:
        candidate_path = output_dir / candidate
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate

    return model_id.replace("_", "-").lower()


def _resolve_max_tokens(model_id: str, max_tokens: int | None) -> int:
    if max_tokens is not None:
        return max_tokens
    model_key = str(model_id or "").strip().lower()
    if model_key in STUDY_C_LMSTUDIO_MODEL_IDS:
        return STUDY_C_LMSTUDIO_MAX_TOKENS
    return STUDY_C_NON_LM_MAX_TOKENS


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(runtime_root)

    from reliable_clinical_benchmark.eval.runtime_checks import validate_study_c_schema
    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.factory import get_model_runner
    from reliable_clinical_benchmark.pipelines.study_c import run_study_c
    from reliable_clinical_benchmark.utils.worker_runtime import resolve_worker_count

    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, os.getenv("BENCHMARK_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    base_data_dir = Path(args.data_dir) if args.data_dir else (runtime_root / "data" / "openr1_psy_splits")
    output_dir = Path(args.output_dir) if args.output_dir else (runtime_root / "results")

    ok, errors = validate_study_c_schema(str(base_data_dir))
    if not ok:
        raise SystemExit("Study C split validation failed:\n- " + "\n- ".join(errors[:30]))

    config = GenerationConfig(max_tokens=_resolve_max_tokens(args.model_id, args.max_tokens))
    runner = get_model_runner(args.model_id, config)
    worker_count = resolve_worker_count(args.workers, runner, lmstudio_default=4, non_lm_default=1)

    normalized_model_id = _normalize_model_id_for_path(args.model_id, output_dir)
    cache_out = args.cache_out or str(output_dir / normalized_model_id / "study_c_generations.jsonl")

    run_study_c(
        model=runner,
        data_dir=str(base_data_dir),
        max_cases=args.max_cases,
        output_dir=str(output_dir),
        model_name=normalized_model_id,
        use_nli=False,
        generate_only=True,
        cache_out=cache_out,
        workers=worker_count,
        progress_interval_seconds=args.progress_interval_seconds,
    )

    print(cache_out)


if __name__ == "__main__":
    raise SystemExit(main())
