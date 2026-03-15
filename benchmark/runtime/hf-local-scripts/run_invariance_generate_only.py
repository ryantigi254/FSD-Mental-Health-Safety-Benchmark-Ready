#!/usr/bin/env python3
"""Canonical invariance generation runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from _invariance_runner_common import (
    DEFAULT_INVARIANCE_DATA_DIR,
    ensure_src_on_path,
    normalize_model_id_for_path,
)


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
INVARIANCE_STUDIES = (
    "study_a_invariance",
    "study_b_invariance",
    "study_b_multi_turn_invariance",
    "study_c_invariance",
)
DEFAULT_MAX_TOKENS = {
    "study_a_invariance": 32000,
    "study_b_invariance": 16384,
    "study_b_multi_turn_invariance": 16384,
    "study_c_invariance": 16384,
}
CACHE_NAME_MAP = {
    "study_a_invariance": "study_a_invariance_generations.jsonl",
    "study_b_invariance": "study_b_invariance_generations.jsonl",
    "study_b_multi_turn_invariance": "study_b_multi_turn_invariance_generations.jsonl",
    "study_c_invariance": "study_c_invariance_generations.jsonl",
}
VARIANT_BUNDLE_CHILDREN = {
    "study_a_invariance": "study_a_lexical",
    "study_b_invariance": "study_b_intensity_mild",
    "study_b_multi_turn_invariance": "study_b_multi_turn_schedule_earlier",
    "study_c_invariance": "study_c_noncritical_reorder",
}
V5_TREE_CHILDREN = {
    "study_a_invariance": Path("study_a") / "lexical",
    "study_b_invariance": Path("study_b") / "mild",
    "study_b_multi_turn_invariance": Path("study_b_multi_turn") / "schedule_earlier",
    "study_c_invariance": Path("study_c") / "noncritical_reorder",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical invariance generation-only runner.")
    parser.add_argument("--study", choices=INVARIANCE_STUDIES, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--cache-out", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--progress-interval-seconds", type=int, default=10)
    return parser.parse_args()


def _resolve_data_dir(study: str, data_dir: str | None) -> Path:
    base_data_dir = Path(data_dir) if data_dir else (RUNTIME_ROOT / DEFAULT_INVARIANCE_DATA_DIR)
    if (base_data_dir / "study_a_test.json").exists():
        return base_data_dir
    if base_data_dir.name == "v5_invariance_variants":
        candidate = base_data_dir / VARIANT_BUNDLE_CHILDREN[study]
        if candidate.exists():
            return candidate
    if base_data_dir.name == "v5" and (base_data_dir / "variant_matrix_manifest.json").exists():
        candidate = base_data_dir / V5_TREE_CHILDREN[study]
        if candidate.exists():
            return candidate
    return base_data_dir


def _resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir is None:
        return RUNTIME_ROOT / "results"
    candidate = Path(output_dir)
    if candidate.is_absolute():
        return candidate
    return RUNTIME_ROOT / candidate


def _resolve_cache_out(study: str, cache_out: str | None, output_dir: Path, model_dir: str) -> str:
    if cache_out:
        return cache_out
    return str(output_dir / model_dir / CACHE_NAME_MAP[study])


def _resolve_max_tokens(study: str, max_tokens: int | None) -> int:
    if max_tokens is not None:
        return max_tokens
    return DEFAULT_MAX_TOKENS[study]


def _validate_args(args: argparse.Namespace) -> None:
    if args.study == "study_c_invariance" and args.max_samples is not None:
        raise SystemExit("--max-samples is only valid for Study A/Study B invariance runs.")
    if args.study != "study_c_invariance" and args.max_cases is not None:
        raise SystemExit("--max-cases is only valid for Study C invariance runs.")


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    ensure_src_on_path(RUNTIME_ROOT)

    from reliable_clinical_benchmark.eval.runtime_checks import (
        validate_study_b_schema,
        validate_study_c_schema,
    )
    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.factory import get_model_runner
    from reliable_clinical_benchmark.pipelines.study_a import run_study_a
    from reliable_clinical_benchmark.pipelines.study_b import run_study_b
    from reliable_clinical_benchmark.pipelines.study_c import run_study_c

    base_data_dir = _resolve_data_dir(args.study, args.data_dir)
    output_dir = _resolve_output_dir(args.output_dir)
    model_dir = normalize_model_id_for_path(args.model_id, output_dir)
    cache_out = _resolve_cache_out(args.study, args.cache_out, output_dir, model_dir)
    max_tokens = _resolve_max_tokens(args.study, args.max_tokens)

    config = GenerationConfig(max_tokens=max_tokens)
    runner = get_model_runner(args.model_id, config)

    if args.study == "study_a_invariance":
        split_path = base_data_dir / "study_a_test.json"
        if not split_path.exists():
            raise SystemExit(f"Study A invariance split not found: {split_path}")

        run_study_a(
            model=runner,
            data_dir=str(base_data_dir),
            max_samples=args.max_samples,
            output_dir=str(output_dir),
            model_name=model_dir,
            generate_only=True,
            cache_out=cache_out,
            workers=args.workers,
            progress_interval_seconds=args.progress_interval_seconds,
        )
    elif args.study == "study_b_invariance":
        ok, errors = validate_study_b_schema(str(base_data_dir))
        if not ok:
            raise SystemExit("Study B invariance split validation failed:\n- " + "\n- ".join(errors[:30]))

        run_study_b(
            model=runner,
            data_dir=str(base_data_dir),
            max_samples=args.max_samples,
            output_dir=str(output_dir),
            model_name=model_dir,
            use_nli=False,
            generate_only=True,
            cache_out=cache_out,
            do_single_turn=True,
            do_multi_turn=False,
            workers=args.workers,
            progress_interval_seconds=args.progress_interval_seconds,
        )
    elif args.study == "study_b_multi_turn_invariance":
        split_path = base_data_dir / "study_b_multi_turn_test.json"
        if not split_path.exists():
            raise SystemExit(f"Study B multi-turn invariance split not found: {split_path}")

        run_study_b(
            model=runner,
            data_dir=str(base_data_dir),
            max_samples=args.max_samples,
            output_dir=str(output_dir),
            model_name=model_dir,
            use_nli=False,
            generate_only=True,
            cache_out=cache_out,
            do_single_turn=False,
            do_multi_turn=True,
            workers=args.workers,
            progress_interval_seconds=args.progress_interval_seconds,
        )
    else:
        ok, errors = validate_study_c_schema(str(base_data_dir))
        if not ok:
            raise SystemExit("Study C invariance split validation failed:\n- " + "\n- ".join(errors[:30]))

        run_study_c(
            model=runner,
            data_dir=str(base_data_dir),
            max_cases=args.max_cases,
            output_dir=str(output_dir),
            model_name=model_dir,
            use_nli=False,
            generate_only=True,
            cache_out=cache_out,
            workers=args.workers,
            progress_interval_seconds=args.progress_interval_seconds,
        )

    print(cache_out)


if __name__ == "__main__":
    main()
