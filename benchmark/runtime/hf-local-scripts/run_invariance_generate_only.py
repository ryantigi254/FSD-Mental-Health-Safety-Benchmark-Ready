#!/usr/bin/env python3
"""Canonical invariance generation runner."""

from __future__ import annotations

import argparse
from pathlib import Path

from _invariance_runner_common import (
    DEFAULT_INVARIANCE_DATA_DIR,
    default_invariance_cache_path,
    ensure_src_on_path,
    normalize_model_id_for_path,
    resolve_invariance_output_dir,
    resolve_output_dir,
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
SPLIT_FILE_NAMES = {
    "study_a_invariance": "study_a_test.json",
    "study_b_invariance": "study_b_test.json",
    "study_b_multi_turn_invariance": "study_b_multi_turn_test.json",
    "study_c_invariance": "study_c_test.json",
}
STUDY_DIR_NAMES = {
    "study_a_invariance": "study_a",
    "study_b_invariance": "study_b",
    "study_b_multi_turn_invariance": "study_b_multi_turn",
    "study_c_invariance": "study_c",
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
LEGACY_VARIANT_PREFIXES = {
    "study_a_invariance": "study_a_",
    "study_b_invariance": "study_b_",
    "study_b_multi_turn_invariance": "study_b_multi_turn_",
    "study_c_invariance": "study_c_",
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


def _iter_variant_children(root: Path, split_file_name: str) -> list[tuple[str, Path]]:
    if not root.exists() or not root.is_dir():
        return []

    targets: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.is_dir() and (child / split_file_name).exists():
            targets.append((child.name, child))
    return targets


def _resolve_data_targets(study: str, data_dir: str | None) -> list[tuple[str | None, Path]]:
    requested_path = Path(data_dir) if data_dir else (RUNTIME_ROOT / DEFAULT_INVARIANCE_DATA_DIR)
    split_file_name = SPLIT_FILE_NAMES[study]
    study_dir_name = STUDY_DIR_NAMES[study]

    if (requested_path / split_file_name).exists():
        return [(None, requested_path)]

    if requested_path.name == "v5_invariance_variants":
        prefix = LEGACY_VARIANT_PREFIXES[study]
        legacy_targets: list[tuple[str, Path]] = []
        for child in sorted(requested_path.iterdir(), key=lambda item: item.name):
            if not child.is_dir() or not child.name.startswith(prefix):
                continue
            if (child / split_file_name).exists():
                variant_tag = child.name.removeprefix(prefix) or child.name
                legacy_targets.append((variant_tag, child))
        if legacy_targets:
            return legacy_targets

        candidate = requested_path / VARIANT_BUNDLE_CHILDREN[study]
        if candidate.exists():
            return [(candidate.name.removeprefix(prefix) or candidate.name, candidate)]

    if requested_path.name == "v5" and (requested_path / "variant_matrix_manifest.json").exists():
        study_root = requested_path / study_dir_name
        v5_targets = _iter_variant_children(study_root, split_file_name)
        if v5_targets:
            return v5_targets

        candidate = requested_path / V5_TREE_CHILDREN[study]
        if candidate.exists():
            return [(candidate.name, candidate)]

    if requested_path.name == study_dir_name:
        study_targets = _iter_variant_children(requested_path, split_file_name)
        if study_targets:
            return study_targets

    return [(None, requested_path)]


def _resolve_output_dir(output_dir: str | None) -> Path:
    return resolve_output_dir(RUNTIME_ROOT, output_dir, default_dir_name="results_invariance")


def _resolve_effective_output_dir(output_dir: str | None, data_dir: Path) -> Path:
    return resolve_invariance_output_dir(RUNTIME_ROOT, output_dir, data_dir)


def _resolve_cache_out(
    study: str,
    cache_out: str | None,
    output_dir: Path,
    model_dir: str,
    *,
    variant_tag: str | None = None,
) -> str:
    if cache_out:
        return cache_out
    return str(
        default_invariance_cache_path(
            output_dir=output_dir,
            model_id=model_dir,
            study_slug=study,
            variant_tag=variant_tag,
        )
    )


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

    data_targets = _resolve_data_targets(args.study, args.data_dir)
    if args.cache_out and len(data_targets) > 1:
        raise SystemExit(
            "--cache-out can only be used with a single concrete invariance split. "
            "Point --data-dir at one child variant folder or omit --cache-out."
        )

    max_tokens = _resolve_max_tokens(args.study, args.max_tokens)

    config = GenerationConfig(max_tokens=max_tokens)
    runner = get_model_runner(args.model_id, config)

    resolved_cache_paths: list[str] = []
    for variant_tag, base_data_dir in data_targets:
        output_dir = _resolve_effective_output_dir(args.output_dir, base_data_dir)
        model_dir = normalize_model_id_for_path(args.model_id, output_dir)
        cache_out = _resolve_cache_out(
            args.study,
            args.cache_out,
            output_dir,
            model_dir,
            variant_tag=variant_tag,
        )
        resolved_cache_paths.append(cache_out)

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

    for cache_path in resolved_cache_paths:
        print(cache_path)


if __name__ == "__main__":
    main()
