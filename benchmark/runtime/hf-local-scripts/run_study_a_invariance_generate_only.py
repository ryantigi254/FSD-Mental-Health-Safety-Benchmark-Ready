import argparse
from pathlib import Path

from _invariance_runner_common import (
    DEFAULT_INVARIANCE_DATA_DIR,
    default_invariance_cache_path,
    ensure_src_on_path,
    normalize_model_id_for_path,
    resolve_invariance_output_dir,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study A invariance generation-only runner.")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument("--cache-out", type=str, default=None)
    parser.add_argument("--variant-tag", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--progress-interval-seconds", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    ensure_src_on_path(runtime_root)

    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.factory import get_model_runner
    from reliable_clinical_benchmark.pipelines.study_a import run_study_a
    from reliable_clinical_benchmark.utils.worker_runtime import resolve_worker_count

    args = _parse_args()
    base_data_dir = Path(args.data_dir) if args.data_dir else (runtime_root / DEFAULT_INVARIANCE_DATA_DIR)
    output_dir = resolve_invariance_output_dir(runtime_root, args.output_dir, base_data_dir)

    study_a_path = base_data_dir / "study_a_test.json"
    if not study_a_path.exists():
        raise SystemExit(f"Study A invariance split not found: {study_a_path}")

    config = GenerationConfig(max_tokens=args.max_tokens)
    runner = get_model_runner(args.model_id, config)
    worker_count = resolve_worker_count(args.workers, runner, lmstudio_default=4, non_lm_default=1)

    normalized_model_id = normalize_model_id_for_path(args.model_id, output_dir)
    cache_out = args.cache_out or str(
        default_invariance_cache_path(
            output_dir=output_dir,
            model_id=normalized_model_id,
            study_slug="study_a_invariance",
            variant_tag=args.variant_tag,
        )
    )

    run_study_a(
        model=runner,
        data_dir=str(base_data_dir),
        max_samples=args.max_samples,
        output_dir=str(output_dir),
        model_name=normalized_model_id,
        generate_only=True,
        cache_out=cache_out,
        workers=worker_count,
        progress_interval_seconds=args.progress_interval_seconds,
    )

    print(f"Workers: {worker_count}")
    print(cache_out)


if __name__ == "__main__":
    main()
