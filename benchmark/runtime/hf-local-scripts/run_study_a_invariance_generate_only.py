import argparse
from pathlib import Path

from _invariance_runner_common import (
    DEFAULT_INVARIANCE_DATA_DIR,
    ensure_src_on_path,
    normalize_model_id_for_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study A invariance generation-only runner.")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--cache-out", type=str, default=None)
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
    output_dir = Path(args.output_dir) if args.output_dir else (runtime_root / "results")

    study_a_path = base_data_dir / "study_a_test.json"
    if not study_a_path.exists():
        raise SystemExit(f"Study A invariance split not found: {study_a_path}")

    config = GenerationConfig(max_tokens=args.max_tokens)
    runner = get_model_runner(args.model_id, config)
    worker_count = resolve_worker_count(args.workers, runner, lmstudio_default=4, non_lm_default=1)

    normalized_model_id = normalize_model_id_for_path(args.model_id, output_dir)
    cache_out = args.cache_out or str(output_dir / normalized_model_id / "study_a_invariance_generations.jsonl")

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
