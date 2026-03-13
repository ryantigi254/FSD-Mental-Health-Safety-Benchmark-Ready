import argparse
from pathlib import Path

from _invariance_runner_common import (
    DEFAULT_INVARIANCE_DATA_DIR,
    ensure_src_on_path,
    normalize_model_id_for_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Study C invariance generation-only runner.")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--cache-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    ensure_src_on_path(runtime_root)

    from reliable_clinical_benchmark.eval.runtime_checks import validate_study_c_schema
    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.factory import get_model_runner
    from reliable_clinical_benchmark.pipelines.study_c import run_study_c

    args = _parse_args()
    base_data_dir = Path(args.data_dir) if args.data_dir else (runtime_root / DEFAULT_INVARIANCE_DATA_DIR)
    output_dir = Path(args.output_dir) if args.output_dir else (runtime_root / "results")

    ok, errors = validate_study_c_schema(str(base_data_dir))
    if not ok:
        raise SystemExit("Study C invariance split validation failed:\n- " + "\n- ".join(errors[:30]))

    config = GenerationConfig(max_tokens=args.max_tokens)
    runner = get_model_runner(args.model_id, config)
    normalized_model_id = normalize_model_id_for_path(args.model_id, output_dir)
    cache_out = args.cache_out or str(output_dir / normalized_model_id / "study_c_invariance_generations.jsonl")

    run_study_c(
        model=runner,
        data_dir=str(base_data_dir),
        max_cases=args.max_cases,
        output_dir=str(output_dir),
        model_name=normalized_model_id,
        use_nli=False,
        generate_only=True,
        cache_out=cache_out,
    )

    print(cache_out)


if __name__ == "__main__":
    main()
