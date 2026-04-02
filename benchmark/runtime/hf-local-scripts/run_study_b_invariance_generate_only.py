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
    parser = argparse.ArgumentParser(description="Study B invariance generation-only runner.")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument("--cache-out", type=str, default=None)
    parser.add_argument("--variant-tag", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    ensure_src_on_path(runtime_root)

    from reliable_clinical_benchmark.eval.runtime_checks import validate_study_b_schema
    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.factory import get_model_runner
    from reliable_clinical_benchmark.pipelines.study_b import run_study_b

    args = _parse_args()
    base_data_dir = Path(args.data_dir) if args.data_dir else (runtime_root / DEFAULT_INVARIANCE_DATA_DIR)
    output_dir = resolve_invariance_output_dir(runtime_root, args.output_dir, base_data_dir)

    ok, errors = validate_study_b_schema(str(base_data_dir))
    if not ok:
        raise SystemExit("Study B invariance split validation failed:\n- " + "\n- ".join(errors[:30]))

    config = GenerationConfig(max_tokens=args.max_tokens)
    runner = get_model_runner(args.model_id, config)
    normalized_model_id = normalize_model_id_for_path(args.model_id, output_dir)
    cache_out = args.cache_out or str(
        default_invariance_cache_path(
            output_dir=output_dir,
            model_id=normalized_model_id,
            study_slug="study_b_invariance",
            variant_tag=args.variant_tag,
        )
    )

    run_study_b(
        model=runner,
        data_dir=str(base_data_dir),
        max_samples=args.max_samples,
        output_dir=str(output_dir),
        model_name=normalized_model_id,
        use_nli=False,
        generate_only=True,
        cache_out=cache_out,
        do_single_turn=True,
        do_multi_turn=False,
    )

    print(cache_out)


if __name__ == "__main__":
    main()
