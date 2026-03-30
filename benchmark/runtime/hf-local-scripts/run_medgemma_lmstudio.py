import argparse
import sys
from pathlib import Path


def _ensure_src_on_path(runtime_root: Path) -> None:
    src_dir = runtime_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    _ensure_src_on_path(runtime_root)

    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.lmstudio_medgemma import MedGemmaLMStudioRunner
    from reliable_clinical_benchmark.pipelines.study_a import run_study_a
    from reliable_clinical_benchmark.pipelines.study_b import run_study_b
    from reliable_clinical_benchmark.utils.worker_runtime import resolve_worker_count

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--api-base",
        default="http://127.0.0.1:1234/v1",
        help="LM Studio API base URL.",
    )
    common.add_argument(
        "--api-identifier",
        default="google/medgemma-27b-it",
        help="Model identifier or Hugging Face repo id to resolve from LM Studio.",
    )
    common.add_argument(
        "--model-name",
        default="medgemma-lmstudio",
        help="Folder name under results/ (and model_name field in JSONL).",
    )
    common.add_argument("--temperature", type=float, default=0.2)
    common.add_argument("--top-p", type=float, default=0.9)
    common.add_argument("--max-new-tokens", type=int, default=4096)
    common.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Defaults to auto-resolved LM Studio settings.",
    )
    common.add_argument(
        "--progress-interval-seconds",
        type=int,
        default=10,
        help="Heartbeat interval for worker progress logs.",
    )

    parser = argparse.ArgumentParser(parents=[common])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prompt = sub.add_parser("prompt", parents=[common])
    p_prompt.add_argument("prompt")
    p_prompt.add_argument("--mode", choices=["cot", "direct"], default="cot")

    p_study_a = sub.add_parser("study-a", parents=[common])
    p_study_a.add_argument("--max-samples", type=int, default=None)
    p_study_a.add_argument(
        "--data-dir",
        default=str(runtime_root / "data" / "releases" / "clinician_readiness_v4_2026-02-22" / "openr1_psy_splits"),
    )
    p_study_a.add_argument(
        "--output-dir",
        default=str(runtime_root / "results"),
    )
    p_study_a.add_argument("--generate-only", action="store_true")
    p_study_a.add_argument("--from-cache", default=None)
    p_study_a.add_argument("--cache-out", default=None)

    p_study_b = sub.add_parser("study-b", parents=[common])
    p_study_b.add_argument("--max-samples", type=int, default=None)
    p_study_b.add_argument(
        "--data-dir",
        default=str(runtime_root / "data" / "releases" / "clinician_readiness_v4_2026-02-22" / "openr1_psy_splits"),
    )
    p_study_b.add_argument(
        "--output-dir",
        default=str(runtime_root / "results"),
    )
    p_study_b.add_argument("--cache-out", default=None)

    args = parser.parse_args()

    runner = MedGemmaLMStudioRunner(
        model_name=args.api_identifier,
        api_base=args.api_base,
        config=GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        ),
    )
    worker_count = resolve_worker_count(args.workers, runner, lmstudio_default=4, non_lm_default=1)

    if args.cmd == "prompt":
        print(runner.generate(args.prompt, mode=args.mode))
        return

    if args.cmd == "study-a":
        cache_out = args.cache_out
        if cache_out is None and args.generate_only:
            cache_out = str(Path(args.output_dir) / args.model_name / "study_a_generations.jsonl")
        run_study_a(
            model=runner,
            data_dir=args.data_dir,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            model_name=args.model_name,
            generate_only=bool(args.generate_only),
            from_cache=args.from_cache,
            cache_out=cache_out,
            workers=worker_count,
            progress_interval_seconds=args.progress_interval_seconds,
        )
        return

    cache_out = args.cache_out
    if cache_out is None:
        cache_out = str(Path(args.output_dir) / args.model_name / "study_b_generations.jsonl")
    run_study_b(
        model=runner,
        data_dir=args.data_dir,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        model_name=args.model_name,
        use_nli=False,
        generate_only=True,
        cache_out=cache_out,
        workers=worker_count,
        progress_interval_seconds=args.progress_interval_seconds,
    )


if __name__ == "__main__":
    main()
