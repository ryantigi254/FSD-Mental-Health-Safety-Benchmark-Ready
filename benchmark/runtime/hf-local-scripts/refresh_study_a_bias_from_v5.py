import argparse
import importlib.util
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path


def _load_bias_generation_module(runtime_root: Path):
    script_path = runtime_root / "hf-local-scripts" / "run_study_a_bias_generate_only.py"
    spec = importlib.util.spec_from_file_location("study_a_bias_generate_only_runtime", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Study A bias generation script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh stale Study A bias generations by re-running ids whose v5 source rows changed."
    )
    parser.add_argument("--model-id", type=str, required=True, help="Model ID/alias used to instantiate the runner.")
    parser.add_argument("--model", type=str, default=None, help="Optional HF model path override for local models.")
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Optional quantisation override for local Psych-Qwen models.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to v5 adversarial bias data (defaults to runtime/data/frozen_splits/v5/adversarial_bias/biased_vignettes.json).",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Existing study_a_bias_generations.jsonl to refresh in place.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max new tokens per refreshed generation (default: 8192).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel refresh workers. Default is auto: 4 for LM Studio models, else 1.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=int,
        default=10,
        help="Heartbeat interval while waiting for worker completions.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit on how many stale ids to regenerate after diffing.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create <cache>.bak before rewriting the refreshed JSONL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report stale ids without running generation or rewriting the file.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map strategy for local Psych-Qwen loading.",
    )
    parser.add_argument(
        "--max-memory-gpu",
        type=str,
        default="22GiB",
        help="Maximum GPU memory to use for local Psych-Qwen loading.",
    )
    parser.add_argument(
        "--max-memory-cpu",
        type=str,
        default="48GiB",
        help="Maximum CPU memory to use for local Psych-Qwen loading.",
    )
    parser.add_argument(
        "--offload-folder",
        type=str,
        default="./offload_tmp",
        help="Folder for disk offloading when GPU/CPU memory is insufficient.",
    )
    parser.add_argument(
        "--low-cpu-mem",
        action="store_true",
        default=True,
        help="Use low CPU memory mode during local model loading.",
    )
    parser.add_argument(
        "--no-low-cpu-mem",
        action="store_true",
        default=False,
        help="Disable low CPU memory mode.",
    )
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def main() -> None:
    runtime_root = Path(__file__).resolve().parents[1]
    bias_gen = _load_bias_generation_module(runtime_root)

    bias_gen._ensure_src_on_path(runtime_root)
    bias_gen._ensure_hf_cache_under_models_dir(runtime_root)

    from reliable_clinical_benchmark.models.base import GenerationConfig
    from reliable_clinical_benchmark.models.factory import get_model_runner
    from reliable_clinical_benchmark.study_a_bias_refresh import (
        find_stale_case_ids,
        load_bias_case_map,
        load_jsonl_rows,
        replace_rows_by_id,
        write_jsonl_rows_atomic,
    )

    args = _parse_args()

    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = runtime_root / "data" / "frozen_splits" / "v5" / "adversarial_bias" / "biased_vignettes.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Bias data not found at {data_path}")

    if args.cache_path:
        cache_path = Path(args.cache_path)
    else:
        cache_path = (
            runtime_root
            / "results"
            / bias_gen._canonical_model_output_dir(args.model_id)
            / "study_a_bias_generations.jsonl"
        )
    if not cache_path.exists():
        raise FileNotFoundError(f"Existing Study A bias cache not found at {cache_path}")

    existing_rows = load_jsonl_rows(cache_path)
    case_map = load_bias_case_map(data_path)
    stale_case_ids = find_stale_case_ids(existing_rows, case_map)

    if args.max_cases is not None:
        stale_case_ids = stale_case_ids[: max(0, args.max_cases)]

    print(f"Existing rows: {len(existing_rows)}")
    print(f"Stale ids detected: {len(stale_case_ids)}")
    if stale_case_ids:
        print("Stale ids:", ", ".join(stale_case_ids))
    if not stale_case_ids:
        print("No stale Study A bias rows detected.")
        return
    if args.dry_run:
        print("Dry run only; no generations were refreshed.")
        return

    model_id_lower = args.model_id.lower()
    runner = None
    if model_id_lower in ("psyllm", "psyllm_gml_local", "psyllm-gml-local", "psyllm-gmlhuhe-local", "gmlhuhe_psyllm_local"):
        from reliable_clinical_benchmark.models.psyllm_gml_local import PsyLLMGMLLocalRunner

        model_path = args.model or bias_gen._resolve_local_model_path("PsyLLM", runtime_root)
        runner = PsyLLMGMLLocalRunner(model_name=model_path, config=GenerationConfig(max_tokens=args.max_tokens))
    elif model_id_lower in ("piaget_local", "piaget-8b-local", "piaget8b-local"):
        from reliable_clinical_benchmark.models.piaget_local import Piaget8BLocalRunner

        model_path = args.model or bias_gen._resolve_local_model_path("Piaget-8B", runtime_root)
        runner = Piaget8BLocalRunner(model_name=model_path, config=GenerationConfig(max_tokens=args.max_tokens))
    elif model_id_lower in ("psyche_r1_local", "psyche-r1-local", "psyche-r1-local-hf"):
        from reliable_clinical_benchmark.models.psyche_r1_local import PsycheR1LocalRunner

        model_path = args.model or bias_gen._resolve_local_model_path("Psyche-R1", runtime_root)
        runner = PsycheR1LocalRunner(model_name=model_path, config=GenerationConfig(max_tokens=args.max_tokens))
    elif model_id_lower in ("psych_qwen_local", "psych-qwen-32b-local", "psych-qwen-local-hf"):
        from reliable_clinical_benchmark.models.psych_qwen_local import PsychQwen32BLocalRunner

        model_path = args.model or bias_gen._resolve_local_model_path("Psych_Qwen_32B", runtime_root)
        quantization = args.quantization or "4bit"
        max_memory = {0: args.max_memory_gpu, "cpu": args.max_memory_cpu}
        runner = PsychQwen32BLocalRunner(
            model_name=model_path,
            quantization=quantization,
            device_map=args.device_map,
            max_memory=max_memory,
            offload_folder=args.offload_folder,
            config=GenerationConfig(max_tokens=args.max_tokens),
        )
    if runner is None:
        runner = get_model_runner(args.model_id, GenerationConfig(max_tokens=args.max_tokens))

    is_lmstudio_runner = hasattr(runner, "api_base")
    if args.workers is None:
        worker_count = 4 if is_lmstudio_runner else 1
    else:
        worker_count = max(1, int(args.workers))
    if worker_count > 1 and not is_lmstudio_runner:
        print("Parallel workers >1 are only enabled for LM Studio runners. Falling back to 1 worker.")
        worker_count = 1

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    stale_cases = [case_map[case_id] for case_id in stale_case_ids]
    current_rows = list(existing_rows)

    if args.backup:
        backup_path = cache_path.with_suffix(cache_path.suffix + ".bak")
        shutil.copy2(cache_path, backup_path)
        print(f"Backup written to {backup_path}")

    def _persist_replacement(entry: dict) -> None:
        nonlocal current_rows
        current_rows, replaced_count = replace_rows_by_id(current_rows, {entry["id"]: entry})
        write_jsonl_rows_atomic(cache_path, current_rows)
        print(f"[persisted] {entry['id']} replaced_rows={replaced_count}")

    def _generate_case_entry(case: dict) -> dict:
        case_id = str(case.get("id", "")).strip()
        prompt_text = case.get("prompt", "")
        formatted_prompt = bias_gen.format_bias_prompt(prompt_text)
        started = time.time()
        status = "ok"
        output_text = ""
        error_message = ""
        try:
            output_text = runner.generate(formatted_prompt, mode="cot")
        except Exception as exc:  # pragma: no cover - exercised against live runners
            status = "error"
            error_message = str(exc)
        latency_ms = int((time.time() - started) * 1000)
        return {
            "id": case_id,
            "bias_feature": case.get("bias_feature", ""),
            "bias_label": case.get("bias_label", ""),
            "prompt": formatted_prompt,
            "output_text": output_text,
            "status": status,
            "error_message": error_message,
            "timestamp": _now_iso(),
            "run_id": run_id,
            "model_name": args.model_id,
            "metadata": case.get("metadata", {}) or {},
            "sampling": {
                "temperature": runner.config.temperature,
                "top_p": runner.config.top_p,
                "max_tokens": runner.config.max_tokens,
            },
            "meta": {"latency_ms": latency_ms},
        }

    replacement_rows = {}
    if worker_count == 1:
        for index, case in enumerate(stale_cases, start=1):
            entry = _generate_case_entry(case)
            replacement_rows[entry["id"]] = entry
            _persist_replacement(entry)
            print(
                f"[refreshed {index}/{len(stale_cases)}] "
                f"{entry['id']} status={entry['status']} latency_ms={entry['meta']['latency_ms']}"
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            pending_iter = iter(stale_cases)
            future_to_id = {}
            for _ in range(worker_count):
                try:
                    case = next(pending_iter)
                except StopIteration:
                    break
                future = executor.submit(_generate_case_entry, case)
                future_to_id[future] = str(case.get("id", "")).strip()

            saved_count = 0
            heartbeat_interval = max(1, int(args.progress_interval_seconds))
            while future_to_id:
                completed_now, _ = wait(
                    set(future_to_id.keys()),
                    timeout=heartbeat_interval,
                    return_when=FIRST_COMPLETED,
                )
                if not completed_now:
                    print(
                        f"[progress] refreshed={saved_count}/{len(stale_cases)} "
                        f"in_flight={len(future_to_id)}"
                    )
                    continue
                for completed_future in completed_now:
                    case_id = future_to_id.pop(completed_future)
                    try:
                        entry = completed_future.result()
                    except Exception as exc:  # pragma: no cover - exercised against live runners
                        entry = {
                            "id": case_id,
                            "bias_feature": "",
                            "bias_label": "",
                            "prompt": "",
                            "output_text": "",
                            "status": "error",
                            "error_message": f"Worker failure: {exc}",
                            "timestamp": _now_iso(),
                            "run_id": run_id,
                            "model_name": args.model_id,
                            "metadata": {},
                            "sampling": {
                                "temperature": runner.config.temperature,
                                "top_p": runner.config.top_p,
                                "max_tokens": runner.config.max_tokens,
                            },
                            "meta": {"latency_ms": 0},
                        }
                    replacement_rows[entry["id"]] = entry
                    _persist_replacement(entry)
                    saved_count += 1
                    print(
                        f"[refreshed {saved_count}/{len(stale_cases)}] "
                        f"{entry['id']} status={entry['status']} latency_ms={entry['meta']['latency_ms']}"
                    )
                    try:
                        next_case = next(pending_iter)
                    except StopIteration:
                        continue
                    future = executor.submit(_generate_case_entry, next_case)
                    future_to_id[future] = str(next_case.get("id", "")).strip()

    print(f"Refreshed unique ids: {len(replacement_rows)}")
    print(f"Updated cache: {cache_path}")


if __name__ == "__main__":
    main()
