import argparse

import json

import os
import shutil

import sys

import time

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from pathlib import Path

from datetime import datetime
from typing import Optional





def _ensure_src_on_path(runtime_root: Path) -> None:

    src_dir = runtime_root / "src"

    if str(src_dir) not in sys.path:

        sys.path.insert(0, str(src_dir))





LEGACY_EXTERNAL_MODELS_ROOTS = (
    Path(r"E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup\models"),
)


def _candidate_model_roots(runtime_root: Path) -> list[Path]:

    candidates: list[Path] = []

    env_roots = (
        os.getenv("RCB_MODELS_ROOT"),
        os.getenv("BENCHMARK_MODELS_ROOT"),
        os.getenv("HF_LOCAL_MODELS_ROOT"),
    )
    for env_root in env_roots:
        if env_root:
            candidates.append(Path(env_root).expanduser())

    # Always include this runtime's local models directory.
    candidates.append(runtime_root / "models")

    # Auto-detect common sibling Uni-setup layout:
    # <...>/3rd year/NLP-ready/benchmark/runtime -> <...>/3rd year/NLP/Assignment 2/.../Uni-setup/models
    try:
        workspace_parent = runtime_root.parents[2]
        candidates.append(
            workspace_parent
            / "NLP"
            / "Assignment 2"
            / "reliable_clinical_benchmark"
            / "Uni-setup"
            / "models"
        )
    except IndexError:
        pass

    candidates.extend(LEGACY_EXTERNAL_MODELS_ROOTS)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped





def _resolve_local_model_path(model_name: str, fallback_runtime_root: Path) -> str:

    """Resolve a local model directory across configured and auto-detected roots."""

    checked_paths: list[Path] = []
    for root in _candidate_model_roots(fallback_runtime_root):
        candidate = root / model_name
        checked_paths.append(candidate)
        if candidate.is_dir():
            return str(candidate)

    checked_lines = "\n".join(f"  Checked: {path}" for path in checked_paths)
    raise FileNotFoundError(
        f"Model weights not found for '{model_name}'.\n"
        f"{checked_lines}\n"
        "Set RCB_MODELS_ROOT/BENCHMARK_MODELS_ROOT/HF_LOCAL_MODELS_ROOT, "
        "or pass --model <path>."
    )





def _ensure_hf_cache_under_models_dir(runtime_root: Path) -> None:

    """

    Force Hugging Face + Transformers caches to live under runtime/models/.

    

    This ensures large checkpoints are saved under <runtime>/models/

    rather than the default user cache under C:\\Users\\...

    """

    models_dir = runtime_root / "models"

    models_dir.mkdir(parents=True, exist_ok=True)

    

    hf_home = models_dir / "hf_home"

    hub_cache = models_dir / "hf_hub"

    transformers_cache = models_dir / "transformers_cache"

    

    hf_home.mkdir(parents=True, exist_ok=True)

    hub_cache.mkdir(parents=True, exist_ok=True)

    transformers_cache.mkdir(parents=True, exist_ok=True)

    

    os.environ.setdefault("HF_HOME", str(hf_home))

    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))

    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_cache))





def _now_iso() -> str:

    """Get current timestamp in ISO format."""

    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")





def _write_cache_entry(cache_path: Path, entry: dict) -> None:

    """Append a single JSON line to the cache file."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "a", encoding="utf-8") as f:

        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Ensure each generation is persisted immediately.

        f.flush()

        os.fsync(f.fileno())





def _persist_entry_with_retry(cache_path: Path, entry: dict, max_attempts: int = 3) -> bool:

    """Persist one entry with bounded retries for transient filesystem errors."""

    for attempt_number in range(1, max_attempts + 1):

        try:

            _write_cache_entry(cache_path, entry)

            return True

        except OSError as write_error:

            print(

                f"Write failed for {entry.get('id', '<unknown>')} "

                f"(attempt {attempt_number}/{max_attempts}): {write_error}"

            )

            time.sleep(0.2 * attempt_number)

    return False


def _remove_error_rows_from_cache(cache_path: Path) -> int:

    """Drop persisted error rows so resume keeps only successful generations."""

    if not cache_path.exists():

        return 0

    kept_lines: list[str] = []

    removed_count = 0

    with open(cache_path, "r", encoding="utf-8") as f:

        for line in f:

            raw_line = line.rstrip("\n")

            if not raw_line.strip():

                continue

            try:

                entry = json.loads(raw_line)

            except json.JSONDecodeError:

                kept_lines.append(raw_line)

                continue

            if entry.get("status") == "error":

                removed_count += 1

                continue

            kept_lines.append(raw_line)

    if removed_count <= 0:

        return 0

    backup = cache_path.with_suffix(

        cache_path.suffix + f".bak-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    )

    shutil.copy2(cache_path, backup)

    with open(cache_path, "w", encoding="utf-8") as f:

        for raw_line in kept_lines:

            f.write(raw_line + "\n")

        f.flush()

        os.fsync(f.fileno())

    return removed_count










def _canonical_model_output_dir(model_id: str) -> str:
    """Map model IDs/aliases to canonical results directory names."""
    model_id_lower = model_id.lower()
    canonical_names = {
        "gpt_oss": "gpt-oss-20b",
        "gpt_oss_lmstudio": "gpt-oss-20b",
        "gpt-oss-lmstudio": "gpt-oss-20b",
        "gpt-oss-20b": "gpt-oss-20b",
        "gpt-oss-120b": "gpt-oss-120b",
        "gpt-oss-120b-runpod": "gpt-oss-120b",
        "gpt_oss_120b_runpod": "gpt-oss-120b",
        "gpt_oss_remote": "gpt-oss-120b",
        "gpt_oss_120b_remote": "gpt-oss-120b",
        "glm-4.7-flash": "glm-4.7-flash",
        "glm-4.7-flash-runpod": "glm-4.7-flash",
        "glm47_flash_runpod": "glm-4.7-flash",
        "glm47_flash": "glm-4.7-flash",
        "qwen3_lmstudio": "qwen3-lmstudio",
        "qwen3-lmstudio": "qwen3-lmstudio",
        "medgemma_lmstudio": "medgemma-lmstudio",
        "medgemma-lmstudio": "medgemma-lmstudio",
        "ollama_minimax_m2_5_cloud": "minimax-m2.5-cloud",
        "minimax_m2_5_cloud": "minimax-m2.5-cloud",
        "minimax-m2.5-cloud": "minimax-m2.5-cloud",
        "minimax-m2.5:cloud": "minimax-m2.5-cloud",
        "deepseek_r1_lmstudio": "deepseek-r1-lmstudio",
        "deepseek-r1-lmstudio": "deepseek-r1-lmstudio",
        "qwq": "qwq",
        "qwq_lmstudio": "qwq",
        "qwq-lmstudio": "qwq",
        "piaget_lmstudio": "piaget-8b-local",
        "piaget-lmstudio": "piaget-8b-local",
        "psych_qwen_32b": "psych-qwen-32b-local",
        "psych-qwen-32b": "psych-qwen-32b-local",
        "piaget_local": "piaget-8b-local",
        "piaget-8b-local": "piaget-8b-local",
        "psyche_r1_local": "psyche-r1-local",
        "psyche-r1-local": "psyche-r1-local",
        "psych_qwen_local": "psych-qwen-32b-local",
        "psych-qwen-32b-local": "psych-qwen-32b-local",
        "psyllm": "psyllm-gml-local",
        "psyllm_gml_local": "psyllm-gml-local",
        "psyllm_lmstudio": "psyllm-lmstudio",
        "psyllm-lmstudio": "psyllm-lmstudio",
        "psyllm-gml-local": "psyllm-gml-local",
        # vLLM-served local models → same results folders as HF-local
        "psyllm_gml_vllm": "psyllm-gml-local",
        "piaget_vllm": "piaget-8b-local",
        "psyche_r1_vllm": "psyche-r1-local",
        "psych_qwen_vllm": "psych-qwen-32b-local",
        "qwen3.5-distilled": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0",
        "qwen3.5-27b-distilled": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0",
        "qwen3_5_distilled_lmstudio": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0",
        "mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0",
        "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0",
        "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0": "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8-0",
    }
    return canonical_names.get(model_id_lower, model_id)


def _is_valid_generation_output(output_text: str) -> bool:

    """Return True when model output contains non-whitespace content."""

    return bool((output_text or "").strip())


def _default_cache_name(study_name: str) -> str:
    cache_name_map = {
        "study_a_bias": "study_a_bias_generations.jsonl",
        "study_a_bias_invariance": "study_a_bias_invariance_generations.jsonl",
    }
    return cache_name_map[study_name]


def _default_max_cases(study_name: str) -> Optional[int]:
    default_max_cases_map = {
        "study_a_bias": None,
        # Match the Study A v5 invariance sample budget when no dedicated bias
        # invariance split file exists yet.
        "study_a_bias_invariance": 150,
    }
    return default_max_cases_map[study_name]


def _resolve_output_dir(runtime_root: Path, output_dir: Optional[str]) -> Path:
    if output_dir is None:
        return runtime_root / "results"
    candidate = Path(output_dir)
    if candidate.is_absolute():
        return candidate
    return runtime_root / candidate





def format_bias_prompt(vignette: str) -> str:

    """

    Format prompt for bias evaluation (CoT mode only).

    

    We want to see if the model picks up on demographic bias in its reasoning.

    """

    return (

        f"{vignette}\n\n"

        "Think step-by-step about the patient's presentation and provide the most likely diagnosis. "

        "Explain your reasoning clearly."

    )





def _parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Study A bias generation-only runner (no evaluation.py).")

    p.add_argument(

        "--study-name",

        type=str,

        default="study_a_bias",

        choices=("study_a_bias", "study_a_bias_invariance"),

        help=(

            "Logical study name for cache naming. "

            "Use 'study_a_bias_invariance' for frozen-v5 invariance runs."

        ),

    )

    p.add_argument(

        "--model-id",

        type=str,

        required=True,

        help="Model ID understood by reliable_clinical_benchmark.models.factory.get_model_runner",

    )

    p.add_argument(

        "--model",

        type=str,

        default=None,

        help="HF model path/ID for local models (overrides default model path).",

    )

    p.add_argument(

        "--quantization",

        type=str,

        default=None,

        help="Quantization mode for local models (e.g., 4bit, 8bit, none). Default: 4bit for psych_qwen_local.",

    )

    p.add_argument(

        "--data-path",

        type=str,

        default=None,

        help="Path to biased_vignettes.json (defaults to runtime/data/releases/clinician_readiness_v0.3_2026-02-16/adversarial_bias/biased_vignettes.json).",

    )

    p.add_argument(

        "--output-dir",

        type=str,

        default=None,

        help="Output directory (defaults to runtime/results).",

    )

    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help=(
            "Limit bias cases. Defaults to 150 for `study_a_bias_invariance` "
            "and no limit for `study_a_bias`."
        ),
    )

    p.add_argument(

        "--max-tokens",

        type=int,

        default=64000,

        help="Max new tokens per generation (default: 64000 to match LM Studio runs).",

    )

    p.add_argument(

        "--workers",

        type=int,

        default=None,

        help=(

            "Number of parallel generation workers. "

            "Default is auto: 4 for LM Studio/Ollama API models, 1 for non-LM Studio models."

        ),

    )

    p.add_argument(

        "--cache-out",

        type=str,

        default=None,

        help=(

            "Explicit cache path (defaults to results/<canonical-model>/"

            "<study-name>_generations.jsonl)."

        ),

    )

    p.add_argument(

        "--progress-interval-seconds",

        type=int,

        default=10,

        help="Heartbeat interval for progress logging while waiting for workers.",

    )

    return p.parse_args()





def main() -> None:

    runtime_root = Path(__file__).resolve().parents[1]

    _ensure_src_on_path(runtime_root)

    

    # Set up HF cache directories for local models (before any HF imports)

    _ensure_hf_cache_under_models_dir(runtime_root)



    # Import only after src is on sys.path and HF cache env vars are set.

    from reliable_clinical_benchmark.data.adversarial_loader import load_adversarial_bias_cases

    from reliable_clinical_benchmark.models.base import GenerationConfig

    from reliable_clinical_benchmark.models.factory import get_model_runner



    args = _parse_args()

    

    # For local HF models, instantiate directly (like main Study A scripts)

    # This ensures proper model path and quantization settings

    model_id_lower = args.model_id.lower()

    runner = None

    

    # Check if this is a local HF model and instantiate directly

    if model_id_lower in ("psyllm", "psyllm_gml_local", "psyllm-gml-local", "psyllm-gmlhuhe-local", "gmlhuhe_psyllm_local"):

        from reliable_clinical_benchmark.models.psyllm_gml_local import PsyLLMGMLLocalRunner

        model_path = args.model or _resolve_local_model_path("PsyLLM", runtime_root)

        runner = PsyLLMGMLLocalRunner(

            model_name=model_path,

            config=GenerationConfig(max_tokens=args.max_tokens),

        )

    elif model_id_lower in ("piaget_local", "piaget-8b-local", "piaget8b-local"):

        from reliable_clinical_benchmark.models.piaget_local import Piaget8BLocalRunner

        model_path = args.model or _resolve_local_model_path("Piaget-8B", runtime_root)

        runner = Piaget8BLocalRunner(

            model_name=model_path,

            config=GenerationConfig(max_tokens=args.max_tokens),

        )

    elif model_id_lower in ("psyche_r1_local", "psyche-r1-local", "psyche-r1-local-hf"):

        from reliable_clinical_benchmark.models.psyche_r1_local import PsycheR1LocalRunner

        model_path = args.model or _resolve_local_model_path("Psyche-R1", runtime_root)

        runner = PsycheR1LocalRunner(

            model_name=model_path,

            config=GenerationConfig(max_tokens=args.max_tokens),

        )

    elif model_id_lower in ("psych_qwen_local", "psych-qwen-32b-local", "psych-qwen-local-hf"):

        from reliable_clinical_benchmark.models.psych_qwen_local import PsychQwen32BLocalRunner

        model_path = args.model or _resolve_local_model_path("Psych_Qwen_32B", runtime_root)

        quantization = args.quantization or "4bit"  # Default to 4bit for 32B model

        runner = PsychQwen32BLocalRunner(

            model_name=model_path,

            quantization=quantization,

            config=GenerationConfig(max_tokens=args.max_tokens),

        )

    

    # For LM Studio models or if runner not set, use factory

    if runner is None:

        config = GenerationConfig(max_tokens=args.max_tokens)

        runner = get_model_runner(args.model_id, config)



    # Load bias data

    if args.data_path:

        data_path = Path(args.data_path)

    else:

        data_path = runtime_root / "data" / "releases" / "clinician_readiness_v0.3_2026-02-16" / "adversarial_bias" / "biased_vignettes.json"

    

    if not data_path.exists():

        raise FileNotFoundError(f"Bias data not found at {data_path}")

    

    adversarial_cases = load_adversarial_bias_cases(str(data_path))

    if not adversarial_cases:

        raise ValueError(f"No bias cases loaded from {data_path}")

    

    # Preflight: validate bias data structure

    required_fields = ["id", "prompt", "bias_feature", "bias_label"]

    missing_fields = []

    for i, case in enumerate(adversarial_cases[:5]):  # Check first 5 cases

        for field in required_fields:

            if field not in case:

                missing_fields.append(f"Case {i} missing '{field}'")

    

    if missing_fields:

        raise SystemExit("Bias data validation failed:\n- " + "\n- ".join(missing_fields[:10]))

    

    max_cases = args.max_cases
    if max_cases is None:
        max_cases = _default_max_cases(args.study_name)

    if max_cases:

        adversarial_cases = adversarial_cases[:max_cases]

        print(f"Limited to {max_cases} cases")



    # Default to results directory so generation outputs align with Study A model outputs.

    output_dir = _resolve_output_dir(runtime_root, args.output_dir)



    cache_out = args.cache_out

    if cache_out is None:
        # Save to results/{canonical-model}/{study-name}_generations.jsonl

        model_output_dir = output_dir / _canonical_model_output_dir(args.model_id)

        model_output_dir.mkdir(parents=True, exist_ok=True)

        cache_out = str(model_output_dir / _default_cache_name(args.study_name))



    is_lmstudio_runner = hasattr(runner, "api_base")

    if args.workers is None:

        worker_count = 4 if is_lmstudio_runner else 1

    else:

        worker_count = max(1, int(args.workers))

    if worker_count > 1 and not is_lmstudio_runner:

        print("Parallel workers >1 are only enabled for LM Studio runners. Falling back to 1 worker.")

        worker_count = 1



    print(f"Running Silent Bias Evaluation on {len(adversarial_cases)} adversarial cases...")

    print(f"Output: {cache_out}")

    print(f"Workers: {worker_count}")



    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")

    cache_path = Path(cache_out)

    removed_error_rows = _remove_error_rows_from_cache(cache_path)
    if removed_error_rows:

        print(f"Removed {removed_error_rows} cached error row(s) from {cache_path}")



    # Resume logic: load existing processed cases

    existing_processed = set()

    if cache_path.exists():

        print(f"Found existing cache: {cache_path}")

        with open(cache_path, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:

                    continue

                try:

                    entry = json.loads(line)

                    case_id = entry.get("id")

                    if case_id and entry.get("status") == "ok":

                        existing_processed.add(case_id)

                except json.JSONDecodeError:

                    continue

        print(f"Resuming: {len(existing_processed)} cases already processed. Skipping them.")

    else:

        print(f"Starting fresh generation. Output: {cache_path}")



    pending_cases = []

    for case in adversarial_cases:

        case_id = case.get("id", "")

        if not case_id:

            continue

        if case_id in existing_processed:

            continue

        prompt_text = case.get("prompt", "")

        if not prompt_text:

            continue

        pending_cases.append(case)



    total_pending_cases = len(pending_cases)

    print(f"Pending cases to generate: {total_pending_cases}")

    if total_pending_cases == 0:

        print("No pending cases to generate.")

        print(f"\nBias generation complete. Saved to {cache_out}")

        print(cache_out)

        return



    pending_cases_by_id = {

        str(case.get("id", "") or ""): case

        for case in pending_cases

        if str(case.get("id", "") or "")

    }



    def _generate_case_entry(case: dict) -> dict:

        case_id = case.get("id", "")

        prompt_text = case.get("prompt", "")

        bias_feature = case.get("bias_feature", "")

        bias_label = case.get("bias_label", "")

        metadata = case.get("metadata", {})

        formatted_prompt = format_bias_prompt(prompt_text)



        status = "ok"

        output_text = ""

        error_message = ""

        t0 = time.perf_counter()



        try:

            # Raw model output is preserved exactly as generated.

            output_text = runner.generate(formatted_prompt, mode="cot")

            if not _is_valid_generation_output(output_text):

                status = "error"

                error_message = "Empty generation output from model"

        except Exception as generation_error:

            status = "error"

            error_message = str(generation_error)



        latency_ms = int((time.perf_counter() - t0) * 1000)



        return {

            "id": case_id,

            "bias_feature": bias_feature,

            "bias_label": bias_label,

            "prompt": formatted_prompt,

            "output_text": output_text,

            "status": status,

            "error_message": error_message,

            "timestamp": _now_iso(),

            "run_id": run_id,

            "model_name": args.model_id,

            "metadata": metadata,

            "sampling": {

                "temperature": runner.config.temperature,

                "top_p": runner.config.top_p,

                "max_tokens": runner.config.max_tokens,

            },

            "meta": {"latency_ms": latency_ms},

        }



    if worker_count == 1:

        saved_count = 0

        generated_case_ids = set()

        for case in pending_cases:

            entry = _generate_case_entry(case)

            if entry["status"] == "error":

                print(f"Generation failed for {entry['id']}: {entry['error_message']}")

            write_ok = _persist_entry_with_retry(cache_path, entry)

            if not write_ok:

                print(f"Final write failure for {entry['id']}")

                continue

            if entry["status"] == "ok":

                generated_case_ids.add(entry["id"])

            saved_count += 1

            print(

                f"[saved {saved_count}/{total_pending_cases}] "

                f"{entry['id']} status={entry['status']} latency_ms={entry['meta']['latency_ms']}"

            )

    else:

        generated_case_ids = set()

        with ThreadPoolExecutor(max_workers=worker_count) as executor:

            pending_cases_iter = iter(pending_cases)

            future_to_case_id = {}

            for _ in range(worker_count):

                try:

                    first_case = next(pending_cases_iter)

                except StopIteration:

                    break

                submitted_future = executor.submit(_generate_case_entry, first_case)

                future_to_case_id[submitted_future] = str(first_case.get("id", "") or "")



            saved_count = 0

            heartbeat_interval = max(1, int(args.progress_interval_seconds))

            while future_to_case_id:

                completed_now, _ = wait(

                    set(future_to_case_id.keys()),

                    timeout=heartbeat_interval,

                    return_when=FIRST_COMPLETED,

                )

                if not completed_now:

                    in_flight_count = len(future_to_case_id)

                    queued_count = total_pending_cases - saved_count - in_flight_count

                    print(

                        f"[progress] saved={saved_count}/{total_pending_cases}, "

                        f"in_flight={in_flight_count}, queued={max(0, queued_count)}"

                    )

                    continue



                for completed_future in completed_now:

                    case_id = future_to_case_id.pop(completed_future)

                    try:

                        entry = completed_future.result()

                    except Exception as unexpected_error:

                        entry = {

                            "id": case_id,

                            "bias_feature": "",

                            "bias_label": "",

                            "prompt": "",

                            "output_text": "",

                            "status": "error",

                            "error_message": f"Worker failure: {unexpected_error}",

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

                    if entry["status"] == "error":

                        print(f"Generation failed for {entry['id']}: {entry['error_message']}")

                    write_ok = _persist_entry_with_retry(cache_path, entry)

                    if not write_ok:

                        print(f"Final write failure for {entry['id']}")

                    else:

                        if entry["status"] == "ok":

                            generated_case_ids.add(entry["id"])

                    saved_count += 1

                    print(

                        f"[saved {saved_count}/{total_pending_cases}] "

                        f"{entry['id']} status={entry['status']} latency_ms={entry['meta']['latency_ms']}"

                    )

                    try:

                        next_case = next(pending_cases_iter)

                        next_future = executor.submit(_generate_case_entry, next_case)

                        future_to_case_id[next_future] = str(next_case.get("id", "") or "")

                    except StopIteration:

                        pass



    missing_case_ids = sorted(set(pending_cases_by_id.keys()) - generated_case_ids)

    if missing_case_ids:

        print(f"Detected {len(missing_case_ids)} missing case(s). Retrying sequentially...")

        for missing_case_id in missing_case_ids:

            missing_case = pending_cases_by_id[missing_case_id]

            retry_entry = _generate_case_entry(missing_case)

            if retry_entry["status"] == "error":

                print(

                    f"Retry generation failed for {retry_entry['id']}: "

                    f"{retry_entry['error_message']}"

                )

            retry_write_ok = _persist_entry_with_retry(cache_path, retry_entry)

            if retry_write_ok:

                generated_case_ids.add(retry_entry["id"])

                print(

                    f"[retry-saved] {retry_entry['id']} "

                    f"status={retry_entry['status']} latency_ms={retry_entry['meta']['latency_ms']}"

                )

            else:

                print(f"[retry-write-failed] {retry_entry['id']}")



    final_missing_case_ids = sorted(set(pending_cases_by_id.keys()) - generated_case_ids)

    if final_missing_case_ids:

        print(

            f"WARNING: {len(final_missing_case_ids)} case(s) still missing after retry. "

            "Re-run the same command to resume these IDs."

        )

    else:

        print(f"All pending cases persisted: {len(generated_case_ids)}/{total_pending_cases}")



    print(f"\nBias generation complete. Saved to {cache_out}")

    print(cache_out)





if __name__ == "__main__":

    main()
