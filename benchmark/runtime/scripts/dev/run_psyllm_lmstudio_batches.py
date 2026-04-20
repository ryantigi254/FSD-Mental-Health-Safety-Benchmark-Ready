#!/usr/bin/env python3
"""Run PsyLLM study batches sequentially from a single command.

This stays self-contained:
- no command-doc edits
- no repo-wide model registration changes
- controllability arm filtering is handled here

The default `--model-id` is `psyllm_gml_vllm` because the existing study
scripts already treat it as a server-side model id. This runner patches that id
to use LM Studio instead of vLLM, so you can run everything from
`mh-llm-benchmark-env` against a model loaded in LM Studio.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Sequence


ARM_SPONTANEOUS = "spontaneous"
ARM_GENERIC = "generic_control"
ARM_EXPLICIT = "explicit_control"
VALID_ARMS = (ARM_SPONTANEOUS, ARM_GENERIC, ARM_EXPLICIT)

PSYLLM_LMSTUDIO_ALIASES = {
    "psyllm_gml_vllm",
    "psyllm_lmstudio",
    "psyllm-gml-lmstudio",
    "psyllm-gml-vllm",
}


@dataclass(frozen=True)
class Job:
    label: str
    script_relative_path: str
    argv: tuple[str, ...]
    supports_arm_filter: bool = False


def _runtime_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_src_on_path(runtime_root: Path) -> None:
    src_dir = runtime_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _parse_arms(raw_value: str) -> tuple[str, ...]:
    value = (raw_value or "all").strip().lower()
    if value == "all":
        return VALID_ARMS
    arms = tuple(part.strip() for part in value.split(",") if part.strip())
    if not arms:
        raise SystemExit("--arms resolved to an empty selection.")
    invalid = sorted(set(arms) - set(VALID_ARMS))
    if invalid:
        raise SystemExit(
            f"Unsupported arm(s): {', '.join(invalid)}. Valid values: {', '.join(VALID_ARMS)} or all."
        )
    return arms


def _load_module(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_psyllm_lmstudio_patch(
    runtime_root: Path,
    *,
    model_name: str,
    api_base: str,
) -> Callable:
    _ensure_src_on_path(runtime_root)

    from reliable_clinical_benchmark.models.base import GenerationConfig, ModelRunner
    from reliable_clinical_benchmark.models.lmstudio_client import chat_completion
    import reliable_clinical_benchmark.models.factory as factory

    class PsyLLMLMStudioRunner(ModelRunner):
        """LM Studio-backed PsyLLM runner injected by the batch launcher."""

        def __init__(
            self,
            model_name_override: str | None = None,
            api_base_override: str | None = None,
            config: GenerationConfig | None = None,
        ) -> None:
            resolved_model_name = model_name_override or model_name
            super().__init__(
                resolved_model_name,
                config or GenerationConfig(temperature=0.7, top_p=0.9, max_tokens=None),
            )
            self.api_base = api_base_override or api_base

        def generate(self, prompt: str, mode: str = "default") -> str:
            formatted_prompt = self._format_prompt(prompt, mode)
            return chat_completion(
                api_base=self.api_base,
                model=self.model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                timeout=None,
            )

        def generate_with_reasoning(self, prompt: str) -> tuple[str, str]:
            full_response = self.generate(prompt, mode="cot")
            return full_response, full_response

        def chat(self, messages: list[dict[str, str]], mode: str = "default") -> str:
            formatted_messages: list[dict[str, str]] = []
            for index, message in enumerate(messages):
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "system":
                    formatted_messages.append({"role": "system", "content": content})
                    continue
                if role == "assistant":
                    formatted_messages.append({"role": "assistant", "content": content})
                    continue
                if index == len(messages) - 1 and mode != "default":
                    content = self._format_prompt(content, mode)
                formatted_messages.append({"role": "user", "content": content})

            return chat_completion(
                api_base=self.api_base,
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                timeout=None,
            )

    original_get_model_runner = factory.get_model_runner

    def patched_get_model_runner(model_id: str, config: GenerationConfig | None = None):
        model_key = str(model_id or "").strip().lower()
        if model_key in PSYLLM_LMSTUDIO_ALIASES:
            return PsyLLMLMStudioRunner(config=config)
        return original_get_model_runner(model_id, config)

    factory.get_model_runner = patched_get_model_runner

    return original_get_model_runner


def _restore_factory_patch(original_get_model_runner: Callable | None) -> None:
    if original_get_model_runner is None:
        return
    import reliable_clinical_benchmark.models.factory as factory

    factory.get_model_runner = original_get_model_runner


def _run_job(
    runtime_root: Path,
    job: Job,
    *,
    selected_arms: Sequence[str],
) -> int:
    script_path = runtime_root / job.script_relative_path
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    module_name = f"_batch_{script_path.stem}_{int(time.time() * 1000000)}"
    module = _load_module(module_name, script_path)
    if job.supports_arm_filter:
        setattr(module, "ARMS", tuple(selected_arms))

    old_argv = sys.argv[:]
    sys.argv = [str(script_path), *job.argv]
    try:
        result = module.main()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = old_argv

    return int(result) if isinstance(result, int) else 0


def _format_command(job: Job, selected_arms: Sequence[str]) -> str:
    command = f"python {job.script_relative_path} " + " ".join(shlex.quote(token) for token in job.argv)
    if job.supports_arm_filter:
        return f"{command}    # arms={','.join(selected_arms)}"
    return command


def _build_controllability_jobs(args: argparse.Namespace) -> list[Job]:
    base_output = args.ctrl_output_dir
    return [
        Job(
            label="controllability: study_a",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_a",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                args.ctrl_dir,
                "--output-dir",
                base_output,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="controllability: study_a_bias",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_a_bias",
                "--model-id",
                args.model_id,
                "--data-path",
                args.ctrl_bias_data_path,
                "--output-dir",
                base_output,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="controllability: study_b",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_b",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                args.ctrl_dir,
                "--output-dir",
                base_output,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="controllability: study_b_multi_turn",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_b_multi_turn",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                args.ctrl_dir,
                "--output-dir",
                base_output,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="controllability: study_c",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_c",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                args.ctrl_dir,
                "--output-dir",
                base_output,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
    ]


def _build_invariance_base_jobs(args: argparse.Namespace) -> list[Job]:
    return [
        Job(
            label="invariance: study_a base",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_a_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                args.invariance_base_dir,
                "--output-dir",
                args.invariance_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance: study_a variants",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_a_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                f"{args.invariance_variants_root}/study_a",
                "--output-dir",
                args.invariance_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance: study_a_bias base",
            script_relative_path="hf-local-scripts/run_study_a_bias_generate_only.py",
            argv=(
                "--study-name",
                "study_a_bias_invariance",
                "--model-id",
                args.model_id,
                "--data-path",
                args.invariance_bias_data_path,
                "--output-dir",
                args.invariance_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance: study_b base",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_b_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                args.invariance_base_dir,
                "--output-dir",
                args.invariance_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance: study_b variants",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_b_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                f"{args.invariance_variants_root}/study_b",
                "--output-dir",
                args.invariance_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance: study_b_multi_turn base",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_b_multi_turn_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                args.invariance_base_dir,
                "--output-dir",
                args.invariance_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance: study_b_multi_turn variants",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_b_multi_turn_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                f"{args.invariance_variants_root}/study_b_multi_turn",
                "--output-dir",
                args.invariance_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance: study_c base",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_c_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                args.invariance_base_dir,
                "--output-dir",
                args.invariance_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance: study_c variants",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_c_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                f"{args.invariance_variants_root}/study_c",
                "--output-dir",
                args.invariance_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
    ]


def _build_invariance_ctrl_jobs(args: argparse.Namespace) -> list[Job]:
    return [
        Job(
            label="invariance_ctrl: study_a base",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_a_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                args.invariance_ctrl_base_dir,
                "--output-dir",
                args.invariance_ctrl_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance_ctrl: study_a variants",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_a_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                f"{args.invariance_ctrl_variants_root}/study_a",
                "--output-dir",
                args.invariance_ctrl_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance_ctrl: study_a_bias base",
            script_relative_path="hf-local-scripts/run_study_a_bias_generate_only.py",
            argv=(
                "--study-name",
                "study_a_bias_invariance",
                "--model-id",
                args.model_id,
                "--data-path",
                args.invariance_ctrl_bias_data_path,
                "--output-dir",
                args.invariance_ctrl_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance_ctrl: study_b base",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_b_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                args.invariance_ctrl_base_dir,
                "--output-dir",
                args.invariance_ctrl_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance_ctrl: study_b variants",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_b_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                f"{args.invariance_ctrl_variants_root}/study_b",
                "--output-dir",
                args.invariance_ctrl_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance_ctrl: study_b_multi_turn base",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_b_multi_turn_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                args.invariance_ctrl_base_dir,
                "--output-dir",
                args.invariance_ctrl_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance_ctrl: study_b_multi_turn variants",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_b_multi_turn_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                f"{args.invariance_ctrl_variants_root}/study_b_multi_turn",
                "--output-dir",
                args.invariance_ctrl_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance_ctrl: study_c base",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_c_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                args.invariance_ctrl_base_dir,
                "--output-dir",
                args.invariance_ctrl_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
        Job(
            label="invariance_ctrl: study_c variants",
            script_relative_path="hf-local-scripts/run_invariance_generate_only.py",
            argv=(
                "--study",
                "study_c_invariance",
                "--model-id",
                args.model_id,
                "--data-dir",
                f"{args.invariance_ctrl_variants_root}/study_c",
                "--output-dir",
                args.invariance_ctrl_output_dir,
                "--workers",
                str(args.workers),
            ),
        ),
    ]


def _build_reverse_jobs(args: argparse.Namespace) -> list[Job]:
    return [
        Job(
            label="reverse: study_a base",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_a",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                args.invariance_ctrl_base_dir,
                "--output-dir",
                args.reverse_output_dir,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="reverse: study_a variants",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_a",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                f"{args.invariance_ctrl_variants_root}/study_a",
                "--output-dir",
                args.reverse_output_dir,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="reverse: study_a_bias base",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_a_bias",
                "--model-id",
                args.model_id,
                "--data-path",
                args.invariance_ctrl_bias_data_path,
                "--output-dir",
                args.reverse_output_dir,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="reverse: study_b base",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_b",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                args.invariance_ctrl_base_dir,
                "--output-dir",
                args.reverse_output_dir,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="reverse: study_b variants",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_b",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                f"{args.invariance_ctrl_variants_root}/study_b",
                "--output-dir",
                args.reverse_output_dir,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="reverse: study_b_multi_turn base",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_b_multi_turn",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                args.invariance_ctrl_base_dir,
                "--output-dir",
                args.reverse_output_dir,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="reverse: study_b_multi_turn variants",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_b_multi_turn",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                f"{args.invariance_ctrl_variants_root}/study_b_multi_turn",
                "--output-dir",
                args.reverse_output_dir,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="reverse: study_c base",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_c",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                args.invariance_ctrl_base_dir,
                "--output-dir",
                args.reverse_output_dir,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
        Job(
            label="reverse: study_c variants",
            script_relative_path="hf-local-scripts/run_ctrl_generate_only.py",
            argv=(
                "--study",
                "ctrl_study_c",
                "--model-id",
                args.model_id,
                "--ctrl-dir",
                f"{args.invariance_ctrl_variants_root}/study_c",
                "--output-dir",
                args.reverse_output_dir,
                "--workers",
                str(args.workers),
            ),
            supports_arm_filter=True,
        ),
    ]


def _build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    if args.suite in {"controllability", "all"}:
        jobs.extend(_build_controllability_jobs(args))

    if args.suite in {"invariance", "all"}:
        if args.invariance_group in {"base", "all"}:
            jobs.extend(_build_invariance_base_jobs(args))
        if args.invariance_group in {"ctrl", "all"}:
            jobs.extend(_build_invariance_ctrl_jobs(args))
        if args.invariance_group in {"reverse", "all"}:
            jobs.extend(_build_reverse_jobs(args))

    return jobs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PsyLLM LM Studio batch jobs sequentially.")
    parser.add_argument(
        "--model-id",
        default="psyllm_gml_vllm",
        help=(
            "Model id passed into the study scripts. Defaults to psyllm_gml_vllm so existing "
            "server-side token and path normalisation logic still applies."
        ),
    )
    parser.add_argument(
        "--lmstudio-model-name",
        default=os.getenv("LMSTUDIO_PSYLLM_MODEL", "psyllm"),
        help="Loaded LM Studio model identifier. Defaults to LMSTUDIO_PSYLLM_MODEL or 'psyllm'.",
    )
    parser.add_argument(
        "--lmstudio-api-base",
        default=os.getenv("LMSTUDIO_API_BASE", "http://127.0.0.1:1234/v1"),
        help="LM Studio API base for the injected PsyLLM runner.",
    )
    parser.add_argument(
        "--suite",
        choices=("controllability", "invariance", "all"),
        default="all",
        help="Which batch family to run.",
    )
    parser.add_argument(
        "--invariance-group",
        choices=("base", "ctrl", "reverse", "all"),
        default="all",
        help="Which invariance subgroup to run when suite includes invariance.",
    )
    parser.add_argument(
        "--arms",
        default="all",
        help="Comma-separated controllability arms or 'all'. Applies to controllability and reverse jobs.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Workers passed to each job.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job sequence without executing it.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a failed job instead of stopping at the first failure.",
    )
    parser.add_argument(
        "--allow-lmstudio-autoload",
        action="store_true",
        help="Set LMSTUDIO_ALLOW_AUTOLOAD=1 for the current process before running jobs.",
    )
    parser.add_argument(
        "--ctrl-dir",
        default="data/controllability/controllability_splits_v2_1",
        help="Base controllability split directory.",
    )
    parser.add_argument(
        "--ctrl-bias-data-path",
        default="data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json",
        help="Study A bias controllability data path.",
    )
    parser.add_argument(
        "--ctrl-output-dir",
        default="results_ctrl_v2_1",
        help="Output directory for standard controllability runs.",
    )
    parser.add_argument(
        "--invariance-base-dir",
        default="data/invariance/v5/base/v2_1",
        help="Base invariance directory.",
    )
    parser.add_argument(
        "--invariance-variants-root",
        default="data/invariance/v5/variants/v2_1",
        help="Root directory containing v5 invariance variant-family folders.",
    )
    parser.add_argument(
        "--invariance-bias-data-path",
        default="data/invariance/v5/base/v2_1/adversarial_bias/biased_vignettes.json",
        help="Study A bias invariance base data path.",
    )
    parser.add_argument(
        "--invariance-output-dir",
        default="results_invariance",
        help="Output directory for base invariance runs.",
    )
    parser.add_argument(
        "--invariance-ctrl-base-dir",
        default="data/invariance/ctrl/base/v2_1",
        help="Base invariance_ctrl directory.",
    )
    parser.add_argument(
        "--invariance-ctrl-variants-root",
        default="data/invariance/ctrl/variants/v2_1",
        help="Root directory containing invariance_ctrl variant-family folders.",
    )
    parser.add_argument(
        "--invariance-ctrl-bias-data-path",
        default="data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json",
        help="Study A bias invariance_ctrl base data path.",
    )
    parser.add_argument(
        "--invariance-ctrl-output-dir",
        default="results_invariance_ctrl",
        help="Output directory for invariance_ctrl runs.",
    )
    parser.add_argument(
        "--reverse-output-dir",
        default="results_ctrl_invariance",
        help="Output directory for reverse invariance runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected_arms = _parse_arms(args.arms)
    runtime_root = _runtime_root()
    os.chdir(runtime_root)

    if args.allow_lmstudio_autoload:
        os.environ["LMSTUDIO_ALLOW_AUTOLOAD"] = "1"

    jobs = _build_jobs(args)
    if not jobs:
        raise SystemExit("No jobs were selected.")

    print(f"Runtime root: {runtime_root}")
    print(f"Model id: {args.model_id}")
    print(f"LM Studio model: {args.lmstudio_model_name}")
    print(f"Selected arms: {', '.join(selected_arms)}")
    print(f"Jobs selected: {len(jobs)}")

    if args.dry_run:
        print("")
        for index, job in enumerate(jobs, start=1):
            print(f"[{index:02d}/{len(jobs):02d}] {job.label}")
            print(f"  {_format_command(job, selected_arms)}")
        return 0

    _ensure_src_on_path(runtime_root)
    original_get_model_runner = _install_psyllm_lmstudio_patch(
        runtime_root,
        model_name=args.lmstudio_model_name,
        api_base=args.lmstudio_api_base,
    )

    failures: list[tuple[str, int]] = []
    started_at = time.time()
    try:
        for index, job in enumerate(jobs, start=1):
            print("")
            print(f"[{index:02d}/{len(jobs):02d}] START {job.label}")
            print(_format_command(job, selected_arms))
            exit_code = _run_job(
                runtime_root,
                job,
                selected_arms=selected_arms,
            )
            if exit_code == 0:
                print(f"[{index:02d}/{len(jobs):02d}] DONE  {job.label}")
                continue

            print(f"[{index:02d}/{len(jobs):02d}] FAIL  {job.label} (exit {exit_code})", file=sys.stderr)
            failures.append((job.label, exit_code))
            if not args.continue_on_error:
                break
    finally:
        _restore_factory_patch(original_get_model_runner)

    elapsed_seconds = int(time.time() - started_at)
    print("")
    print(f"Elapsed: {elapsed_seconds}s")
    if failures:
        print("Failed jobs:", file=sys.stderr)
        for label, exit_code in failures:
            print(f"  - {label} (exit {exit_code})", file=sys.stderr)
        return 1

    print("All selected jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
