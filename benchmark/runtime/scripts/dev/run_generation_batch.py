#!/usr/bin/env python3
"""Run documented generation batches sequentially."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
AUTO_RUNNER_PATH = RUNTIME_ROOT / "scripts" / "dev" / "run_generation_auto.py"

QWEN35_DISTILLED_ALIASES = {
    "mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2",
    "qwen3.5-distilled",
    "qwen3.5-27b-distilled",
    "qwen3_5_distilled_lmstudio",
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0",
}

DEFAULT_WORKERS_BY_MODEL = {
    "gpt_oss": 2,
    "gpt_oss_lmstudio": 2,
    "gpt-oss-20b": 2,
    "qwen3_lmstudio": 6,
    "qwq": 6,
    "piaget_lmstudio": 4,
    "psyllm_lmstudio": 4,
    "deepseek_r1_lmstudio": 4,
    "psych_qwen_32b": 4,
    "medgemma_lmstudio": 4,
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0": 8,
    "mlx-qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2": 8,
    "qwen3.5-distilled": 8,
    "qwen3.5-27b-distilled": 8,
    "qwen3_5_distilled_lmstudio": 8,
}


@dataclass(frozen=True)
class BatchStep:
    name: str
    study: str
    output_dir: str
    data_dir: str | None = None
    data_path: str | None = None
    ctrl_dir: str | None = None
    worker_overrides: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedCommand:
    name: str
    study: str
    command: tuple[str, ...]


CONTROLLABILITY_STEPS: tuple[BatchStep, ...] = (
    BatchStep(name="ctrl-study-a", study="ctrl_study_a", output_dir="results"),
    BatchStep(name="ctrl-study-a-bias", study="ctrl_study_a_bias", output_dir="results"),
    BatchStep(name="ctrl-study-b", study="ctrl_study_b", output_dir="results"),
    BatchStep(name="ctrl-study-b-multi-turn", study="ctrl_study_b_multi_turn", output_dir="results"),
    BatchStep(name="ctrl-study-c", study="ctrl_study_c", output_dir="results"),
)

INVARIANCE_STEPS: tuple[BatchStep, ...] = (
    BatchStep(
        name="study-a-bias-invariance",
        study="study_a_bias_invariance",
        output_dir="results_invariance",
        data_path="data/invariance/v5/base/v2_1/adversarial_bias/biased_vignettes.json",
    ),
    BatchStep(
        name="study-a-bias-invariance-ctrl",
        study="study_a_bias_invariance",
        output_dir="results_invariance_ctrl",
        data_path="data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json",
    ),
    BatchStep(
        name="study-a-bias-reverse-invariance",
        study="ctrl_study_a_bias",
        output_dir="results_ctrl_invariance",
        ctrl_dir="data/invariance/ctrl/base/v2_1",
    ),
    BatchStep(
        name="study-a-invariance",
        study="study_a_invariance",
        output_dir="results_invariance",
        data_dir="data/invariance/v5/base/v2_1",
    ),
    BatchStep(
        name="study-a-reverse-invariance",
        study="ctrl_study_a",
        output_dir="results_ctrl_invariance",
        ctrl_dir="data/invariance/ctrl/base/v2_1",
    ),
    BatchStep(
        name="study-a-invariance-ctrl",
        study="study_a_invariance",
        output_dir="results_invariance_ctrl",
        data_dir="data/invariance/ctrl/base/v2_1",
    ),
    BatchStep(
        name="study-b-invariance",
        study="study_b_invariance",
        output_dir="results_invariance",
        data_dir="data/invariance/v5/base/v2_1",
    ),
    BatchStep(
        name="study-b-invariance-ctrl",
        study="study_b_invariance",
        output_dir="results_invariance_ctrl",
        data_dir="data/invariance/ctrl/base/v2_1",
    ),
    BatchStep(
        name="study-b-multi-turn-invariance",
        study="study_b_multi_turn_invariance",
        output_dir="results_invariance",
        data_dir="data/invariance/v5/base/v2_1",
    ),
    BatchStep(
        name="study-b-multi-turn-invariance-ctrl",
        study="study_b_multi_turn_invariance",
        output_dir="results_invariance_ctrl",
        data_dir="data/invariance/ctrl/base/v2_1",
    ),
    BatchStep(
        name="study-b-multi-turn-reverse-invariance",
        study="ctrl_study_b_multi_turn",
        output_dir="results_ctrl_invariance",
        ctrl_dir="data/invariance/ctrl/base/v2_1",
    ),
    BatchStep(
        name="study-b-reverse-invariance",
        study="ctrl_study_b",
        output_dir="results_ctrl_invariance",
        ctrl_dir="data/invariance/ctrl/base/v2_1",
    ),
    BatchStep(
        name="study-c-invariance",
        study="study_c_invariance",
        output_dir="results_invariance",
        data_dir="data/invariance/v5/base/v2_1",
        worker_overrides={"gpt_oss": 2, **{alias: 1 for alias in QWEN35_DISTILLED_ALIASES}},
    ),
    BatchStep(
        name="study-c-invariance-ctrl",
        study="study_c_invariance",
        output_dir="results_invariance_ctrl",
        data_dir="data/invariance/ctrl/base/v2_1",
        worker_overrides={"gpt_oss": 2, **{alias: 1 for alias in QWEN35_DISTILLED_ALIASES}},
    ),
    BatchStep(
        name="study-c-reverse-invariance",
        study="ctrl_study_c",
        output_dir="results_ctrl_invariance",
        ctrl_dir="data/invariance/ctrl/base/v2_1",
        worker_overrides={"gpt_oss": 1},
    ),
)

PRESET_STEPS = {
    "controllability": CONTROLLABILITY_STEPS,
    "invariance": INVARIANCE_STEPS,
    "invariance-base": (
        INVARIANCE_STEPS[0],
        INVARIANCE_STEPS[3],
        INVARIANCE_STEPS[6],
        INVARIANCE_STEPS[8],
        INVARIANCE_STEPS[12],
    ),
    "invariance-ctrl": (
        INVARIANCE_STEPS[1],
        INVARIANCE_STEPS[5],
        INVARIANCE_STEPS[7],
        INVARIANCE_STEPS[9],
        INVARIANCE_STEPS[13],
    ),
    "reverse-invariance": (
        INVARIANCE_STEPS[2],
        INVARIANCE_STEPS[4],
        INVARIANCE_STEPS[10],
        INVARIANCE_STEPS[11],
        INVARIANCE_STEPS[14],
    ),
    "all": CONTROLLABILITY_STEPS + INVARIANCE_STEPS,
}


def _normalise_model_id(model_id: str) -> str:
    return (model_id or "").strip().lower()


def infer_default_env(model_id: str) -> str:
    model_key = _normalise_model_id(model_id)
    if model_key.endswith("_vllm"):
        return "mh-llm-vllm-env"
    if model_key.endswith("_local"):
        return "mh-llm-local-env"
    return "mh-llm-benchmark-env"


def _resolve_workers(step: BatchStep, model_id: str, requested_workers: int | None) -> int | None:
    if requested_workers is not None:
        return requested_workers

    model_key = _normalise_model_id(model_id)
    if model_key in step.worker_overrides:
        return step.worker_overrides[model_key]
    return DEFAULT_WORKERS_BY_MODEL.get(model_key)


def _build_command(
    step: BatchStep,
    *,
    model_id: str,
    env: str | None,
    python_executable: str,
    requested_workers: int | None,
    check_only: bool,
    allow_lmstudio_autoload: bool,
    passthrough: Sequence[str],
) -> tuple[str, ...]:
    command: list[str] = [python_executable, str(AUTO_RUNNER_PATH), "--study", step.study, "--model-id", model_id]

    effective_env = env or infer_default_env(model_id)
    if effective_env:
        command.extend(["--env", effective_env])
    if step.data_dir:
        command.extend(["--data-dir", step.data_dir])
    if step.data_path:
        command.extend(["--data-path", step.data_path])
    if step.ctrl_dir:
        command.extend(["--ctrl-dir", step.ctrl_dir])
    command.extend(["--output-dir", step.output_dir])

    workers = _resolve_workers(step, model_id, requested_workers)
    if workers is not None:
        command.extend(["--workers", str(workers)])
    if check_only:
        command.append("--check-only")
    if allow_lmstudio_autoload:
        command.append("--allow-lmstudio-autoload")
    command.extend(passthrough)
    return tuple(command)


def build_batch_plan(
    *,
    preset: str,
    model_id: str,
    env: str | None,
    python_executable: str,
    passthrough: Sequence[str],
    workers: int | None = None,
    check_only: bool = False,
    allow_lmstudio_autoload: bool = False,
) -> list[PlannedCommand]:
    steps = PRESET_STEPS[preset]
    return [
        PlannedCommand(
            name=step.name,
            study=step.study,
            command=_build_command(
                step,
                model_id=model_id,
                env=env,
                python_executable=python_executable,
                requested_workers=workers,
                check_only=check_only,
                allow_lmstudio_autoload=allow_lmstudio_autoload,
                passthrough=passthrough,
            ),
        )
        for step in steps
    ]


def run_batch(plan: Sequence[PlannedCommand], *, continue_on_error: bool) -> int:
    final_exit_code = 0
    total_steps = len(plan)
    for index, step in enumerate(plan, start=1):
        print(f"[{index}/{total_steps}] {step.name}: {step.study}")
        print(f"  {shlex.join(step.command)}")
        completed = subprocess.run(step.command, cwd=RUNTIME_ROOT, check=False)
        if completed.returncode != 0:
            final_exit_code = completed.returncode
            print(f"Step failed with exit code {completed.returncode}: {step.name}", file=sys.stderr)
            if not continue_on_error:
                return completed.returncode
        print("")
    return final_exit_code


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run documented study commands one after another so each study finishes "
            "and saves normally before the next starts."
        )
    )
    parser.add_argument("--preset", choices=sorted(PRESET_STEPS), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--env", default=None, help="Optional shared conda env override for every step.")
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable used to launch the shared auto runner.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Optional shared worker override for every step.")
    parser.add_argument("--check-only", action="store_true", help="Pass --check-only through to each child run.")
    parser.add_argument(
        "--allow-lmstudio-autoload",
        action="store_true",
        help="Pass --allow-lmstudio-autoload through to each child run.",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Keep going if one step fails.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them.")
    parser.add_argument("--list-steps", action="store_true", help="Print just the step names for the chosen preset.")
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = _parse_args()
    plan = build_batch_plan(
        preset=args.preset,
        model_id=args.model_id,
        env=args.env,
        python_executable=args.python_executable,
        passthrough=passthrough,
        workers=args.workers,
        check_only=args.check_only,
        allow_lmstudio_autoload=args.allow_lmstudio_autoload,
    )

    if args.list_steps:
        for step in plan:
            print(step.name)
        return 0

    if args.dry_run:
        for index, step in enumerate(plan, start=1):
            print(f"[{index}/{len(plan)}] {step.name}: {shlex.join(step.command)}")
        return 0

    return run_batch(plan, continue_on_error=args.continue_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
