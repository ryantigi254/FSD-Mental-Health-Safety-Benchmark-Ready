#!/usr/bin/env python3
"""Run invariance generation suites sequentially from one command.

This is a thin serial orchestrator around ``scripts/dev/run_generation_auto.py``.
It preserves the normal behaviour of each individual generation command:

- each study still writes to its usual cache path
- stdout/stderr stream through as normal
- the next study starts only after the prior one exits

Examples
--------

Run every invariance-related generation suite for Qwen3 with the documented
Windows worker count:

    python scripts/dev/run_invariance_batch.py --model-id qwen3_lmstudio

Run only forward invariance base + variant-family passes:

    python scripts/dev/run_invariance_batch.py \
      --model-id qwen3_lmstudio \
      --suite invariance \
      --root-kind both

Preview the exact commands without executing them:

    python scripts/dev/run_invariance_batch.py \
      --model-id qwen3_lmstudio \
      --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
AUTO_SCRIPT_PATH = RUNTIME_ROOT / "scripts" / "dev" / "run_generation_auto.py"

SUITE_ALL = "all"
SUITE_INVARIANCE = "invariance"
SUITE_CTRL = "ctrl"
SUITE_REVERSE = "reverse"

ROOT_BASE = "base"
ROOT_VARIANTS = "variants"
ROOT_BOTH = "both"


@dataclass(frozen=True)
class StudySpec:
    study: str
    arg_name: str
    variant_subdir: Optional[str]
    bias_relative_path: Optional[str] = None


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    output_dir: str
    base_root: str
    variant_root: str
    studies: Sequence[StudySpec]


@dataclass(frozen=True)
class PlannedCommand:
    suite: str
    root_kind: str
    study: str
    command: List[str]


INVARIANCE_STUDIES: tuple[StudySpec, ...] = (
    StudySpec("study_a_invariance", "data-dir", "study_a"),
    StudySpec(
        "study_a_bias_invariance",
        "data-path",
        None,
        bias_relative_path="adversarial_bias/biased_vignettes.json",
    ),
    StudySpec("study_b_invariance", "data-dir", "study_b"),
    StudySpec("study_b_multi_turn_invariance", "data-dir", "study_b_multi_turn"),
    StudySpec("study_c_invariance", "data-dir", "study_c"),
)

REVERSE_STUDIES: tuple[StudySpec, ...] = (
    StudySpec("ctrl_study_a", "ctrl-dir", "study_a"),
    StudySpec("ctrl_study_a_bias", "ctrl-dir", None),
    StudySpec("ctrl_study_b", "ctrl-dir", "study_b"),
    StudySpec("ctrl_study_b_multi_turn", "ctrl-dir", "study_b_multi_turn"),
    StudySpec("ctrl_study_c", "ctrl-dir", "study_c"),
)

SUITE_SPECS = {
    SUITE_INVARIANCE: SuiteSpec(
        name=SUITE_INVARIANCE,
        output_dir="results_invariance",
        base_root="data/invariance/v5/base/v2_1",
        variant_root="data/invariance/v5/variants/v2_1",
        studies=INVARIANCE_STUDIES,
    ),
    SUITE_CTRL: SuiteSpec(
        name=SUITE_CTRL,
        output_dir="results_invariance_ctrl",
        base_root="data/invariance/ctrl/base/v2_1",
        variant_root="data/invariance/ctrl/variants/v2_1",
        studies=INVARIANCE_STUDIES,
    ),
    SUITE_REVERSE: SuiteSpec(
        name=SUITE_REVERSE,
        output_dir="results_ctrl_invariance",
        base_root="data/invariance/ctrl/base/v2_1",
        variant_root="data/invariance/ctrl/variants/v2_1",
        studies=REVERSE_STUDIES,
    ),
}

DEFAULT_WORKERS = {
    "gpt_oss": 2,
    "gpt_oss_lmstudio": 2,
    "qwen3_lmstudio": 6,
    "qwq": 6,
    "deepseek_r1_lmstudio": 4,
    "piaget_lmstudio": 4,
    "medgemma_lmstudio": 4,
    "psych_qwen_32b": 4,
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled@q8_0": 8,
    "gpt-oss-120b-runpod": 4,
    "gpt_oss_120b_runpod": 4,
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Model ID passed through to run_generation_auto.py.")
    parser.add_argument(
        "--suite",
        choices=(SUITE_ALL, SUITE_INVARIANCE, SUITE_CTRL, SUITE_REVERSE),
        default=SUITE_ALL,
        help="Which invariance suite to run. Default: all.",
    )
    parser.add_argument(
        "--root-kind",
        choices=(ROOT_BASE, ROOT_VARIANTS, ROOT_BOTH),
        default=ROOT_BOTH,
        help="Run base roots, variant-family roots, or both. Default: both.",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Optional conda env name forwarded to run_generation_auto.py.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override workers for every command. If omitted, uses a model-specific doc default when known.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional max-cases forwarded to every command.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the generated commands without starting any generation jobs.",
    )
    parser.add_argument(
        "--allow-lmstudio-autoload",
        action="store_true",
        help="Forward LM Studio autoload bypass to run_generation_auto.py when needed.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a failed command instead of stopping immediately.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated commands and exit without running them.",
    )
    return parser.parse_args(argv)


def _selected_suites(suite: str) -> Sequence[SuiteSpec]:
    if suite == SUITE_ALL:
        return (
            SUITE_SPECS[SUITE_INVARIANCE],
            SUITE_SPECS[SUITE_CTRL],
            SUITE_SPECS[SUITE_REVERSE],
        )
    return (SUITE_SPECS[suite],)


def _selected_roots(root_kind: str) -> Sequence[str]:
    if root_kind == ROOT_BOTH:
        return (ROOT_BASE, ROOT_VARIANTS)
    return (root_kind,)


def _default_workers_for_model(model_id: str) -> Optional[int]:
    return DEFAULT_WORKERS.get(model_id)


def _data_value_for(study: StudySpec, suite: SuiteSpec, root_kind: str) -> Optional[str]:
    if root_kind == ROOT_BASE:
        root = Path(suite.base_root)
        if study.arg_name == "data-path":
            assert study.bias_relative_path is not None
            return str((root / study.bias_relative_path).as_posix())
        return str(root.as_posix())

    if study.variant_subdir is None:
        return None
    root = Path(suite.variant_root) / study.variant_subdir
    if study.arg_name == "data-path":
        assert study.bias_relative_path is not None
        return str((root / study.bias_relative_path).as_posix())
    return str(root.as_posix())


def build_execution_plan(args: argparse.Namespace) -> List[PlannedCommand]:
    plan: List[PlannedCommand] = []
    worker_value = args.workers if args.workers is not None else _default_workers_for_model(args.model_id)

    for suite in _selected_suites(args.suite):
        for root_kind in _selected_roots(args.root_kind):
            for study in suite.studies:
                data_value = _data_value_for(study, suite, root_kind)
                if data_value is None:
                    continue

                command = [sys.executable, str(AUTO_SCRIPT_PATH), "--study", study.study, "--model-id", args.model_id]
                if args.env:
                    command.extend(["--env", args.env])
                command.extend([f"--{study.arg_name}", data_value, "--output-dir", suite.output_dir])
                if worker_value is not None:
                    command.extend(["--workers", str(worker_value)])
                if args.max_cases is not None:
                    command.extend(["--max-cases", str(args.max_cases)])
                if args.check_only:
                    command.append("--check-only")
                if args.allow_lmstudio_autoload:
                    command.append("--allow-lmstudio-autoload")

                plan.append(
                    PlannedCommand(
                        suite=suite.name,
                        root_kind=root_kind,
                        study=study.study,
                        command=command,
                    )
                )

    return plan


def _print_plan(plan: Iterable[PlannedCommand]) -> None:
    plan = list(plan)
    print(f"Runtime root: {RUNTIME_ROOT}")
    print(f"Planned commands: {len(plan)}")
    for index, entry in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] suite={entry.suite} root={entry.root_kind} study={entry.study}")
        print(f"  {shlex.join(entry.command)}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    plan = build_execution_plan(args)
    if not plan:
        print("No commands generated for the selected suite/root combination.", file=sys.stderr)
        return 2

    _print_plan(plan)
    if args.dry_run:
        print("Dry-run mode: no commands executed.")
        return 0

    failures = 0
    for index, entry in enumerate(plan, start=1):
        print(f"\n=== [{index}/{len(plan)}] Starting {entry.study} ({entry.suite}, {entry.root_kind}) ===")
        completed = subprocess.run(entry.command, cwd=str(RUNTIME_ROOT))
        if completed.returncode == 0:
            continue

        failures += 1
        print(
            f"Command failed with exit code {completed.returncode}: {entry.study} "
            f"({entry.suite}, {entry.root_kind})",
            file=sys.stderr,
        )
        if not args.continue_on_error:
            return completed.returncode

    if failures:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 1

    print("\nAll planned invariance commands completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
