from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
BATCH_SCRIPT_PATH = RUNTIME_ROOT / "scripts" / "dev" / "run_invariance_batch.py"


def _load_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, BATCH_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_build_execution_plan_for_qwen3_all_suites_both_roots():
    module = _load_module("run_invariance_batch_test_module_all")
    args = module.parse_args(["--model-id", "qwen3_lmstudio"])

    plan = module.build_execution_plan(args)

    assert len(plan) == 27

    studies = [entry.study for entry in plan]
    assert studies.count("study_a_invariance") == 4
    assert studies.count("study_b_invariance") == 4
    assert studies.count("study_b_multi_turn_invariance") == 4
    assert studies.count("study_c_invariance") == 4
    assert studies.count("study_a_bias_invariance") == 2
    assert studies.count("ctrl_study_a") == 2
    assert studies.count("ctrl_study_b") == 2
    assert studies.count("ctrl_study_b_multi_turn") == 2
    assert studies.count("ctrl_study_c") == 2
    assert studies.count("ctrl_study_a_bias") == 1

    variant_entries = [entry for entry in plan if entry.root_kind == module.ROOT_VARIANTS]
    assert all("study_a_bias_invariance" != entry.study for entry in variant_entries)
    assert all("ctrl_study_a_bias" != entry.study for entry in variant_entries)

    qwen3_commands = [" ".join(entry.command) for entry in plan]
    assert all("--workers 6" in command for command in qwen3_commands)


def test_build_execution_plan_for_reverse_base_override_workers():
    module = _load_module("run_invariance_batch_test_module_reverse")
    args = module.parse_args(
        [
            "--model-id",
            "gpt_oss",
            "--suite",
            "reverse",
            "--root-kind",
            "base",
            "--workers",
            "3",
            "--env",
            "mh-llm-benchmark-env",
        ]
    )

    plan = module.build_execution_plan(args)

    assert len(plan) == 5
    assert {entry.suite for entry in plan} == {module.SUITE_REVERSE}
    assert {entry.root_kind for entry in plan} == {module.ROOT_BASE}
    assert all("--ctrl-dir" in entry.command for entry in plan)
    assert all("results_ctrl_invariance" in entry.command for entry in plan)
    assert all("--workers" in entry.command and "3" in entry.command for entry in plan)
    assert all("--env" in entry.command and "mh-llm-benchmark-env" in entry.command for entry in plan)


def test_main_dry_run_prints_planned_commands(monkeypatch, capsys):
    module = _load_module("run_invariance_batch_test_module_dry_run")

    exit_code = module.main(
        [
            "--model-id",
            "qwen3_lmstudio",
            "--suite",
            "invariance",
            "--root-kind",
            "base",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Planned commands: 5" in captured.out
    assert "study_a_invariance" in captured.out
    assert "study_c_invariance" in captured.out
    assert "Dry-run mode: no commands executed." in captured.out
