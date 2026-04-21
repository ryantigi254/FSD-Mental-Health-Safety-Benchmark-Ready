"""Unit tests for the sequential generation batch wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
BATCH_RUNNER_PATH = RUNTIME_ROOT / "scripts" / "dev" / "run_generation_batch.py"


def _load_batch_module():
    spec = importlib.util.spec_from_file_location("run_generation_batch", BATCH_RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_build_batch_plan_for_controllability_uses_expected_order():
    batch_module = _load_batch_module()

    plan = batch_module.build_batch_plan(
        preset="controllability",
        model_id="qwq",
        env="mh-llm-benchmark-env",
        python_executable="python",
        passthrough=[],
    )

    assert [step.name for step in plan] == [
        "ctrl-study-a",
        "ctrl-study-a-bias",
        "ctrl-study-b",
        "ctrl-study-b-multi-turn",
        "ctrl-study-c",
    ]
    assert [step.study for step in plan] == [
        "ctrl_study_a",
        "ctrl_study_a_bias",
        "ctrl_study_b",
        "ctrl_study_b_multi_turn",
        "ctrl_study_c",
    ]
    for step in plan:
        assert step.command[0] == "python"
        assert str(batch_module.AUTO_RUNNER_PATH) in step.command
        assert step.command[step.command.index("--model-id") + 1] == "qwq"
        assert step.command[step.command.index("--output-dir") + 1] == "results"


@pytest.mark.unit
def test_build_batch_plan_for_invariance_keeps_documented_sequence_and_roots():
    batch_module = _load_batch_module()

    plan = batch_module.build_batch_plan(
        preset="invariance",
        model_id="gpt_oss",
        env="mh-llm-benchmark-env",
        python_executable="python",
        passthrough=[],
    )

    assert [step.name for step in plan] == [
        "study-a-bias-invariance",
        "study-a-bias-invariance-ctrl",
        "study-a-bias-reverse-invariance",
        "study-a-invariance",
        "study-a-reverse-invariance",
        "study-a-invariance-ctrl",
        "study-b-invariance",
        "study-b-invariance-ctrl",
        "study-b-multi-turn-invariance",
        "study-b-multi-turn-invariance-ctrl",
        "study-b-multi-turn-reverse-invariance",
        "study-b-reverse-invariance",
        "study-c-invariance",
        "study-c-invariance-ctrl",
        "study-c-reverse-invariance",
    ]

    expected_flags = {
        "study-a-bias-invariance": ("--data-path", "data/invariance/v5/base/v2_1/adversarial_bias/biased_vignettes.json", "results_invariance"),
        "study-a-bias-invariance-ctrl": ("--data-path", "data/invariance/ctrl/base/v2_1/adversarial_bias/biased_vignettes.json", "results_invariance_ctrl"),
        "study-a-bias-reverse-invariance": ("--ctrl-dir", "data/invariance/ctrl/base/v2_1", "results_ctrl_invariance"),
        "study-a-invariance": ("--data-dir", "data/invariance/v5/base/v2_1", "results_invariance"),
        "study-a-reverse-invariance": ("--ctrl-dir", "data/invariance/ctrl/base/v2_1", "results_ctrl_invariance"),
        "study-a-invariance-ctrl": ("--data-dir", "data/invariance/ctrl/base/v2_1", "results_invariance_ctrl"),
        "study-b-invariance": ("--data-dir", "data/invariance/v5/base/v2_1", "results_invariance"),
        "study-b-invariance-ctrl": ("--data-dir", "data/invariance/ctrl/base/v2_1", "results_invariance_ctrl"),
        "study-b-multi-turn-invariance": ("--data-dir", "data/invariance/v5/base/v2_1", "results_invariance"),
        "study-b-multi-turn-invariance-ctrl": ("--data-dir", "data/invariance/ctrl/base/v2_1", "results_invariance_ctrl"),
        "study-b-multi-turn-reverse-invariance": ("--ctrl-dir", "data/invariance/ctrl/base/v2_1", "results_ctrl_invariance"),
        "study-b-reverse-invariance": ("--ctrl-dir", "data/invariance/ctrl/base/v2_1", "results_ctrl_invariance"),
        "study-c-invariance": ("--data-dir", "data/invariance/v5/base/v2_1", "results_invariance"),
        "study-c-invariance-ctrl": ("--data-dir", "data/invariance/ctrl/base/v2_1", "results_invariance_ctrl"),
        "study-c-reverse-invariance": ("--ctrl-dir", "data/invariance/ctrl/base/v2_1", "results_ctrl_invariance"),
    }

    for step in plan:
        flag_name, flag_value, output_dir = expected_flags[step.name]
        assert step.command[step.command.index(flag_name) + 1] == flag_value
        assert step.command[step.command.index("--output-dir") + 1] == output_dir


@pytest.mark.unit
def test_run_batch_stops_after_first_failure(monkeypatch):
    batch_module = _load_batch_module()

    plan = [
        batch_module.PlannedCommand(
            name="step-one",
            study="ctrl_study_a",
            command=("python", "runner.py", "--study", "ctrl_study_a"),
        ),
        batch_module.PlannedCommand(
            name="step-two",
            study="ctrl_study_b",
            command=("python", "runner.py", "--study", "ctrl_study_b"),
        ),
    ]
    calls: list[tuple[str, ...]] = []

    def _fake_run(command, cwd=None, check=None):  # noqa: ANN001
        calls.append(tuple(command))
        return type("Completed", (), {"returncode": 9})()

    monkeypatch.setattr(batch_module.subprocess, "run", _fake_run)

    exit_code = batch_module.run_batch(plan, continue_on_error=False)

    assert exit_code == 9
    assert calls == [("python", "runner.py", "--study", "ctrl_study_a")]
