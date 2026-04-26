"""Unit tests for controllability generate-only worker execution."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Optional

import pytest

from reliable_clinical_benchmark.models.base import GenerationConfig, ModelRunner


def _load_ctrl_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "hf-local-scripts"
        / "run_ctrl_generate_only.py"
    )
    spec = importlib.util.spec_from_file_location("run_ctrl_generate_only", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DummyLMStudioRunner(ModelRunner):
    def __init__(self, model_name: str = "dummy-lm", config: Optional[GenerationConfig] = None):
        super().__init__(model_name=model_name, config=config or GenerationConfig())
        self.api_base = "http://localhost:1234/v1"

    def generate(self, prompt: str, mode: str = "default") -> str:
        constraint = getattr(self, "cot_controlled_constraint", "")
        return f"{mode.upper()}::{constraint[:8]}::{prompt[:12]}"

    def generate_with_reasoning(self, prompt: str):
        return self.generate(prompt), "reasoning"


@pytest.mark.unit
def test_ctrl_study_a_generation_writes_cache_with_workers(tmp_path: Path) -> None:
    ctrl_module = _load_ctrl_module()
    cache_path = tmp_path / "ctrl_study_a_generations.jsonl"
    runner = _DummyLMStudioRunner(config=GenerationConfig(max_tokens=64))

    items = [
        {
            "id": "ctrl_a_001",
            "prompt": "Case one",
            "cot_controlled_constraint": "Mention symptom chronology.",
            "metadata": {"persona_id": "aisha"},
        },
        {
            "id": "ctrl_a_002",
            "prompt": "Case two",
            "cot_controlled_constraint": "Mention risk factors.",
            "metadata": {"persona_id": "jamal"},
        },
    ]

    ctrl_module.generate_study_a(
        runner=runner,
        items=items,
        cache_path=cache_path,
        run_id="test-run",
        model_id="qwq",
        existing=set(),
        worker_count=4,
        progress_interval_seconds=1,
        runner_factory=lambda: _DummyLMStudioRunner(config=GenerationConfig(max_tokens=64)),
    )

    assert cache_path.exists()
    rows = [
        json.loads(line)
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 6

    modes = sorted(row["mode"] for row in rows)
    assert modes.count("cot_controlled") == 4
    assert modes.count("cot") == 2

    for row in rows:
        assert row["status"] in ("ok", "error")
        assert row["id"] in {"ctrl_a_001", "ctrl_a_002"}
        assert isinstance(row["output_text"], str)
        assert row["sampling"]["max_tokens"] == 64


@pytest.mark.unit
def test_resolve_output_dir_defaults_to_results_root() -> None:
    ctrl_module = _load_ctrl_module()

    resolved = ctrl_module._resolve_output_dir(None)

    assert resolved == ctrl_module.RUNTIME_ROOT / "results"


@pytest.mark.unit
def test_resolve_output_dir_preserves_explicit_reverse_invariance_root() -> None:
    ctrl_module = _load_ctrl_module()

    resolved = ctrl_module._resolve_output_dir("results_ctrl_invariance")

    assert resolved == ctrl_module.RUNTIME_ROOT / "results_ctrl_invariance"
