"""Unit tests for Study C generation-only JSONL cache."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

from reliable_clinical_benchmark.models.base import ModelRunner, GenerationConfig
from reliable_clinical_benchmark.pipelines.study_c import run_study_c


class _DummyRunner(ModelRunner):
    def __init__(self, model_name: str = "dummy", config: Optional[GenerationConfig] = None):
        super().__init__(model_name=model_name, config=config or GenerationConfig())

    def generate(self, prompt: str, mode: str = "default") -> str:
        return f"{mode.upper()}::{prompt[:32]}"

    def generate_with_reasoning(self, prompt: str):
        return self.generate(prompt), "reasoning"


class _BlockingDialogueRunner(ModelRunner):
    def __init__(
        self,
        dialogue_block: threading.Event,
        dialogue_release: threading.Event,
        model_name: str = "dummy",
        config: Optional[GenerationConfig] = None,
    ):
        super().__init__(model_name=model_name, config=config or GenerationConfig())
        self.dialogue_block = dialogue_block
        self.dialogue_release = dialogue_release

    def generate(self, prompt: str, mode: str = "default") -> str:
        return f"{mode.upper()}::{prompt[:32]}"

    def chat(self, messages, mode: str = "default") -> str:
        self.dialogue_block.set()
        released = self.dialogue_release.wait(timeout=5)
        if not released:
            raise TimeoutError("dialogue release was not signalled")
        return "DIALOGUE::ok"

    def generate_with_reasoning(self, prompt: str):
        return self.generate(prompt), "reasoning"


@pytest.mark.unit
def test_study_c_generate_only_writes_jsonl(tmp_path: Path) -> None:
    data_dir = tmp_path / "openr1_psy_splits"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cases": [
            {
                "id": "c_001",
                "patient_summary": "Patient summary",
                "critical_entities": ["sertraline 50mg", "major depressive disorder"],
                "turns": [
                    {"turn": 1, "message": "Turn one"},
                    {"turn": 2, "message": "Turn two"},
                ],
                "metadata": {"persona_id": "aisha", "source_openr1_ids": [16]},
            }
        ]
    }
    (data_dir / "study_c_test.json").write_text(json.dumps(payload), encoding="utf-8")

    cache_path = tmp_path / "study_c_generations.jsonl"
    model = _DummyRunner(config=GenerationConfig(max_tokens=64))

    run_study_c(
        model=model,
        data_dir=str(data_dir),
        max_cases=1,
        output_dir=str(tmp_path),
        model_name="dummy",
        use_nli=False,
        generate_only=True,
        cache_out=str(cache_path),
    )

    assert cache_path.exists()
    lines = [ln for ln in cache_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # 2 turns * (summary + dialogue)
    assert len(lines) == 4

    rows = [json.loads(ln) for ln in lines]
    variants = {r.get("variant") for r in rows}
    assert variants == {"summary", "dialogue"}

    for r in rows:
        assert r.get("case_id") == "c_001"
        assert r.get("persona_id") == "aisha"
        assert isinstance(r.get("turn_num"), int)
        assert r.get("status") in ("ok", "error")
        assert isinstance(r.get("timestamp"), str) and r["timestamp"]
        assert isinstance(r.get("run_id"), str) and r["run_id"]
        assert isinstance(r.get("model_name"), str) and r["model_name"]
        assert isinstance(r.get("response_text"), str)

        if r["variant"] == "summary":
            assert isinstance(r.get("prompt"), str) and r["prompt"].startswith("Summarise the current patient state")
        if r["variant"] == "dialogue":
            assert isinstance(r.get("conversation_text"), str) and r["conversation_text"]


@pytest.mark.unit
def test_study_c_generate_only_persists_rows_before_case_completes(tmp_path: Path) -> None:
    data_dir = tmp_path / "openr1_psy_splits"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cases": [
            {
                "id": "c_001",
                "patient_summary": "Patient summary",
                "critical_entities": ["sertraline 50mg", "major depressive disorder"],
                "turns": [
                    {"turn": 1, "message": "Turn one"},
                    {"turn": 2, "message": "Turn two"},
                ],
                "metadata": {"persona_id": "aisha", "source_openr1_ids": [16]},
            }
        ]
    }
    (data_dir / "study_c_test.json").write_text(json.dumps(payload), encoding="utf-8")

    cache_path = tmp_path / "study_c_generations.jsonl"
    dialogue_block = threading.Event()
    dialogue_release = threading.Event()
    model = _BlockingDialogueRunner(
        dialogue_block=dialogue_block,
        dialogue_release=dialogue_release,
        config=GenerationConfig(max_tokens=64),
    )

    worker = threading.Thread(
        target=run_study_c,
        kwargs={
            "model": model,
            "data_dir": str(data_dir),
            "max_cases": 1,
            "output_dir": str(tmp_path),
            "model_name": "dummy",
            "use_nli": False,
            "generate_only": True,
            "cache_out": str(cache_path),
        },
        daemon=True,
    )
    worker.start()

    assert dialogue_block.wait(timeout=5), "study_c dialogue step did not start"

    deadline = time.time() + 5
    lines: list[str] = []
    while time.time() < deadline:
        if cache_path.exists():
            lines = [ln for ln in cache_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                break
        time.sleep(0.05)

    assert lines, "expected at least one persisted row before the case completed"
    first_row = json.loads(lines[0])
    assert first_row["variant"] == "summary"
    assert first_row["turn_num"] == 1

    dialogue_release.set()
    worker.join(timeout=5)
    assert not worker.is_alive(), "study_c worker thread did not finish"


