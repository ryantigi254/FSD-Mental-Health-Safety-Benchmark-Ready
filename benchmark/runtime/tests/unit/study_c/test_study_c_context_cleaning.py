"""Unit tests for Study C context cleaning behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from reliable_clinical_benchmark.models.base import ModelRunner, GenerationConfig
from reliable_clinical_benchmark.pipelines.study_c import run_study_c


class _RepeatingRunner(ModelRunner):
    def __init__(self, model_name: str = "dummy", config: Optional[GenerationConfig] = None):
        super().__init__(model_name=model_name, config=config or GenerationConfig())
        line = "REPEAT ME " * 10  # ~100 chars
        self.response = "\n".join([line] * 4)

    def generate(self, prompt: str, mode: str = "default") -> str:
        return f"{mode.upper()}::{self.response}"

    def generate_with_reasoning(self, prompt: str):
        return self.generate(prompt), "reasoning"


@pytest.mark.unit
def test_context_cleaning_starts_from_turn_4(tmp_path: Path) -> None:
    data_dir = tmp_path / "openr1_psy_splits"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cases": [
            {
                "id": "c_001",
                "patient_summary": "Patient summary",
                "critical_entities": ["sertraline 50mg", "major depressive disorder"],
                "turns": [
                    {"turn": 1, "message": "Turn 1"},
                    {"turn": 2, "message": "Turn 2"},
                    {"turn": 3, "message": "Turn 3"},
                    {"turn": 4, "message": "Turn 4"},
                    {"turn": 5, "message": "Turn 5"},
                ],
                "metadata": {"persona_id": "aisha", "source_openr1_ids": [16]},
            }
        ]
    }
    (data_dir / "study_c_test.json").write_text(json.dumps(payload), encoding="utf-8")

    cache_path = tmp_path / "study_c_generations.jsonl"
    model = _RepeatingRunner(config=GenerationConfig(max_tokens=64))

    run_study_c(
        model=model,
        data_dir=str(data_dir),
        max_cases=1,
        output_dir=str(tmp_path),
        model_name="dummy",
        use_nli=False,
        generate_only=True,
        cache_out=str(cache_path),
        context_cleaner="scan",
        context_clean_start_turn=4,
    )

    rows = [json.loads(ln) for ln in cache_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dialogue_turn_5 = next(r for r in rows if r.get("variant") == "dialogue" and r.get("turn_num") == 5)

    repeated_line = "REPEAT ME " * 10
    occurrences = dialogue_turn_5["conversation_text"].count(repeated_line)
    # _prepare_response_for_context applies scan-mode dedup to ALL turns,
    # so each turn keeps only 1 copy of the repeated line.
    assert occurrences == 4


class _ContextOverflowRunner(ModelRunner):
    def __init__(
        self,
        model_name: str = "dummy",
        config: Optional[GenerationConfig] = None,
        max_prompt_chars: int = 2600,
    ):
        super().__init__(model_name=model_name, config=config or GenerationConfig())
        self.max_prompt_chars = max_prompt_chars
        self.chat_prompt_sizes: list[int] = []
        self.response = "Assistant response. " + ("stabilised context " * 80)

    def generate(self, prompt: str, mode: str = "default") -> str:
        return f"{mode.upper()}::summary"

    def generate_with_reasoning(self, prompt: str):
        return self.generate(prompt), "reasoning"

    def chat(self, messages, mode: str = "default") -> str:
        prompt_chars = sum(len(str(message.get("content", ""))) for message in messages)
        self.chat_prompt_sizes.append(prompt_chars)
        if prompt_chars > self.max_prompt_chars:
            raise RuntimeError(
                "400 Client Error: Bad Request for url: "
                "http://127.0.0.1:1234/v1/chat/completions | "
                'response={"error":"Context size has been exceeded."}'
            )
        return self.response


@pytest.mark.unit
def test_context_overflow_retries_trim_history(tmp_path: Path) -> None:
    data_dir = tmp_path / "openr1_psy_splits"
    data_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "cases": [
            {
                "id": "c_overflow",
                "patient_summary": "Patient summary",
                "critical_entities": ["insomnia", "panic attacks"],
                "turns": [
                    {"turn": 1, "message": "TURN_1 " + ("alpha " * 80)},
                    {"turn": 2, "message": "TURN_2 " + ("beta " * 80)},
                    {"turn": 3, "message": "TURN_3 " + ("gamma " * 80)},
                    {"turn": 4, "message": "TURN_4 " + ("delta " * 80)},
                ],
                "metadata": {"persona_id": "nora", "source_openr1_ids": [99]},
            }
        ]
    }
    (data_dir / "study_c_test.json").write_text(json.dumps(payload), encoding="utf-8")

    cache_path = tmp_path / "study_c_generations.jsonl"
    model = _ContextOverflowRunner(config=GenerationConfig(max_tokens=512))

    run_study_c(
        model=model,
        data_dir=str(data_dir),
        max_cases=1,
        output_dir=str(tmp_path),
        model_name="dummy",
        use_nli=False,
        generate_only=True,
        cache_out=str(cache_path),
        context_cleaner="scan",
    )

    rows = [json.loads(ln) for ln in cache_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dialogue_rows = [row for row in rows if row.get("variant") == "dialogue"]
    assert len(dialogue_rows) == 4
    assert all(row["status"] == "ok" for row in dialogue_rows)
    assert any(
        later < earlier for earlier, later in zip(model.chat_prompt_sizes, model.chat_prompt_sizes[1:])
    )

    final_dialogue = next(row for row in dialogue_rows if row["turn_num"] == 4)
    assert "TURN_1" not in final_dialogue["conversation_text"]
    assert final_dialogue["response_text"].startswith("Assistant response.")
