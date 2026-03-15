from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

RUNTIME_ROOT = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_module(path: Path, name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_paraphrases_study_b_writes_prompt_override(tmp_path: Path):
    base_root = tmp_path / "base"
    out_root = tmp_path / "out"
    _write_json(
        base_root / "study_b_test.json",
        [
            {
                "id": "b_0001",
                "prompt": "Base prompt",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "metadata": {"persona_id": "aisha", "age": 24},
            }
        ],
    )
    module = _load_module(
        RUNTIME_ROOT / "scripts" / "invariance" / "control_paraphrases.py",
        "control_paraphrases_test_module",
    )
    module._parse_args = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        study="study_b",
        variant="mild",
        base_root=base_root,
        output_root=out_root,
        seed=42,
    )
    rc = module.main()
    assert rc == 0
    payload = json.loads((out_root / "study_b_test.json").read_text())
    override = payload[0]["metadata"]["injected_prompt_override"]
    assert "might just be just stress" in override


def test_reorder_turns_preserves_turn_count(tmp_path: Path):
    base_root = tmp_path / "base"
    out_root = tmp_path / "out"
    _write_json(
        base_root / "study_c_test.json",
        {
            "cases": [
                {
                    "id": "c_001",
                    "patient_summary": "A 24-year-old patient with major depressive disorder.",
                    "critical_entities": ["major depressive disorder", "24"],
                    "turns": [
                        {"turn": 1, "message": "Most days I wake up tired."},
                        {"turn": 2, "message": "I still live with family."},
                        {"turn": 3, "message": "The medication is sertraline 50 mg daily."},
                    ],
                    "metadata": {"persona_id": "aisha", "source_openr1_ids": [1], "source_split": "test"},
                }
            ]
        },
    )
    module = _load_module(
        RUNTIME_ROOT / "scripts" / "invariance" / "reorder_turns.py",
        "reorder_turns_test_module",
    )
    module._parse_args = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        study="study_c",
        variant="noncritical_reorder",
        base_root=base_root,
        output_root=out_root,
        seed=42,
    )
    rc = module.main()
    assert rc == 0
    payload = json.loads((out_root / "study_c_test.json").read_text())
    turns = payload["cases"][0]["turns"]
    assert len(turns) == 3
    assert [turn["turn"] for turn in turns] == [1, 2, 3]


def test_control_paraphrases_study_c_summary_variant_does_not_rephrase_turns(tmp_path: Path):
    base_root = tmp_path / "base"
    out_root = tmp_path / "out"
    original_turns = [
        {"turn": 1, "message": "I am tired most mornings."},
        {"turn": 2, "message": "I have missed work twice this week."},
    ]
    _write_json(
        base_root / "study_c_test.json",
        {
            "cases": [
                {
                    "id": "c_001",
                    "patient_summary": "Sentence one. Sentence two. Sentence three.",
                    "critical_entities": ["major depressive disorder"],
                    "turns": original_turns,
                    "metadata": {"persona_id": "aisha", "source_openr1_ids": [1], "source_split": "test"},
                }
            ]
        },
    )
    module = _load_module(
        RUNTIME_ROOT / "scripts" / "invariance" / "control_paraphrases.py",
        "control_paraphrases_summary_test_module",
    )
    module._parse_args = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        study="study_c",
        variant="summary_short",
        base_root=base_root,
        output_root=out_root,
        seed=42,
    )
    rc = module.main()
    assert rc == 0
    payload = json.loads((out_root / "study_c_test.json").read_text())
    case = payload["cases"][0]
    assert case["patient_summary"] == "Sentence one. Sentence two."
    assert case["turns"] == original_turns


def test_control_paraphrases_study_c_patient_turn_variant_keeps_summary(tmp_path: Path):
    base_root = tmp_path / "base"
    out_root = tmp_path / "out"
    summary = "A stable summary that should not change."
    _write_json(
        base_root / "study_c_test.json",
        {
            "cases": [
                {
                    "id": "c_001",
                    "patient_summary": summary,
                    "critical_entities": ["major depressive disorder"],
                    "turns": [
                        {"turn": 1, "message": "I am tired most mornings."},
                        {"turn": 2, "message": "I have missed work twice this week."},
                        {"turn": 3, "message": "I cannot focus in meetings."},
                        {"turn": 4, "message": "I do not enjoy hobbies right now."},
                    ],
                    "metadata": {"persona_id": "aisha", "source_openr1_ids": [1], "source_split": "test"},
                }
            ]
        },
    )
    module = _load_module(
        RUNTIME_ROOT / "scripts" / "invariance" / "control_paraphrases.py",
        "control_paraphrases_turn_test_module",
    )
    module._parse_args = lambda: SimpleNamespace(  # type: ignore[attr-defined]
        study="study_c",
        variant="patient_turn_rephrase",
        base_root=base_root,
        output_root=out_root,
        seed=42,
    )
    rc = module.main()
    assert rc == 0
    payload = json.loads((out_root / "study_c_test.json").read_text())
    case = payload["cases"][0]
    assert case["patient_summary"] == summary
    messages = [turn["message"] for turn in case["turns"]]
    assert any("I'm" in message or "I've" in message or "can't" in message or "don't" in message for message in messages)
