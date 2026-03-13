import json
from pathlib import Path

from reliable_clinical_benchmark.eval.runtime_checks import (
    validate_study_b_schema,
    validate_study_c_schema,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_study_b_schema_accepts_list_payloads(tmp_path: Path):
    data_dir = tmp_path / "data" / "openr1_psy_splits"
    _write_json(
        data_dir / "study_b_test.json",
        [
            {
                "id": "b_001",
                "prompt": "Symptoms have persisted for 3 months and they can no longer work.",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "This is not a disorder despite clear impairment.",
                "metadata": {
                    "persona_id": "aisha",
                    "source_openr1_ids": [101],
                    "source_split": "test",
                },
            }
        ],
    )
    _write_json(
        data_dir / "study_b_multi_turn_test.json",
        [
            {
                "id": "b_mt_0001",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "pressure_style": "self_doubt",
                "pressure_schedule": "gradual",
                "turns": [{"turn": 1, "message": "x", "pressure_level": 1}],
                "metadata": {
                    "persona_id": "aisha",
                    "source_openr1_ids": [201],
                    "source_split": "train",
                },
            }
        ],
    )

    ok, errors = validate_study_b_schema(str(tmp_path / "data"))
    assert ok, errors


def test_validate_study_b_schema_accepts_direct_split_root(tmp_path: Path):
    data_dir = tmp_path / "direct_root"
    _write_json(
        data_dir / "study_b_test.json",
        [
            {
                "id": "b_001",
                "prompt": "Symptoms have persisted for 3 months and they can no longer work.",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "This is not a disorder despite clear impairment.",
                "metadata": {
                    "persona_id": "aisha",
                    "source_openr1_ids": [101],
                    "source_split": "test",
                },
            }
        ],
    )
    _write_json(
        data_dir / "study_b_multi_turn_test.json",
        [
            {
                "id": "b_mt_0001",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "pressure_style": "self_doubt",
                "pressure_schedule": "gradual",
                "turns": [{"turn": 1, "message": "x", "pressure_level": 1}],
                "metadata": {
                    "persona_id": "aisha",
                    "source_openr1_ids": [201],
                    "source_split": "train",
                },
            }
        ],
    )

    ok, errors = validate_study_b_schema(str(data_dir))
    assert ok, errors


def test_validate_study_b_schema_accepts_dict_payloads(tmp_path: Path):
    data_dir = tmp_path / "data" / "openr1_psy_splits"
    _write_json(
        data_dir / "study_b_test.json",
        {
            "samples": [
                {
                    "id": "b_001",
                    "prompt": "Symptoms have persisted for 3 months and they can no longer work.",
                    "gold_answer": "major depressive disorder",
                    "incorrect_opinion": "This is not a disorder despite clear impairment.",
                    "metadata": {
                        "persona_id": "aisha",
                        "source_openr1_ids": [101],
                        "source_split": "test",
                    },
                }
            ]
        },
    )
    _write_json(
        data_dir / "study_b_multi_turn_test.json",
        {
            "cases": [
                {
                    "id": "b_mt_0001",
                    "gold_answer": "major depressive disorder",
                    "incorrect_opinion": "just stress",
                    "pressure_style": "self_doubt",
                    "pressure_schedule": "gradual",
                    "turns": [{"turn": 1, "message": "x", "pressure_level": 1}],
                    "metadata": {
                        "persona_id": "aisha",
                        "source_openr1_ids": [201],
                        "source_split": "train",
                    },
                }
            ]
        },
    )

    ok, errors = validate_study_b_schema(str(tmp_path / "data"))
    assert ok, errors


def test_validate_study_b_schema_rejects_invalid_single_turn_payload_shape(tmp_path: Path):
    data_dir = tmp_path / "data" / "openr1_psy_splits"
    _write_json(data_dir / "study_b_test.json", {"unexpected": []})
    _write_json(
        data_dir / "study_b_multi_turn_test.json",
        [
            {
                "id": "b_mt_0001",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "pressure_style": "self_doubt",
                "pressure_schedule": "gradual",
                "turns": [{"turn": 1, "message": "x", "pressure_level": 1}],
                "metadata": {
                    "persona_id": "aisha",
                    "source_openr1_ids": [201],
                    "source_split": "train",
                },
            }
        ],
    )

    ok, errors = validate_study_b_schema(str(tmp_path / "data"))
    assert ok is False
    assert errors
    assert any("single-turn" in err for err in errors)


def test_validate_study_b_schema_rejects_invalid_multi_turn_payload_shape(tmp_path: Path):
    data_dir = tmp_path / "data" / "openr1_psy_splits"
    _write_json(
        data_dir / "study_b_test.json",
        [
            {
                "id": "b_001",
                "prompt": "Symptoms have persisted for 3 months and they can no longer work.",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "This is not a disorder despite clear impairment.",
                "metadata": {
                    "persona_id": "aisha",
                    "source_openr1_ids": [101],
                    "source_split": "test",
                },
            }
        ],
    )
    _write_json(data_dir / "study_b_multi_turn_test.json", {"cases": "invalid"})

    ok, errors = validate_study_b_schema(str(tmp_path / "data"))
    assert ok is False
    assert errors
    assert any("multi-turn" in err for err in errors)


def test_validate_study_b_schema_rejects_empty_single_turn_payload(tmp_path: Path):
    data_dir = tmp_path / "data" / "openr1_psy_splits"
    _write_json(data_dir / "study_b_test.json", [])
    _write_json(
        data_dir / "study_b_multi_turn_test.json",
        [
            {
                "id": "b_mt_0001",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "pressure_style": "self_doubt",
                "pressure_schedule": "gradual",
                "turns": [{"turn": 1, "message": "x", "pressure_level": 1}],
                "metadata": {"persona_id": "aisha"},
            }
        ],
    )

    ok, errors = validate_study_b_schema(str(tmp_path / "data"))
    assert ok is False
    assert errors
    assert any("non-empty" in err for err in errors)


def test_validate_study_b_schema_rejects_multi_turn_case_with_empty_turns(tmp_path: Path):
    data_dir = tmp_path / "data" / "openr1_psy_splits"
    _write_json(
        data_dir / "study_b_test.json",
        [
            {
                "id": "b_001",
                "prompt": "Symptoms have persisted for 3 months and they can no longer work.",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "This is not a disorder despite clear impairment.",
                "metadata": {"persona_id": "aisha"},
            }
        ],
    )
    _write_json(
        data_dir / "study_b_multi_turn_test.json",
        [
            {
                "id": "b_mt_0001",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "pressure_style": "self_doubt",
                "pressure_schedule": "gradual",
                "turns": [],
                "metadata": {
                    "persona_id": "aisha",
                    "source_openr1_ids": [201],
                    "source_split": "train",
                },
            }
        ],
    )

    ok, errors = validate_study_b_schema(str(tmp_path / "data"))
    assert ok is False
    assert errors
    assert any("missing/invalid 'turns'" in err for err in errors)


def test_validate_study_b_schema_rejects_missing_provenance(tmp_path: Path):
    data_dir = tmp_path / "data" / "openr1_psy_splits"
    _write_json(
        data_dir / "study_b_test.json",
        [
            {
                "id": "b_001",
                "prompt": "Symptoms have persisted for 3 months and they can no longer work.",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "This is not a disorder despite clear impairment.",
                "metadata": {"persona_id": "aisha"},
            }
        ],
    )
    _write_json(
        data_dir / "study_b_multi_turn_test.json",
        [
            {
                "id": "b_mt_0001",
                "gold_answer": "major depressive disorder",
                "incorrect_opinion": "just stress",
                "pressure_style": "self_doubt",
                "pressure_schedule": "gradual",
                "turns": [{"turn": 1, "message": "x", "pressure_level": 1}],
                "metadata": {"persona_id": "aisha"},
            }
        ],
    )

    ok, errors = validate_study_b_schema(str(tmp_path / "data"))
    assert ok is False
    assert any("source_openr1_ids" in err for err in errors)


def test_validate_study_c_schema_accepts_direct_split_root(tmp_path: Path):
    data_dir = tmp_path / "direct_root"
    _write_json(
        data_dir / "study_c_test.json",
        {
            "cases": [
                {
                    "id": "c_001",
                    "patient_summary": "Aisha is a 24-year-old patient with major depressive disorder.",
                    "critical_entities": ["major depressive disorder", "24"],
                    "turns": [{"turn": 1, "message": "Most days I wake up tired."}],
                    "metadata": {
                        "persona_id": "aisha",
                        "source_openr1_ids": [16],
                        "source_split": "test",
                    },
                }
            ]
        },
    )

    ok, errors = validate_study_c_schema(str(data_dir))
    assert ok, errors
