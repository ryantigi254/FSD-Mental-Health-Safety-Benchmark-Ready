import json
import pytest
from pathlib import Path
from reliable_clinical_benchmark.data.study_b_loader import load_study_b_data, load_multi_turn_cases

def test_load_study_b_data_flat_list(tmp_path):
    data = [
        {
            "id": "b_001",
            "prompt": "Test prompt",
            "gold_answer": "Depression",
            "incorrect_opinion": "Anxiety",
            "metadata": {"persona_id": "aisha"}
        }
    ]
    data_file = tmp_path / "study_b_test.json"
    with open(data_file, "w") as f:
        json.dump(data, f)
    
    items = load_study_b_data(str(data_file))
    assert len(items) == 1
    assert items[0].id == "b_001"
    assert items[0].metadata["persona_id"] == "aisha"

def test_load_study_b_data_legacy_dict(tmp_path):
    data = {
        "samples": [
            {
                "id": "b_002",
                "prompt": "Test prompt 2",
                "gold_answer": "PTSD",
                "incorrect_opinion": "Anxiety",
                "metadata": {"persona_id": "jamal"}
            }
        ]
    }
    data_file = tmp_path / "study_b_legacy.json"
    with open(data_file, "w") as f:
        json.dump(data, f)
    
    items = load_study_b_data(str(data_file))
    assert len(items) == 1
    assert items[0].id == "b_002"

def test_load_multi_turn_cases_flat_list(tmp_path):
    data = [
        {
            "id": "mt_001",
            "gold_answer": "Depression",
            "turns": [{"message": "User turn 1"}],
            "metadata": {"persona_id": "maya"}
        }
    ]
    data_file = tmp_path / "study_b_multi_turn.json"
    with open(data_file, "w") as f:
        json.dump(data, f)
    
    cases = load_multi_turn_cases(str(data_file))
    assert len(cases) == 1
    assert cases[0]["id"] == "mt_001"


def test_load_multi_turn_cases_reads_utf8_json(tmp_path):
    data = [
        {
            "id": "mt_utf8",
            "gold_answer": "Depression",
            "turns": [{"message": "Client reports intrusive thoughts 😅 and 漢字 details"}],
            "metadata": {"persona_id": "maya"},
        }
    ]
    data_file = tmp_path / "study_b_multi_turn_utf8.json"
    data_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    cases = load_multi_turn_cases(str(data_file))

    assert len(cases) == 1
    assert cases[0]["turns"][0]["message"].endswith("漢字 details")
