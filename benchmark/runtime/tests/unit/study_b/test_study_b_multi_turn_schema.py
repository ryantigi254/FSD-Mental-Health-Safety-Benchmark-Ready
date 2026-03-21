import json
from pathlib import Path


def test_study_b_multi_turn_schema_fields_present():
    path = (
        Path(__file__).resolve().parents[3]
        / "data"
        / "openr1_psy_splits"
        / "study_b_multi_turn_test.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload if isinstance(payload, list) else payload.get("cases", [])

    assert len(cases) == 120
    for case in cases:
        assert case.get("incorrect_opinion")
        assert case.get("pressure_style") in {
            "self_doubt",
            "social_proof",
            "authority_pressure",
        }
        assert case.get("pressure_schedule") in {"early_spike", "gradual", "late_spike"}
        assert case.get("turns")
        assert len(case["turns"]) == 20
        metadata = case.get("metadata", {})
        assert metadata.get("source_split") in {"test", "train"}
        assert isinstance(metadata.get("source_openr1_ids"), list)
        assert len(metadata["source_openr1_ids"]) == 1
        for turn in case["turns"]:
            assert isinstance(turn.get("pressure_level"), int)
            assert 0 <= turn["pressure_level"] <= 3
