"""Regression tests for Study C v6 — per-turn provenance and entity continuity."""

import json
from pathlib import Path

import pytest

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
V6_C_PATH = RUNTIME_ROOT / "data" / "frozen_splits" / "v6" / "study_c_test.json"


@pytest.mark.skipif(not V6_C_PATH.exists(), reason="v6 Study C not generated")
class TestStudyCv6:

    @pytest.fixture(scope="class")
    def cases(self):
        with open(V6_C_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("cases", data if isinstance(data, list) else [])

    def test_per_turn_provenance(self, cases):
        missing = []
        for case in cases:
            for turn in case.get("turns", []):
                meta = turn.get("metadata", {})
                if "provenance_type" not in meta:
                    missing.append(f"{case.get('id')}_t{turn.get('turn')}")
        assert not missing, f"{len(missing)} turns missing provenance: {missing[:5]}"

    def test_critical_entities_preserved(self, cases):
        """Critical entities from patient_summary should be present in turns."""
        for case in cases:
            entities = set(e.lower() for e in case.get("critical_entities", []))
            if not entities:
                continue
            all_turn_text = " ".join(
                t.get("message", "") for t in case.get("turns", [])
            ).lower()
            missing = {e for e in entities if e not in all_turn_text}
            # Allow some entities to be mentioned in summary only
            assert len(missing) < len(entities) * 0.5, (
                f"Case {case.get('id')}: too many critical entities missing from turns: "
                f"{list(missing)[:5]}"
            )

    def test_patient_summary_present(self, cases):
        for case in cases:
            assert case.get("patient_summary"), (
                f"Case {case.get('id')} missing patient_summary"
            )
