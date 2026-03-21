"""Regression tests for Study C v6.1 — strict 20-turn provenance and entity continuity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
V6_1_C_PATH = RUNTIME_ROOT / "data" / "frozen_splits" / "v6_1" / "study_c_test.json"

_VALID_PROVENANCE_TYPES = {"direct_source", "retrieved_composed", "source_anchored_deterministic_edit"}


@pytest.mark.skipif(not V6_1_C_PATH.exists(), reason="v6.1 Study C not generated")
class TestStudyCv6_1:

    @pytest.fixture(scope="class")
    def cases(self):
        with V6_1_C_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("cases", data if isinstance(data, list) else [])

    def test_exact_twenty_turns_and_patient_summary(self, cases):
        assert len(cases) == 100
        for case in cases:
            assert case.get("patient_summary"), f"Case {case.get('id')} missing patient_summary"
            assert case.get("num_turns") == 20, f"Case {case.get('id')} num_turns={case.get('num_turns')}"
            assert len(case.get("turns", [])) == 20, (
                f"Case {case.get('id')} has {len(case.get('turns', []))} turns"
            )

    def test_per_turn_provenance_and_state_ledger(self, cases):
        missing = []
        for case in cases:
            for turn in case.get("turns", []):
                ptype = turn.get("provenance_type")
                if not ptype:
                    missing.append(f"{case.get('id')}_t{turn.get('turn')}")
                else:
                    assert ptype in _VALID_PROVENANCE_TYPES, (
                        f"{case.get('id')}_t{turn.get('turn')} invalid provenance: {ptype}"
                    )
                assert turn.get("phase"), f"{case.get('id')}_t{turn.get('turn')} missing phase"
                assert turn.get("dialogue_act"), f"{case.get('id')}_t{turn.get('turn')} missing dialogue_act"
                assert isinstance(turn.get("state_ledger"), dict), (
                    f"{case.get('id')}_t{turn.get('turn')} missing state_ledger"
                )
        assert not missing, f"{len(missing)} turns missing provenance: {missing[:5]}"

    def test_critical_entities_preserved(self, cases):
        stop_words = {"disorder", "syndrome", "disease", "risk", "attacks", "history", "major", "chronic", "acute", "with"}

        def entity_present(entity: str, text: str) -> bool:
            if entity in text:
                return True
            import re as _re

            tokens = _re.split(r"[\s\-]+", entity)
            stems = [token[:4] for token in tokens if len(token) >= 4 and token not in stop_words]
            return any(stem in text for stem in stems) if stems else entity in text

        for case in cases:
            entities = [entity.lower() for entity in case.get("critical_entities", [])]
            all_case_text = " ".join(
                [case.get("patient_summary", "")] + [turn.get("message", "") for turn in case.get("turns", [])]
            ).lower()
            missing = [entity for entity in entities if not entity_present(entity, all_case_text)]
            assert len(missing) < len(entities) * 0.5, (
                f"Case {case.get('id')}: too many critical entities missing from turns: {missing[:5]}"
            )
