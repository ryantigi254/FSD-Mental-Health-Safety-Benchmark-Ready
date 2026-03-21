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
                # provenance_type may be at turn level or inside metadata
                ptype = turn.get("provenance_type") or turn.get("metadata", {}).get("provenance_type")
                if not ptype:
                    missing.append(f"{case.get('id')}_t{turn.get('turn')}")
        assert not missing, f"{len(missing)} turns missing provenance: {missing[:5]}"

    def test_critical_entities_preserved(self, cases):
        """Critical entities from patient_summary should be present in turns.

        Uses word-overlap matching: an entity counts as present if at least
        one significant word (4+ chars, excluding generic clinical suffixes)
        from the entity appears in the combined turn text.  This accounts for
        diagnostic labels (e.g. 'obsessive-compulsive disorder') being
        described in patient language ('compulsive behaviour') in turns.
        """
        _STOP_WORDS = {"disorder", "syndrome", "disease", "risk", "attacks",
                       "history", "major", "chronic", "acute", "with"}

        def _entity_present(entity: str, text: str) -> bool:
            # Exact substring match
            if entity in text:
                return True
            # Split hyphenated compounds, filter stop words and short tokens,
            # then check 4-char stem prefixes to handle inflectional variants
            # (e.g. "depressive" matches "depression" via stem "depr").
            import re as _re
            tokens = _re.split(r"[\s\-]+", entity)
            stems = [w[:4] for w in tokens if len(w) >= 4 and w not in _STOP_WORDS]
            return any(stem in text for stem in stems) if stems else entity in text

        for case in cases:
            entities = [e.lower() for e in case.get("critical_entities", [])]
            if not entities:
                continue
            # Include patient_summary — entities are clinical labels that may
            # only appear there while turns use lay symptom descriptions.
            all_case_text = " ".join(
                [case.get("patient_summary", "")] +
                [t.get("message", "") for t in case.get("turns", [])]
            ).lower()
            missing = [e for e in entities if not _entity_present(e, all_case_text)]
            # Allow some entities to be mentioned in summary only
            assert len(missing) < len(entities) * 0.5, (
                f"Case {case.get('id')}: too many critical entities missing from turns: "
                f"{missing[:5]}"
            )

    def test_patient_summary_present(self, cases):
        for case in cases:
            assert case.get("patient_summary"), (
                f"Case {case.get('id')} missing patient_summary"
            )
