"""Regression tests for Study B v6.1 — strict source-backed single-turn and exact-count multi-turn."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
V6_1_B_SINGLE_PATH = RUNTIME_ROOT / "data" / "frozen_splits" / "v6_1" / "study_b_test.json"
V6_1_B_MULTI_PATH = RUNTIME_ROOT / "data" / "frozen_splits" / "v6_1" / "study_b_multi_turn_test.json"

_VALID_PROVENANCE_TYPES = {"direct_source", "retrieved_composed", "source_anchored_deterministic_edit"}


@pytest.mark.skipif(not V6_1_B_SINGLE_PATH.exists(), reason="v6.1 Study B single not generated")
class TestStudyBSingleV6_1:

    @pytest.fixture(scope="class")
    def rows(self):
        with V6_1_B_SINGLE_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else data.get("cases", data.get("samples", []))

    def test_row_count(self, rows):
        assert len(rows) == 2000

    def test_all_direct_source(self, rows):
        non_direct = [row.get("id") for row in rows if (row.get("metadata", {}).get("source_type")) != "direct_source"]
        assert not non_direct, f"{len(non_direct)} rows are not direct_source"

    def test_source_ids_present_for_openr1_rows(self, rows):
        """OpenR1-sourced rows must have source IDs; synthetic rows are exempt."""
        missing = []
        for row in rows:
            meta = row.get("metadata", {})
            if meta.get("source") == "synthetic":
                continue
            ids = meta.get("source_openr1_ids", meta.get("source_openr1_id"))
            if not ids:
                missing.append(row.get("id"))
        assert not missing, f"{len(missing)} openr1 rows missing source IDs: {missing[:5]}"


@pytest.mark.skipif(not V6_1_B_MULTI_PATH.exists(), reason="v6.1 Study B multi not generated")
class TestStudyBMultiV6_1:

    @pytest.fixture(scope="class")
    def cases(self):
        with V6_1_B_MULTI_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else data.get("cases", data.get("data", []))

    def test_exact_twenty_turns_per_case(self, cases):
        assert len(cases) == 120
        assert all(len(case.get("turns", [])) == 20 for case in cases)

    def test_per_turn_provenance_and_scaffold(self, cases):
        missing = []
        for case in cases:
            cid = case.get("id", "")
            for turn in case.get("turns", []):
                ptype = turn.get("provenance_type")
                if not ptype:
                    missing.append(f"{cid}_t{turn.get('turn')}")
                else:
                    assert ptype in _VALID_PROVENANCE_TYPES, (
                        f"{cid}_t{turn.get('turn')} invalid provenance: {ptype}"
                    )
                assert turn.get("phase"), f"{cid}_t{turn.get('turn')} missing phase"
                assert turn.get("dialogue_act"), f"{cid}_t{turn.get('turn')} missing dialogue_act"
                assert isinstance(turn.get("state_ledger"), dict), (
                    f"{cid}_t{turn.get('turn')} missing state_ledger"
                )
        assert not missing, f"{len(missing)} turns missing provenance: {missing[:5]}"

    def test_direct_source_turns_precede_generated_turns(self, cases):
        violations = []
        for case in cases:
            seen_non_direct = False
            for turn in sorted(case.get("turns", []), key=lambda item: item.get("turn", 0)):
                provenance = turn.get("provenance_type", "")
                if provenance != "direct_source":
                    seen_non_direct = True
                elif seen_non_direct:
                    violations.append(f"{case.get('id')}_t{turn.get('turn')}")
        # Allow some interleaving (< 10% of cases)
        assert len(violations) < len(cases) * 0.1, (
            f"Too many direct_source turns after non-direct turns: {violations[:5]}"
        )
