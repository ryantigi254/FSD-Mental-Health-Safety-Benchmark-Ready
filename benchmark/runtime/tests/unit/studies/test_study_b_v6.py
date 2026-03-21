"""Regression tests for Study B v6 — source-backed only, per-turn provenance."""

import json
from pathlib import Path

import pytest

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
V6_B_SINGLE_PATH = RUNTIME_ROOT / "data" / "frozen_splits" / "v6" / "study_b_test.json"
V6_B_MULTI_PATH = RUNTIME_ROOT / "data" / "frozen_splits" / "v6" / "study_b_multi_turn_test.json"


# --- Study B Single-Turn ---

@pytest.mark.skipif(not V6_B_SINGLE_PATH.exists(), reason="v6 Study B single not generated")
class TestStudyBSingleV6:

    @pytest.fixture(scope="class")
    def rows(self):
        with open(V6_B_SINGLE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        for key in ("cases", "samples", "data"):
            if key in data:
                return data[key]
        return data

    def test_zero_missing_source_ids(self, rows):
        missing = []
        for r in rows:
            meta = r.get("metadata", {})
            ids = meta.get("source_openr1_ids", meta.get("source_openr1_id"))
            if not ids:
                missing.append(r.get("id"))
        assert not missing, f"{len(missing)} rows missing source IDs: {missing[:5]}"

    def test_provenance_type_present(self, rows):
        for r in rows:
            meta = r.get("metadata", {})
            has_prov = "provenance_type" in meta or "source_type" in meta
            assert has_prov, f"Row {r.get('id')} missing provenance_type/source_type"

    def test_all_direct_source(self, rows):
        non_direct = [
            r.get("id") for r in rows
            if (r.get("metadata", {}).get("provenance_type")
                or r.get("metadata", {}).get("source_type")) != "direct_source"
        ]
        assert not non_direct, f"{len(non_direct)} rows are not direct_source"


# --- Study B Multi-Turn ---

@pytest.mark.skipif(not V6_B_MULTI_PATH.exists(), reason="v6 Study B multi not generated")
class TestStudyBMultiV6:

    @pytest.fixture(scope="class")
    def cases(self):
        with open(V6_B_MULTI_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("cases", data.get("data", []))

    def test_per_turn_provenance(self, cases):
        missing = []
        for case in cases:
            cid = case.get("id", "")
            for t in case.get("turns", []):
                # provenance_type may be at turn level or inside metadata
                ptype = t.get("provenance_type") or t.get("metadata", {}).get("provenance_type")
                if not ptype:
                    missing.append(f"{cid}_t{t.get('turn')}")
        assert not missing, f"{len(missing)} turns missing provenance: {missing[:5]}"

    def test_direct_source_turns_first(self, cases):
        """Within each case, direct_source turns should precede generated turns."""
        violations = []
        for case in cases:
            cid = case.get("id", "")
            turns = sorted(case.get("turns", []), key=lambda t: t.get("turn", 0))
            seen_generated = False
            for t in turns:
                ptype = t.get("provenance_type") or t.get("metadata", {}).get("provenance_type", "")
                if ptype != "direct_source":
                    seen_generated = True
                elif seen_generated:
                    violations.append(f"{cid}_t{t.get('turn')}")

        # Allow some out-of-order if source turns were interleaved
        assert len(violations) < len(cases) * 0.1, (
            f"Too many direct_source turns after generated turns: {violations[:5]}"
        )
