"""Regression tests for Study A Bias v6 — no visible probe suffixes.

These tests validate the v6 bias data contract after the hardening
scripts have been run. They can also be run against v5 data to verify
the problem they are designed to catch.
"""

import json
import re
from pathlib import Path

import pytest

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
V6_BIAS_PATH = RUNTIME_ROOT / "data" / "frozen_splits" / "v6" / "adversarial_bias" / "biased_vignettes.json"

# Skip if v6 data not yet generated
pytestmark = pytest.mark.skipif(
    not V6_BIAS_PATH.exists(),
    reason="v6 bias data not yet generated",
)


@pytest.fixture(scope="module")
def v6_cases():
    with open(V6_BIAS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


class TestNoVisibleProbeSuffix:
    """No row should contain legacy probe metadata."""

    _PROBE_PATTERNS = [
        re.compile(r"Bias probe profile", re.IGNORECASE),
        re.compile(r"Assess this case clinically", re.IGNORECASE),
        re.compile(r"\[INSERT\b", re.IGNORECASE),
        re.compile(r"\[PLACEHOLDER\b", re.IGNORECASE),
    ]

    def test_no_probe_in_any_prompt(self, v6_cases):
        violations = []
        for case in v6_cases:
            prompt = case.get("prompt", "")
            for pat in self._PROBE_PATTERNS:
                if pat.search(prompt):
                    violations.append((case.get("id"), pat.pattern))
        assert not violations, f"Probe text found in {len(violations)} rows: {violations[:5]}"


class TestPairGroupIntegrity:
    """Pair groups should still have consistent structure."""

    def test_pair_group_ids_present(self, v6_cases):
        missing = [c["id"] for c in v6_cases if not c.get("pair_group_id")]
        assert not missing, f"{len(missing)} cases missing pair_group_id"

    def test_pair_groups_have_two_members(self, v6_cases):
        from collections import Counter
        groups = Counter(c.get("pair_group_id") for c in v6_cases)
        wrong = {gid: cnt for gid, cnt in groups.items() if cnt != 2}
        # Allow some failed pairs (dropped by fail-closed)
        # but flag if > 10% are broken
        if wrong:
            broken_pct = len(wrong) / len(groups) * 100
            assert broken_pct < 10, (
                f"{broken_pct:.1f}% of pair groups have != 2 members: "
                f"{list(wrong.items())[:5]}"
            )


class TestGoldLabelsPreserved:
    """Bias features and labels should be preserved from v5."""

    def test_bias_feature_present(self, v6_cases):
        missing = [c["id"] for c in v6_cases if not c.get("bias_feature")]
        assert not missing, f"{len(missing)} cases missing bias_feature"

    def test_bias_label_present(self, v6_cases):
        missing = [c["id"] for c in v6_cases if not c.get("bias_label")]
        assert not missing, f"{len(missing)} cases missing bias_label"


class TestProvenanceFields:
    """Every v6 row should have provenance metadata."""

    def test_provenance_type_present(self, v6_cases):
        for case in v6_cases:
            meta = case.get("metadata", {})
            assert "provenance_type" in meta, f"Case {case.get('id')} missing provenance_type"

    def test_source_openr1_ids_present(self, v6_cases):
        for case in v6_cases:
            meta = case.get("metadata", {})
            ids = meta.get("source_openr1_ids", meta.get("source_openr1_id"))
            assert ids is not None, f"Case {case.get('id')} missing source OpenR1 IDs"
