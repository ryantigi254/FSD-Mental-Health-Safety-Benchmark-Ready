"""Regression tests for Study A Bias v6.1 — strict no-generation deterministic edits."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
V6_1_BIAS_PATH = RUNTIME_ROOT / "data" / "frozen_splits" / "v6_1" / "adversarial_bias" / "biased_vignettes.json"

pytestmark = pytest.mark.skipif(
    not V6_1_BIAS_PATH.exists(),
    reason="v6.1 bias data not yet generated",
)


@pytest.fixture(scope="module")
def v6_1_cases():
    with V6_1_BIAS_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("cases", [])


class TestNoVisibleProbeSuffix:
    _PROBE_PATTERNS = [
        re.compile(r"Bias probe profile", re.IGNORECASE),
        re.compile(r"Assess this case clinically", re.IGNORECASE),
        re.compile(r"\[INSERT\b", re.IGNORECASE),
        re.compile(r"\[PLACEHOLDER\b", re.IGNORECASE),
    ]

    def test_no_probe_in_any_prompt(self, v6_1_cases):
        violations = []
        for case in v6_1_cases:
            prompt = case.get("prompt", "")
            for pat in self._PROBE_PATTERNS:
                if pat.search(prompt):
                    violations.append((case.get("id"), pat.pattern))
        assert not violations, f"Probe text found in {len(violations)} rows: {violations[:5]}"


class TestPairGroupIntegrity:
    def test_pair_groups_have_exactly_two_members(self, v6_1_cases):
        groups: dict[str, list[dict]] = defaultdict(list)
        for case in v6_1_cases:
            groups[str(case.get("pair_group_id") or "")].append(case)
        assert all(len(items) == 2 for items in groups.values())

    def test_bias_feature_and_label_present(self, v6_1_cases):
        missing_feature = [c["id"] for c in v6_1_cases if not c.get("bias_feature")]
        missing_label = [c["id"] for c in v6_1_cases if not c.get("bias_label")]
        assert not missing_feature, f"{len(missing_feature)} cases missing bias_feature"
        assert not missing_label, f"{len(missing_label)} cases missing bias_label"


class TestProvenanceFields:
    def test_strict_provenance_present(self, v6_1_cases):
        for case in v6_1_cases:
            meta = case.get("metadata", {})
            assert meta.get("source_type") == "source_anchored_deterministic_edit", (
                f"Case {case.get('id')} source_type={meta.get('source_type')}"
            )
            assert meta.get("manual_review_required") is False, (
                f"Case {case.get('id')} manual_review_required={meta.get('manual_review_required')}"
            )

    def test_source_openr1_ids_present(self, v6_1_cases):
        missing = [
            c["id"] for c in v6_1_cases
            if not (c.get("metadata", {}).get("source_openr1_ids") or c.get("metadata", {}).get("source_openr1_id"))
        ]
        assert not missing, f"{len(missing)} cases missing source_openr1_ids: {missing[:5]}"
