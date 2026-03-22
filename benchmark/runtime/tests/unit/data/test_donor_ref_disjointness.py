"""Tests that multi-turn studies have disjoint turn-level source IDs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
V6_1_ROOT = BASE_DIR / "data" / "frozen_splits" / "v6_1"


def _turn_refs(path: Path) -> set[tuple[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload if isinstance(payload, list) else payload.get("cases", [])
    refs: set[tuple[str, int]] = set()
    for case in cases:
        for turn in case.get("turns", []):
            split = str(turn.get("source_openr1_split", "") or "").strip().lower()
            sid = turn.get("source_openr1_id")
            if split and sid is not None:
                refs.add((split, int(sid)))
    return refs


def _case_metadata_refs(path: Path) -> set[tuple[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload if isinstance(payload, list) else payload.get("cases", [])
    refs: set[tuple[str, int]] = set()
    for case in cases:
        metadata = case.get("metadata", {}) or {}
        split = str(metadata.get("source_openr1_split", "") or "").strip().lower()
        for sid in metadata.get("source_openr1_ids", []) or []:
            if split:
                refs.add((split, int(sid)))
    return refs


@pytest.mark.unit
def test_study_b_mt_and_c_have_disjoint_turn_level_ids():
    """Study B MT and Study C must not share any turn-level source IDs."""
    if not V6_1_ROOT.exists():
        pytest.skip("v6.1 data not present")

    b_mt_refs = _turn_refs(V6_1_ROOT / "study_b_multi_turn_test.json")
    c_refs = _turn_refs(V6_1_ROOT / "study_c_test.json")
    overlap = b_mt_refs & c_refs
    assert not overlap, (
        f"B MT and C share {len(overlap)} turn-level source IDs: {sorted(overlap)[:10]}"
    )


@pytest.mark.unit
def test_metadata_source_ids_include_all_donors():
    """Multi-turn case metadata.source_openr1_ids must include all donor IDs,
    not just the primary."""
    if not V6_1_ROOT.exists():
        pytest.skip("v6.1 data not present")

    for rel in ("study_b_multi_turn_test.json", "study_c_test.json"):
        path = V6_1_ROOT / rel
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = payload if isinstance(payload, list) else payload.get("cases", [])
        for case in cases:
            metadata = case.get("metadata", {}) or {}
            meta_ids = set(int(x) for x in metadata.get("source_openr1_ids", []) or [])
            turn_ids = set()
            for turn in case.get("turns", []):
                sid = turn.get("source_openr1_id")
                if sid is not None:
                    turn_ids.add(int(sid))
            missing = turn_ids - meta_ids
            assert not missing, (
                f"{rel} case {case.get('id')}: turn-level IDs {missing} "
                f"not in metadata.source_openr1_ids"
            )


@pytest.mark.unit
def test_all_study_pairs_have_disjoint_source_ids():
    """No two studies should share any source IDs at any level."""
    if not V6_1_ROOT.exists():
        pytest.skip("v6.1 data not present")

    def _all_refs(path: Path) -> set[tuple[str, int]]:
        return _turn_refs(path) | _case_metadata_refs(path)

    studies = {
        "study_a": V6_1_ROOT / "study_a_test.json",
        "study_a_bias": V6_1_ROOT / "adversarial_bias" / "biased_vignettes.json",
        "study_b": V6_1_ROOT / "study_b_test.json",
        "study_b_mt": V6_1_ROOT / "study_b_multi_turn_test.json",
        "study_c": V6_1_ROOT / "study_c_test.json",
    }

    # Study A and Study A Bias share sources by design (pair groups),
    # so we check all other pairs
    skip_pairs = {("study_a", "study_a_bias")}

    refs = {}
    for name, path in studies.items():
        if path.exists():
            refs[name] = _all_refs(path)

    violations = []
    names = sorted(refs.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (a, b) in skip_pairs or (b, a) in skip_pairs:
                continue
            overlap = refs[a] & refs[b]
            if overlap:
                violations.append(f"{a} ∩ {b}: {len(overlap)} shared IDs")

    assert not violations, "Cross-study source ID leakage:\n" + "\n".join(violations)
