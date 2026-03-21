"""Contract checks for the source-backed frozen `v6` snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
V6_ROOT = BASE_DIR / "data" / "frozen_splits" / "v6"


def _read_json(relative_path: str):
    path = V6_ROOT / relative_path
    assert path.exists(), f"Expected data file not found: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _primary_ref(metadata: dict) -> tuple[str, int]:
    return (
        str(metadata.get("source_openr1_split", "") or "").strip().lower(),
        int(metadata.get("source_openr1_id", -1)),
    )


@pytest.mark.unit
def test_v6_root_exists_and_manifest_is_present() -> None:
    assert V6_ROOT.exists()
    manifest = _read_json("manifest.json")
    files = {entry["file"] for entry in manifest.get("files", [])}
    for rel in {
        "study_b_test.json",
        "study_b_multi_turn_test.json",
        "study_c_test.json",
        "study_c_target_plans.json",
        "entity_evidence_map.json",
        "README.md",
        "README_NOTE.txt",
    }:
        assert rel in files


@pytest.mark.unit
def test_v6_study_b_single_turn_is_fully_source_backed() -> None:
    items = _read_json("study_b_test.json")
    assert isinstance(items, list)
    assert items
    assert len(items) <= 2000

    refs: set[tuple[str, int]] = set()
    for item in items:
        metadata = item.get("metadata") or {}
        assert metadata.get("source_type") == "direct_source"
        assert metadata.get("source_split") in {"test", "train"}
        assert metadata.get("source_openr1_split") in {"test", "train"}
        assert isinstance(metadata.get("source_openr1_ids"), list)
        assert len(metadata["source_openr1_ids"]) == 1
        assert metadata["source_openr1_ids"][0] == metadata.get("source_openr1_id")
        ref = _primary_ref(metadata)
        assert ref[1] >= 0
        refs.add(ref)
    assert len(refs) == len(items)


@pytest.mark.unit
def test_v6_study_b_multi_turn_has_explicit_turn_provenance() -> None:
    cases = _read_json("study_b_multi_turn_test.json")
    assert isinstance(cases, list)
    assert cases
    assert len(cases) <= 120

    for case in cases:
        metadata = case.get("metadata") or {}
        assert metadata.get("source_openr1_split") in {"test", "train"}
        assert isinstance(metadata.get("source_openr1_ids"), list)
        assert len(metadata["source_openr1_ids"]) == 1
        counts = metadata.get("case_provenance_counts") or {}
        assert sum(int(value) for value in counts.values()) == 20
        turns = case.get("turns") or []
        assert len(turns) == 20
        for turn in turns:
            assert turn.get("provenance_type") in {
                "source",
                "retrieved_composed",
                "source_anchored_continuation",
            }
            assert isinstance(turn.get("pressure_level"), int)
            if turn.get("provenance_type") == "source":
                assert isinstance(turn.get("source_round_index"), int)
            else:
                retrieval_refs = turn.get("retrieval_refs")
                assert isinstance(retrieval_refs, list)
                assert retrieval_refs


@pytest.mark.unit
def test_v6_study_c_has_linked_plans_and_evidence() -> None:
    payload = _read_json("study_c_test.json")
    assert isinstance(payload, dict)
    cases = payload.get("cases", [])
    assert cases
    assert len(cases) <= 100

    plans = _read_json("study_c_target_plans.json")
    evidence = _read_json("entity_evidence_map.json")
    plan_items = plans.get("plans", {})
    case_evidence = evidence.get("case_evidence", {})

    for case in cases:
        case_id = case["id"]
        metadata = case.get("metadata") or {}
        assert metadata.get("source_openr1_split") in {"test", "train"}
        assert isinstance(metadata.get("source_openr1_ids"), list)
        assert metadata["source_openr1_ids"]
        turns = case.get("turns") or []
        assert len(turns) == 20
        assert case.get("num_turns") == 20
        assert case_id in plan_items
        assert case_id in case_evidence
        assert case.get("critical_entities")
        for turn in turns:
            if turn.get("provenance_type") == "source":
                assert isinstance(turn.get("source_round_index"), int)
            else:
                retrieval_refs = turn.get("retrieval_refs")
                assert isinstance(retrieval_refs, list)


@pytest.mark.unit
def test_v6_rebuilt_primary_refs_are_cross_study_disjoint() -> None:
    study_b_single = _read_json("study_b_test.json")
    study_b_multi = _read_json("study_b_multi_turn_test.json")
    study_c_cases = _read_json("study_c_test.json").get("cases", [])

    single_refs = {_primary_ref(item.get("metadata") or {}) for item in study_b_single}
    multi_refs = {_primary_ref(case.get("metadata") or {}) for case in study_b_multi}
    study_c_refs = {_primary_ref(case.get("metadata") or {}) for case in study_c_cases}

    assert all(ref[0] in {"test", "train"} and ref[1] >= 0 for ref in single_refs | multi_refs | study_c_refs)
    assert single_refs.isdisjoint(multi_refs)
    assert single_refs.isdisjoint(study_c_refs)
    assert multi_refs.isdisjoint(study_c_refs)
