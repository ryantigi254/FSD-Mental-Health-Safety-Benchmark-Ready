"""Contract checks for the strict no-generation frozen `v6.1` snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
V6_1_ROOT = BASE_DIR / "data" / "frozen_splits" / "v6_1"
STRICT_BUILDER = BASE_DIR / "src" / "reliable_clinical_benchmark" / "data" / "strict_snapshot_v6_1.py"

_VALID_PROVENANCE_TYPES = {"direct_source", "retrieved_composed", "source_anchored_deterministic_edit"}


def _read_json(relative_path: str):
    path = V6_1_ROOT / relative_path
    assert path.exists(), f"Expected data file not found: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_v6_1_root_exists_and_manifest_is_present() -> None:
    assert V6_1_ROOT.exists()
    manifest = _read_json("manifest.json")
    assert manifest["version"] == "v6.1"
    assert manifest["generation_policy"] == "strict_no_generation"
    assert manifest["turn_count_policy"] == "exact_count_with_manual_review"
    assert manifest["provenance_taxonomy"] == [
        "direct_source",
        "retrieved_composed",
        "source_anchored_deterministic_edit",
    ]
    # Manifest uses "name" key for file entries
    files = {entry["name"] for entry in manifest.get("files", [])}
    for rel in {
        "study_b_test.json",
        "study_b_multi_turn_test.json",
        "study_c_test.json",
        "adversarial_bias/biased_vignettes.json",
    }:
        assert rel in files, f"Missing file in manifest: {rel}"


@pytest.mark.unit
def test_v6_1_builder_module_does_not_import_generation_backends() -> None:
    text = STRICT_BUILDER.read_text(encoding="utf-8")
    assert "reliable_clinical_benchmark.models" not in text
    assert "ModelFactory" not in text


@pytest.mark.unit
def test_v6_1_study_a_remains_direct_source() -> None:
    study_a = _read_json("study_a_test.json").get("samples", [])
    assert len(study_a) == 2000
    for sample in study_a[:100]:
        metadata = sample.get("metadata") or {}
        assert metadata.get("source_type") == "direct_source"


@pytest.mark.unit
def test_v6_1_bias_rows_use_strict_taxonomy() -> None:
    cases = _read_json("adversarial_bias/biased_vignettes.json").get("cases", [])
    assert len(cases) == 2000
    groups: dict[str, list[str]] = {}
    for case in cases:
        metadata = case.get("metadata") or {}
        assert metadata.get("source_type") == "source_anchored_deterministic_edit", (
            f"Case {case.get('id')} source_type={metadata.get('source_type')}"
        )
        assert metadata.get("manual_review_required") is False
        groups.setdefault(str(case.get("pair_group_id")), []).append(str(case.get("prompt") or ""))
    assert all(len(prompts) == 2 for prompts in groups.values()), "Not all pair groups have exactly 2 members"


@pytest.mark.unit
def test_v6_1_study_b_single_turn_direct_source() -> None:
    items = _read_json("study_b_test.json")
    assert isinstance(items, list)
    assert len(items) == 2000
    for item in items:
        metadata = item.get("metadata") or {}
        assert metadata.get("source_type") == "direct_source", (
            f"Row {item.get('id')} source_type={metadata.get('source_type')}"
        )


@pytest.mark.unit
def test_v6_1_multi_turn_cases_preserve_full_turn_scaffolds() -> None:
    for relative_path, expected_n in (
        ("study_b_multi_turn_test.json", 120),
        ("study_c_test.json", 100),
    ):
        payload = _read_json(relative_path)
        cases = payload if isinstance(payload, list) else payload.get("cases", [])
        assert len(cases) == expected_n, f"{relative_path}: expected {expected_n} cases, got {len(cases)}"
        for case in cases:
            turns = case.get("turns") or []
            assert len(turns) == 20, f"Case {case.get('id')} has {len(turns)} turns"
            for expected_turn, turn in enumerate(turns, start=1):
                assert turn.get("turn") == expected_turn
                assert turn.get("phase"), f"Case {case.get('id')} turn {expected_turn} missing phase"
                assert turn.get("dialogue_act"), f"Case {case.get('id')} turn {expected_turn} missing dialogue_act"
                assert isinstance(turn.get("state_ledger"), dict), (
                    f"Case {case.get('id')} turn {expected_turn} missing state_ledger"
                )
                assert turn.get("provenance_type") in _VALID_PROVENANCE_TYPES, (
                    f"Case {case.get('id')} turn {expected_turn} invalid provenance: {turn.get('provenance_type')}"
                )
