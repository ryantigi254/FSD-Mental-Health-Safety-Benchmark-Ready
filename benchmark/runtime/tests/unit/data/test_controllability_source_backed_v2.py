"""Regression checks for the source-backed controllability `_v2` roots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
SMALL_ROOT = BASE_DIR / "data" / "controllability" / "misc" / "controllability_splits_v2"
LARGE_ROOT = BASE_DIR / "data" / "controllability" / "misc" / "controllability_splits_large_resolved_v2"
VERIFY_ROOT = BASE_DIR / "data" / "verification"


def _read_json(path: Path):
    assert path.exists(), f"Missing file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _small_cases(filename: str, key: str = "cases"):
    payload = _read_json(SMALL_ROOT / filename)
    if key is None or not isinstance(payload, dict):
        return payload
    return payload.get(key, payload)


def _large_cases(filename: str, key: str = "cases"):
    payload = _read_json(LARGE_ROOT / filename)
    if key is None or not isinstance(payload, dict):
        return payload
    return payload.get(key, payload)


@pytest.mark.unit
def test_source_backed_controllability_v2_counts_match_expected() -> None:
    assert len(_small_cases("study_a_controllability_test.json", key="samples")) == 300
    assert len(_small_cases("study_a_bias_controllability_test.json")) == 300
    assert len(_small_cases("study_b_controllability_test.json", key=None)) == 300
    assert len(_small_cases("study_b_multi_turn_controllability_test.json", key=None)) == 30
    assert len(_small_cases("study_c_controllability_test.json")) == 30

    assert len(_large_cases("study_a_controllability_test.json", key="samples")) == 1700
    assert len(_large_cases("study_a_bias_controllability_test.json")) == 1700
    assert len(_large_cases("study_b_controllability_test.json", key=None)) == 1700
    assert len(_large_cases("study_b_multi_turn_controllability_test.json", key=None)) == 90
    assert len(_large_cases("study_c_controllability_test.json")) == 70


@pytest.mark.unit
def test_source_backed_controllability_v2_bias_metadata_is_resolved() -> None:
    for root in (SMALL_ROOT, LARGE_ROOT):
        cases = _read_json(root / "study_a_bias_controllability_test.json").get("cases", [])
        assert cases
        for case in cases:
            metadata = case.get("metadata") or {}
            assert metadata.get("source_openr1_split") in {"test", "train"}
            assert metadata.get("source_openr1_id") in metadata.get("source_openr1_ids", [])
            assert metadata.get("inferred_condition")
            assert str(metadata.get("inferred_condition")).strip().lower() != "unresolved"
            assert metadata.get("condition_resolution_source")


@pytest.mark.unit
def test_source_backed_controllability_v2_gold_meta_declares_parent_provenance() -> None:
    for root, expected_cases in ((SMALL_ROOT, 30), (LARGE_ROOT, 70)):
        labels_meta = _read_json(root / "ctrl_gold_diagnosis_labels.json").get("meta", {})
        plans_meta = _read_json(root / "ctrl_target_plans.json").get("meta", {})

        assert labels_meta.get("backend") == "probe"
        assert labels_meta.get("probe_meta", {}).get("mode") == "direct_parent"
        assert labels_meta.get("probe_meta", {}).get("parent_root") == "benchmark/runtime/data/frozen_splits/v6"

        assert plans_meta.get("backend") == "probe"
        assert plans_meta.get("probe_meta", {}).get("mode") == "direct_parent"
        assert plans_meta.get("probe_meta", {}).get("parent_root") == "benchmark/runtime/data/frozen_splits/v6"
        assert int(plans_meta.get("n_cases", -1)) == expected_cases


@pytest.mark.unit
def test_source_backed_controllability_large_stage2_report_is_ready() -> None:
    payload = _read_json(
        VERIFY_ROOT / "controllability_v2_large_resolved" / "stage2_gate_report.json"
    )
    assert payload["overall_passed"] is True
    assert payload["release_status"] == "ready"
    failed = [gate for gate in payload.get("gates", []) if not gate.get("passed")]
    assert not failed
