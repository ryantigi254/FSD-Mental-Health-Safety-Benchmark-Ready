"""Regression checks for source-backed invariance `_v2` roots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
V5_ROOT = BASE_DIR / "data" / "invariance" / "misc" / "v5_invariance_samples"
V5_ROOT_V2 = BASE_DIR / "data" / "invariance" / "misc" / "v5_invariance_samples_v2"
V6_PARENT = BASE_DIR / "data" / "frozen_splits" / "v6"

CTRL_BASE = BASE_DIR / "data" / "invariance" / "misc" / "invariance_variants" / "variant_family" / "base"
CTRL_BASE_V2 = BASE_DIR / "data" / "invariance" / "misc" / "invariance_variants" / "variant_family" / "base_v2"
CTRL_PARENT_V2 = BASE_DIR / "data" / "invariance" / "misc" / "controllability_splits_large_resolved_invariance_samples_v2"
CTRL_SAMPLE_V2 = BASE_DIR / "data" / "invariance" / "misc" / "controllability_splits_large_resolved_invariance_samples_v2"
VARIANT_FAMILY_V2 = BASE_DIR / "data" / "invariance" / "misc" / "invariance_variants" / "variant_family_v2"


def _read_json(path: Path):
    assert path.exists(), f"Missing file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _study_rows(root: Path, study: str):
    candidates = {
        "study_a": [root / "study_a_test.json", root / "study_a_controllability_test.json"],
        "study_a_bias": [
            root / "adversarial_bias" / "biased_vignettes.json",
            root / "study_a_bias_controllability_test.json",
        ],
        "study_b": [root / "study_b_test.json", root / "study_b_controllability_test.json"],
        "study_b_multi_turn": [
            root / "study_b_multi_turn_test.json",
            root / "study_b_multi_turn_controllability_test.json",
        ],
        "study_c": [root / "study_c_test.json", root / "study_c_controllability_test.json"],
    }
    payload = None
    for candidate in candidates[study]:
        if candidate.exists():
            payload = _read_json(candidate)
            break
    assert payload is not None, f"Unable to resolve dataset for {study} under {root}"
    if study in {"study_a", "study_a_bias", "study_c"} and isinstance(payload, dict):
        return payload.get("samples") or payload.get("cases") or payload
    return payload


def _ids(rows):
    return [str(row.get("id", "")).strip() for row in rows]


@pytest.mark.unit
def test_source_backed_invariance_v2_counts_match_expected() -> None:
    expected_v5 = {
        "study_a": 150,
        "study_a_bias": 150,
        "study_b": 160,
        "study_b_multi_turn": 12,
        "study_c": 15,
    }
    expected_ctrl = {
        "study_a": 140,
        "study_a_bias": 140,
        "study_b": 150,
        "study_b_multi_turn": 10,
        "study_c": 12,
    }
    for study, count in expected_v5.items():
        assert len(_study_rows(V5_ROOT_V2, study)) == count
    for study, count in expected_ctrl.items():
        assert len(_study_rows(CTRL_SAMPLE_V2, study)) == count
        assert len(_study_rows(CTRL_BASE_V2, study)) == count


@pytest.mark.unit
def test_source_backed_invariance_v2_keeps_unchanged_study_ids() -> None:
    for study in ("study_a", "study_a_bias"):
        assert _ids(_study_rows(V5_ROOT, study)) == _ids(_study_rows(V5_ROOT_V2, study))
        assert _ids(_study_rows(CTRL_BASE, study)) == _ids(_study_rows(CTRL_BASE_V2, study))


@pytest.mark.unit
def test_source_backed_invariance_v2_regenerated_ids_come_from_corrected_parents() -> None:
    for study in ("study_b", "study_b_multi_turn", "study_c"):
        parent_ids = set(_ids(_study_rows(V6_PARENT, study)))
        assert set(_ids(_study_rows(V5_ROOT_V2, study))).issubset(parent_ids)

        ctrl_parent_ids = set(_ids(_study_rows(CTRL_PARENT_V2, study)))
        assert set(_ids(_study_rows(CTRL_SAMPLE_V2, study))).issubset(ctrl_parent_ids)
        assert set(_ids(_study_rows(CTRL_BASE_V2, study))).issubset(ctrl_parent_ids)


@pytest.mark.unit
def test_variant_family_v2_manifest_points_at_new_base_root() -> None:
    manifest = _read_json(VARIANT_FAMILY_V2 / "variant_matrix_manifest.json")
    assert Path(manifest["base_root"]).name == "base_v2"
    assert Path(manifest["output_root"]).name == "variant_family_v2"
