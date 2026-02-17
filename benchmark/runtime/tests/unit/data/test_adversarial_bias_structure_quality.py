"""Structure-quality checks for canonical vs legacy adversarial bias datasets."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data" / "adversarial_bias"

CANONICAL_PATH = DATA_DIR / "biased_vignettes.json"
LEGACY_PATH = DATA_DIR / "biased_vignettes_legacy_2016.json"
CATALOG_PATH = DATA_DIR / "dimension_catalog_v3_2.json"
PERSONA_PATH = BENCHMARK_DIR / "docs" / "personas" / "persona_registry_v2.json"

EXPECTED_LEGACY_COUNT = 2016
EXPECTED_CANONICAL_COUNT = 2000
EXPECTED_PERSONA_COUNT = 40
EXPECTED_PERSONA_CASES = 50
EXPECTED_PAIR_GROUPS = 1000
STRUCTURE_VERSION = "v3.2"


def _load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    assert isinstance(cases, list), f"Invalid cases list in: {path}"
    return cases


@pytest.mark.unit
def test_legacy_archive_exists_and_retains_original_count():
    assert LEGACY_PATH.exists(), f"Missing legacy archive: {LEGACY_PATH}"
    legacy_cases = _load_cases(LEGACY_PATH)
    assert len(legacy_cases) == EXPECTED_LEGACY_COUNT


@pytest.mark.unit
def test_canonical_count_and_uniqueness_contract():
    canonical_cases = _load_cases(CANONICAL_PATH)
    assert len(canonical_cases) == EXPECTED_CANONICAL_COUNT

    triples = {
        (
            str(c.get("prompt", "")).strip().lower(),
            str(c.get("bias_feature", "")).strip().lower(),
            str(c.get("bias_label", "")).strip().lower(),
        )
        for c in canonical_cases
    }
    assert len(triples) == EXPECTED_CANONICAL_COUNT


@pytest.mark.unit
def test_canonical_rows_have_v32_structure_fields_and_pair_integrity():
    canonical_cases = _load_cases(CANONICAL_PATH)

    variant_sum = 0
    signatures = {}
    pair_counts = Counter()

    for case in canonical_cases:
        assert case.get("structure_version") == STRUCTURE_VERSION
        assert isinstance(case.get("pair_group_id"), str) and case["pair_group_id"].strip()
        assert isinstance(case.get("template_signature"), str) and case["template_signature"].strip()
        assert isinstance(case.get("source_variant_count"), int)
        assert case["source_variant_count"] >= 1
        variant_sum += case["source_variant_count"]

        signature = case["template_signature"]
        signatures.setdefault(signature, set()).add(case["pair_group_id"])
        pair_counts[case["pair_group_id"]] += 1

    assert variant_sum == EXPECTED_CANONICAL_COUNT
    assert len(pair_counts) == EXPECTED_PAIR_GROUPS
    assert all(count == 2 for count in pair_counts.values())


@pytest.mark.unit
def test_persona_coverage_and_dimension_floors():
    canonical_cases = _load_cases(CANONICAL_PATH)
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    personas = json.loads(PERSONA_PATH.read_text(encoding="utf-8"))

    assert isinstance(personas, list)
    assert len(personas) == EXPECTED_PERSONA_COUNT

    persona_ids = {p["id"] for p in personas}
    persona_counts = Counter(c.get("metadata", {}).get("persona_id") for c in canonical_cases)

    assert set(persona_counts.keys()) == persona_ids
    assert all(persona_counts[pid] == EXPECTED_PERSONA_CASES for pid in persona_ids)

    dim_counts = Counter(c.get("metadata", {}).get("dimension") for c in canonical_cases)
    dims = catalog.get("dimensions", [])
    assert len(dims) == 44

    for dim in dims:
        name = dim["dimension"]
        floor = int(dim.get("minimum_cases", 0))
        assert dim_counts[name] >= floor, f"Dimension floor unmet for {name}: {dim_counts[name]} < {floor}"

    # Sanity check: case total should be consistent with floor sums.
    floor_total = sum(int(d.get("minimum_cases", 0)) for d in dims)
    assert floor_total <= EXPECTED_CANONICAL_COUNT


@pytest.mark.unit
def test_catalog_group_targets_are_feasible():
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    dims = catalog.get("dimensions", [])

    mins = [int(math.ceil(int(d.get("minimum_cases", 0)) / 2.0)) for d in dims]
    assert sum(mins) <= EXPECTED_PAIR_GROUPS
