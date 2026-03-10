from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    runtime_root = Path(__file__).resolve().parents[3]
    module_path = runtime_root / "scripts" / "studies" / "v5_review" / "run_v5_refresh.py"
    spec = importlib.util.spec_from_file_location("run_v5_refresh_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refresh_study_b_single_turn_replaces_only_overlapping_real_rows(monkeypatch):
    module = _load_module()

    frozen_rows = [
        {
            "id": "b_001",
            "prompt": "old prompt 1",
            "gold_answer": "old answer 1",
            "incorrect_opinion": "old incorrect 1",
            "metadata": {"persona_id": "p1", "age": 30, "source": "openr1_test", "original_id": "10"},
        },
        {
            "id": "b_002",
            "prompt": "old prompt 2",
            "gold_answer": "old answer 2",
            "incorrect_opinion": "old incorrect 2",
            "metadata": {"persona_id": "p2", "age": 32, "source": "openr1_test", "original_id": "20"},
        },
        {
            "id": "b_003",
            "prompt": "synthetic",
            "gold_answer": "synthetic answer",
            "incorrect_opinion": "synthetic incorrect",
            "metadata": {"persona_id": "p3", "age": 28, "source": "synthetic", "original_id": "synthetic"},
        },
    ]

    monkeypatch.setattr(module, "_build_openr1_post_id_lookup", lambda: {("test", "10"): 1, ("test", "20"): 2})
    monkeypatch.setattr(module, "_study_b_bucket_for_persona", lambda persona_id: "major depressive disorder")

    grouped_candidates = {
        "major depressive disorder": [
            {
                "prompt": "replacement prompt",
                "gold_answer": "replacement answer",
                "incorrect_opinion": "replacement incorrect",
                "metadata": {
                    "source": "openr1_train",
                    "original_id": "500",
                    "source_split": "train",
                    "source_openr1_ids": [500],
                    "matched_condition": "major depressive disorder",
                    "source_type": "real",
                },
            }
        ]
    }

    refreshed, replacements = module.refresh_study_b_single_turn_against_study_a(
        frozen_study_b=frozen_rows,
        study_a_refs={("test", 1)},
        grouped_candidates=grouped_candidates,
    )

    assert len(replacements) == 1
    assert refreshed[0]["prompt"] == "replacement prompt"
    assert refreshed[1]["prompt"] == "old prompt 2"
    assert refreshed[2]["prompt"] == "synthetic"


def test_refresh_study_a_bias_replaces_only_overlapping_groups(monkeypatch):
    module = _load_module()
    frozen_cases = [
        {
            "id": "abias_0001",
            "pair_group_id": "g1",
            "prompt": "old 1a",
            "bias_feature": "f1",
            "bias_label": "l1",
            "metadata": {"dimension": "race", "dimension_family": "demographic", "persona_id": "p1", "persona_age": 25, "source_openr1_split": "test", "source_openr1_id": 1},
        },
        {
            "id": "abias_0002",
            "pair_group_id": "g1",
            "prompt": "old 1b",
            "bias_feature": "f2",
            "bias_label": "l2",
            "metadata": {"dimension": "race", "dimension_family": "demographic", "persona_id": "p1", "persona_age": 25, "source_openr1_split": "test", "source_openr1_id": 1},
        },
        {
            "id": "abias_0003",
            "pair_group_id": "g2",
            "prompt": "old 2a",
            "bias_feature": "f3",
            "bias_label": "l3",
            "metadata": {"dimension": "gender", "dimension_family": "demographic", "persona_id": "p2", "persona_age": 30, "source_openr1_split": "test", "source_openr1_id": 2},
        },
        {
            "id": "abias_0004",
            "pair_group_id": "g2",
            "prompt": "old 2b",
            "bias_feature": "f4",
            "bias_label": "l4",
            "metadata": {"dimension": "gender", "dimension_family": "demographic", "persona_id": "p2", "persona_age": 30, "source_openr1_split": "test", "source_openr1_id": 2},
        },
    ]

    monkeypatch.setattr(
        module,
        "_build_bias_seed_pool",
        lambda excluded_pairs: [
            {"prompt": "replacement prompt", "source_openr1_split": "train", "source_openr1_id": 10},
            {"prompt": "unused prompt", "source_openr1_split": "train", "source_openr1_id": 11},
        ],
    )

    refreshed, replacements = module.refresh_study_a_bias_against_refs(
        frozen_bias_cases=frozen_cases,
        reserved_refs={("test", 1)},
    )

    assert len(replacements) == 1
    refreshed_by_id = {row["id"]: row for row in refreshed}
    assert refreshed_by_id["abias_0001"]["metadata"]["source_openr1_id"] == 10
    assert refreshed_by_id["abias_0003"]["metadata"]["source_openr1_id"] == 2


def test_refresh_study_c_replaces_overlaps_and_internal_duplicates(monkeypatch):
    module = _load_module()
    frozen_cases = [
        {"id": "c_001", "metadata": {"source_split": "test", "source_openr1_ids": [1]}},
        {"id": "c_002", "metadata": {"source_split": "test", "source_openr1_ids": [2]}},
        {"id": "c_003", "metadata": {"source_split": "test", "source_openr1_ids": [2]}},
    ]
    frozen_plans = {
        "c_001": {"source_split": "test", "source_openr1_id": 1},
        "c_002": {"source_split": "test", "source_openr1_id": 2},
        "c_003": {"source_split": "test", "source_openr1_id": 2},
    }

    refs = iter([("train", 11), ("train", 12)])
    monkeypatch.setattr(module.build_splits, "_available_source_refs", lambda reserved: refs)

    refreshed_cases, refreshed_plans, replacements = module.refresh_study_c_against_refs(
        frozen_cases=frozen_cases,
        frozen_plans=frozen_plans,
        reserved_refs={("test", 1)},
    )

    assert len(replacements) == 2
    assert refreshed_cases[0]["metadata"]["source_openr1_ids"] == [11]
    assert refreshed_cases[1]["metadata"]["source_openr1_ids"] == [2]
    assert refreshed_cases[2]["metadata"]["source_openr1_ids"] == [12]
    assert refreshed_plans["c_003"]["source_openr1_id"] == 12
