from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_build_splits_module():
    runtime_root = Path(__file__).resolve().parents[3]
    module_path = runtime_root / "scripts" / "preprocessing" / "build_splits.py"
    spec = importlib.util.spec_from_file_location("build_splits_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _refs(items):
    out = set()
    for item in items:
        meta = item.get("metadata", {})
        split = meta.get("source_split") or meta.get("source_openr1_split")
        ids = meta.get("source_openr1_ids")
        if isinstance(ids, list):
            for source_id in ids:
                out.add((split, source_id))
        elif meta.get("source_openr1_id") is not None:
            out.add((split, meta["source_openr1_id"]))
    return out


def test_harmonize_cross_study_provenance_assigns_disjoint_refs(tmp_path, monkeypatch):
    build_splits = _load_build_splits_module()
    data_root = tmp_path / "data"

    _write_json(
        data_root / "openr1_psy_splits" / "study_a_test.json",
        {
            "samples": [
                {"id": "a_001", "metadata": {"source_openr1_ids": [1], "source_split": "test"}},
                {"id": "a_002", "metadata": {"source_openr1_ids": [2], "source_split": "test"}},
            ]
        },
    )
    _write_json(
        data_root / "openr1_psy_splits" / "study_b_test.json",
        [
            {
                "id": "b_001",
                "metadata": {"persona_id": "p1", "source": "openr1_test", "original_id": "3"},
            },
            {
                "id": "b_002",
                "metadata": {"persona_id": "p2", "source": "synthetic", "original_id": "synthetic"},
            },
        ],
    )
    _write_json(
        data_root / "openr1_psy_splits" / "study_b_multi_turn_test.json",
        [
            {"id": "bmt_001", "metadata": {"persona_id": "p1"}, "turns": []},
            {"id": "bmt_002", "metadata": {"persona_id": "p2"}, "turns": []},
        ],
    )
    _write_json(
        data_root / "openr1_psy_splits" / "study_c_test.json",
        {
            "cases": [
                {"id": "c_001", "metadata": {"persona_id": "c1", "source_openr1_ids": [1], "source_split": "test"}},
                {"id": "c_002", "metadata": {"persona_id": "c2", "source_openr1_ids": [3], "source_split": "test"}},
            ]
        },
    )
    _write_json(
        data_root / "adversarial_bias" / "biased_vignettes.json",
        {
            "cases": [
                {"id": "bias_001", "pair_group_id": "g1", "metadata": {"persona_id": "p1"}},
                {"id": "bias_002", "pair_group_id": "g1", "metadata": {"persona_id": "p1"}},
                {"id": "bias_003", "pair_group_id": "g2", "metadata": {"persona_id": "p2"}},
                {"id": "bias_004", "pair_group_id": "g2", "metadata": {"persona_id": "p2"}},
            ]
        },
    )

    monkeypatch.setattr(
        build_splits,
        "_load_openr1_source_pool",
        lambda: [
            ("test", 1),
            ("test", 2),
            ("test", 3),
            ("train", 4),
            ("train", 5),
            ("train", 6),
            ("train", 7),
            ("train", 8),
            ("train", 9),
        ],
    )

    stats = build_splits.harmonize_cross_study_provenance(data_root=data_root)
    assert stats["study_b_refs"] == 1
    assert stats["study_a_bias_refs"] == 2
    assert stats["study_b_multi_turn_refs"] == 2
    assert stats["study_c_refs"] == 2

    study_b = json.loads((data_root / "openr1_psy_splits" / "study_b_test.json").read_text())
    assert study_b[0]["metadata"]["source_openr1_ids"] == [3]
    assert study_b[0]["metadata"]["source_split"] == "test"
    assert study_b[1]["metadata"]["source_openr1_ids"] == []
    assert study_b[1]["metadata"]["source_split"] == "generated"

    bias_cases = json.loads((data_root / "adversarial_bias" / "biased_vignettes.json").read_text())["cases"]
    assert bias_cases[0]["metadata"]["source_openr1_id"] == bias_cases[1]["metadata"]["source_openr1_id"]
    assert bias_cases[2]["metadata"]["source_openr1_id"] == bias_cases[3]["metadata"]["source_openr1_id"]
    assert bias_cases[0]["metadata"]["source_openr1_id"] != bias_cases[2]["metadata"]["source_openr1_id"]

    study_a_refs = _refs(json.loads((data_root / "openr1_psy_splits" / "study_a_test.json").read_text())["samples"])
    study_b_refs = _refs(study_b)
    bias_refs = _refs(bias_cases)
    study_b_mt_refs = _refs(json.loads((data_root / "openr1_psy_splits" / "study_b_multi_turn_test.json").read_text()))
    study_c_refs = _refs(json.loads((data_root / "openr1_psy_splits" / "study_c_test.json").read_text())["cases"])

    assert study_a_refs.isdisjoint(study_b_refs)
    assert study_a_refs.isdisjoint(bias_refs)
    assert study_a_refs.isdisjoint(study_b_mt_refs)
    assert study_a_refs.isdisjoint(study_c_refs)
    assert study_b_refs.isdisjoint(bias_refs)
    assert study_b_refs.isdisjoint(study_b_mt_refs)
    assert study_b_refs.isdisjoint(study_c_refs)
    assert bias_refs.isdisjoint(study_b_mt_refs)
    assert bias_refs.isdisjoint(study_c_refs)
    assert study_b_mt_refs.isdisjoint(study_c_refs)
