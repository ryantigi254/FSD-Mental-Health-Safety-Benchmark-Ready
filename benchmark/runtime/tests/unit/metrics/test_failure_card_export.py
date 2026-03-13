from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    runtime_root = Path(__file__).resolve().parents[3]
    module_path = runtime_root / "scripts" / "evaluation" / "export_failure_cards.py"
    spec = importlib.util.spec_from_file_location("failure_cards_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_cards_groups_findings_by_model():
    module = _load_module()
    invariance_rows = [
        {
            "study": "study_a",
            "metric": "faithfulness_gap",
            "delta": 0.12,
            "ci_low": 0.11,
            "ci_high": 0.13,
            "variant_cache": "/tmp/results/qwen3-lmstudio/study_a_invariance.jsonl",
        }
    ]
    controllability_payloads = [
        {
            "study": "study_b",
            "base_cache": "/tmp/results/qwen3-lmstudio/study_b_generations.jsonl",
            "variants": [
                {
                    "tag": "strong",
                    "variant_type": "control",
                    "cache_path": "/tmp/results/qwen3-lmstudio/study_b_invariance_strong.jsonl",
                    "metrics": {
                        "sycophancy_probability": {
                            "delta_c": 0.12,
                            "ci_low": 0.08,
                            "ci_high": 0.14,
                        }
                    },
                }
            ],
        }
    ]

    cards = module._build_cards(invariance_rows, controllability_payloads)

    assert "qwen3-lmstudio" in cards
    assert len(cards["qwen3-lmstudio"]) == 2
    assert {entry["taxonomy"] for entry in cards["qwen3-lmstudio"]} == {
        "fundamental",
        "application-specific",
    }


def test_build_cards_prefers_base_cache_model_name():
    module = _load_module()
    invariance_rows = [
        {
            "study": "study_b",
            "metric": "sycophancy_probability",
            "delta": 0.12,
            "ci_low": 0.08,
            "ci_high": 0.14,
            "base_cache": "/tmp/results/qwen3-lmstudio/study_b_generations.jsonl",
            "variant_cache": "/tmp/elsewhere/study_b_variant.jsonl",
        }
    ]
    cards = module._build_cards(invariance_rows, [])
    assert "qwen3-lmstudio" in cards


def test_build_cards_uses_filename_heuristic_when_no_results_dir():
    module = _load_module()
    invariance_rows = [
        {
            "study": "study_c",
            "metric": "entity_recall_t10",
            "delta": -0.2,
            "ci_low": -0.3,
            "ci_high": -0.1,
            "base_cache": "/tmp/notebooks/invariance_smoke_outputs/study_c_base_qwen3_enriched.jsonl",
            "variant_cache": "/tmp/notebooks/invariance_smoke_outputs/study_c_variant_qwen3.jsonl",
        }
    ]
    cards = module._build_cards(invariance_rows, [])
    assert "qwen3-lmstudio" in cards
