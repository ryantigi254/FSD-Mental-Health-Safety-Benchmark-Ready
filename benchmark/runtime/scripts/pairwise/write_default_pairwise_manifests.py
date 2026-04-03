#!/usr/bin/env python3
"""
Write the canonical pairwise judge manifest and default run configs.
"""

from __future__ import annotations

import json
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent.parent.parent
PAIRWISE_ROOT = RUNTIME_ROOT / "metric-results" / "pairwise" / "manifests"
CONFIG_ROOT = PAIRWISE_ROOT / "configs"

JUDGE_MANIFEST = {
    "manifest_version": "pairwise.judges.v1",
    "judges": [
        {
            "judge_id": "gemma4_31b_it",
            "display_name": "Gemma 4 31B IT",
            "hf_source": "google/gemma-4-31B-it",
            "local_model_id": "google/gemma-4-31B-it",
            "generation_params": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
        },
        {
            "judge_id": "jackrong_qwopus35_27b_v3",
            "display_name": "Qwopus 3.5 27B v3",
            "hf_source": "Jackrong/Qwopus3.5-27B-v3-GGUF",
            "local_model_id": "Jackrong/Qwopus3.5-27B-v3-GGUF",
            "generation_params": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
        },
        {
            "judge_id": "teichai_glm47_flash_opus45",
            "display_name": "GLM 4.7 Flash Claude Opus 4.5 Distill",
            "hf_source": "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill",
            "local_model_id": "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill",
            "generation_params": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
        },
        {
            "judge_id": "teichai_qwen3_14b_gpt52",
            "display_name": "Qwen 3 14B GPT 5.2 Distill",
            "hf_source": "TeichAI/Qwen3-14B-GPT-5.2-High-Reasoning-Distill-GGUF",
            "local_model_id": "TeichAI/Qwen3-14B-GPT-5.2-High-Reasoning-Distill-GGUF",
            "generation_params": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
        },
    ],
}

RUN_SPECS = [
    ("study_a", "core", "core_communication"),
    ("study_a_bias", "core", "core_communication"),
    ("study_b", "core", "core_communication"),
    ("study_b_multiturn", "core", "stakeholder_tagged"),
    ("study_c", "core", "core_communication"),
    ("study_a_controllability", "controllability", "controllability"),
    ("study_a_bias_controllability", "controllability", "controllability"),
    ("study_b_controllability", "controllability", "controllability"),
    ("study_b_multiturn_controllability", "controllability", "controllability"),
    ("study_c_controllability", "controllability", "controllability"),
    ("invariance", "invariance", "invariance"),
    ("invariance_under_control", "invariance", "invariance"),
    ("control_under_invariance", "invariance", "invariance"),
]


def main() -> int:
    PAIRWISE_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    judge_manifest_path = PAIRWISE_ROOT / "judge_panel.v1.json"
    judge_manifest_path.write_text(json.dumps(JUDGE_MANIFEST, indent=2), encoding="utf-8")

    for slice_id, layer, rubric_family in RUN_SPECS:
        payload = {
            "run_id": f"pairwise_{slice_id}_v1",
            "layer": layer,
            "slice_id": slice_id,
            "case_manifest_path": f"../{slice_id}_case_manifest.json",
            "judge_manifest_path": "../judge_panel.v1.json",
            "rubric_family": rubric_family,
            "allow_ties": True,
            "orders": ["AB", "BA"],
            "max_retries_per_invalid_parse": 3,
            "output_root": "../../",
            "pooled_requires_all_judges": True,
        }
        target = CONFIG_ROOT / f"{slice_id}.run_config.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"judge_manifest": str(judge_manifest_path), "configs_written": len(RUN_SPECS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
