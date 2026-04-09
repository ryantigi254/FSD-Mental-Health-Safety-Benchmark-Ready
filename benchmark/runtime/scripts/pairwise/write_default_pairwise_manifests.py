#!/usr/bin/env python3
"""Write the canonical pairwise judge manifests, configs, and audit stubs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from reliable_clinical_benchmark.pairwise.manifest_builder import build_case_manifest  # noqa: E402


PAIRWISE_ROOT = RUNTIME_ROOT / "metric-results" / "pairwise" / "manifests"
CONFIG_ROOT = PAIRWISE_ROOT / "configs"
JUDGE_AUDIT_ROOT = RUNTIME_ROOT / "metric-results" / "pairwise" / "judge_audit" / "manifests"

JUDGE_MANIFEST_V1 = {
    "manifest_version": "pairwise.judges.v1",
    "judges": [
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
        {
            "judge_id": "gemma4_31b_it",
            "display_name": "Gemma 4 31B IT",
            "hf_source": "google/gemma-4-31B-it",
            "local_model_id": "google/gemma-4-31B-it",
            "generation_params": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
        },
    ],
}

JUDGE_MANIFEST_V2 = {
    "manifest_version": "pairwise.judges.v2",
    "judges": [
        {
            "judge_id": "jackrong_qwopus35_27b_v3",
            "display_name": "Qwopus 3.5 27B v3",
            "hf_source": "Jackrong/Qwopus3.5-27B-v3-GGUF",
            "local_model_id": "Jackrong/Qwopus3.5-27B-v3-GGUF",
            "role": "primary",
            "generation_params": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
        },
        {
            "judge_id": "teichai_glm47_flash_opus45",
            "display_name": "GLM 4.7 Flash Claude Opus 4.5 Distill",
            "hf_source": "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill",
            "local_model_id": "TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill",
            "role": "audit",
            "generation_params": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
        },
        {
            "judge_id": "teichai_qwen3_14b_gpt52",
            "display_name": "Qwen 3 14B GPT 5.2 Distill",
            "hf_source": "TeichAI/Qwen3-14B-GPT-5.2-High-Reasoning-Distill-GGUF",
            "local_model_id": "TeichAI/Qwen3-14B-GPT-5.2-High-Reasoning-Distill-GGUF",
            "role": "escalation",
            "escalation_rank": 1,
            "generation_params": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
        },
        {
            "judge_id": "gemma4_31b_it",
            "display_name": "Gemma 4 31B IT",
            "hf_source": "google/gemma-4-31B-it",
            "local_model_id": "google/gemma-4-31B-it",
            "role": "escalation",
            "escalation_rank": 2,
            "generation_params": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95},
        },
    ],
}

JUDGE_AUDIT_MANIFEST = {
    "manifest_version": "pairwise.judge_audit.v1",
    "items": [],
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
    JUDGE_AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    judge_manifest_v1_path = PAIRWISE_ROOT / "judge_panel.v1.json"
    judge_manifest_v2_path = PAIRWISE_ROOT / "judge_panel.v2.json"
    judge_audit_manifest_path = JUDGE_AUDIT_ROOT / "gold_audit_slice.v1.json"

    judge_manifest_v1_path.write_text(json.dumps(JUDGE_MANIFEST_V1, indent=2), encoding="utf-8")
    judge_manifest_v2_path.write_text(json.dumps(JUDGE_MANIFEST_V2, indent=2), encoding="utf-8")
    if not judge_audit_manifest_path.exists():
        judge_audit_manifest_path.write_text(
            json.dumps(JUDGE_AUDIT_MANIFEST, indent=2),
            encoding="utf-8",
        )

    built_manifests = []
    skipped_manifests = []
    for slice_id, _layer, _rubric_family in RUN_SPECS:
        manifest_path = PAIRWISE_ROOT / f"{slice_id}_case_manifest.json"
        should_build = True
        if manifest_path.exists():
            try:
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if existing_manifest.get("status") == "ready":
                    should_build = False
            except json.JSONDecodeError:
                should_build = True
        if should_build:
            build_case_manifest(
                slice_id=slice_id,
                runtime_root=RUNTIME_ROOT,
                output_path=manifest_path,
            )
            built_manifests.append(str(manifest_path))
        else:
            skipped_manifests.append(str(manifest_path))

    for slice_id, layer, rubric_family in RUN_SPECS:
        payload = {
            "run_id": f"pairwise_{slice_id}_v2",
            "layer": layer,
            "slice_id": slice_id,
            "case_manifest_path": f"../{slice_id}_case_manifest.json",
            "judge_manifest_path": "../judge_panel.v2.json",
            "rubric_family": rubric_family,
            "run_mode": "stacked",
            "allow_ties": True,
            "orders": ["AB", "BA"],
            "max_retries_per_invalid_parse": 3,
            "output_root": "../../",
            "high_risk_tags": [
                "high_risk",
                "crisis",
                "crisis_adjacent",
                "safety_critical",
                "method_fit",
                "multi_turn",
            ],
            "escalate_on": [
                "disagreement",
                "tie",
                "invalid",
                "swap_failure",
                "high_risk",
            ],
            "persistent_disagreement_policy": "mark_uncertain",
            "pooled_requires_all_judges": True,
        }
        target = CONFIG_ROOT / f"{slice_id}.run_config.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "judge_manifest_v1": str(judge_manifest_v1_path),
                "judge_manifest_v2": str(judge_manifest_v2_path),
                "judge_audit_manifest": str(judge_audit_manifest_path),
                "case_manifests_written": len(built_manifests),
                "case_manifests_skipped": len(skipped_manifests),
                "configs_written": len(RUN_SPECS),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
