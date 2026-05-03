#!/usr/bin/env python3
"""Write local pairwise configs for the currently loaded LM Studio judges."""

from __future__ import annotations

import json
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent.parent.parent
PAIRWISE_ROOT = RUNTIME_ROOT / "metric-results" / "pairwise" / "manifests"
CONFIG_ROOT = PAIRWISE_ROOT / "configs"

QWEN_MODEL_ID = "qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2"
GEMMA_MODEL_ID = "gemma-4-31b-it-claude-opus-distill-v2"

LOCAL_JUDGE_MANIFEST = {
    "manifest_version": "pairwise.judges.v2",
    "judges": [
        {
            "judge_id": "local_qwen35_primary",
            "display_name": "Local Qwen 3.5 27B Claude 4.6 Opus Distill v2",
            "hf_source": "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF",
            "local_model_id": QWEN_MODEL_ID,
            "role": "primary",
            "generation_params": {"temperature": 0.1, "top_p": 0.95},
        },
        {
            "judge_id": "local_gemma4_audit",
            "display_name": "Local Gemma 4 31B IT Claude Opus Distill v2",
            "hf_source": "TeichAI/gemma-4-31B-it-Claude-Opus-Distill-GGUF",
            "local_model_id": GEMMA_MODEL_ID,
            "role": "audit",
            "generation_params": {"temperature": 0.1, "top_p": 0.95},
        },
        {
            "judge_id": "local_qwen35_escalation",
            "display_name": "Local Qwen 3.5 27B Claude 4.6 Opus Distill v2 Escalation",
            "hf_source": "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF",
            "local_model_id": QWEN_MODEL_ID,
            "role": "escalation",
            "escalation_rank": 1,
            "generation_params": {"temperature": 0.1, "top_p": 0.95},
        },
        {
            "judge_id": "local_gemma4_escalation",
            "display_name": "Local Gemma 4 31B IT Claude Opus Distill v2 Escalation",
            "hf_source": "TeichAI/gemma-4-31B-it-Claude-Opus-Distill-GGUF",
            "local_model_id": GEMMA_MODEL_ID,
            "role": "escalation",
            "escalation_rank": 2,
            "generation_params": {"temperature": 0.1, "top_p": 0.95},
        },
    ],
}


def main() -> int:
    PAIRWISE_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)

    judge_manifest_path = PAIRWISE_ROOT / "judge_panel.local.json"
    judge_manifest_path.write_text(
        json.dumps(LOCAL_JUDGE_MANIFEST, indent=2),
        encoding="utf-8",
    )

    written_configs = []
    for config_path in sorted(CONFIG_ROOT.glob("*.run_config.json")):
        if ".local." in config_path.name:
            continue
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["run_id"] = f"{payload['run_id']}_local"
        payload["judge_manifest_path"] = "../judge_panel.local.json"
        target = CONFIG_ROOT / config_path.name.replace(".run_config.json", ".local.run_config.json")
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written_configs.append(str(target))

    print(
        json.dumps(
            {
                "judge_manifest": str(judge_manifest_path),
                "configs_written": len(written_configs),
                "qwen_model_id": QWEN_MODEL_ID,
                "gemma_model_id": GEMMA_MODEL_ID,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
