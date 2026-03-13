from __future__ import annotations

import sys
from pathlib import Path


DEFAULT_INVARIANCE_DATA_DIR = "data/frozen_splits/v5_invariance_samples"


def ensure_src_on_path(runtime_root: Path) -> None:
    src_dir = runtime_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def normalize_model_id_for_path(model_id: str, output_dir: Path) -> str:
    """Resolve model IDs to existing results folder names when possible."""

    alias_map = {
        "gpt_oss": "gpt-oss-20b",
        "deepseek_r1_lmstudio": "deepseek-r1-lmstudio",
        "piaget_local": "piaget-8b-local",
        "psych_qwen_local": "psych-qwen-32b-local",
        "psyllm_gml_local": "psyllm-gml-local",
        "psyche_r1_local": "psyche-r1-local",
        "qwen3_lmstudio": "qwen3-lmstudio",
        "ollama_minimax_m2_5_cloud": "minimax-m2.5-cloud",
        "minimax_m2_5_cloud": "minimax-m2.5-cloud",
        "psyllm_gml_vllm": "psyllm-gml-local",
        "piaget_vllm": "piaget-8b-local",
        "psyche_r1_vllm": "psyche-r1-local",
        "psych_qwen_vllm": "psych-qwen-32b-local",
    }
    alias_target = alias_map.get(model_id)
    if alias_target:
        alias_path = output_dir / alias_target
        if alias_path.exists() and alias_path.is_dir():
            return alias_target

    candidates = [
        model_id,
        model_id.replace("_", "-"),
        model_id.lower(),
        model_id.lower().replace("_", "-"),
        model_id.replace("-", "_"),
        model_id.lower().replace("-", "_"),
    ]
    for candidate in candidates:
        candidate_path = output_dir / candidate
        if candidate_path.exists() and candidate_path.is_dir():
            return candidate

    return model_id.replace("_", "-").lower()
