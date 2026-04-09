from __future__ import annotations

import sys
from pathlib import Path
import re


DEFAULT_INVARIANCE_DATA_DIR = "data/frozen_splits/v5_invariance_samples"
CONTROLLABILITY_VARIANTS_ROOT = Path("data") / "invariance_variants" / "controllability"
CANONICAL_MODEL_DIRS = {
    "gpt_oss": "gpt-oss-20b",
    "gpt_oss_lmstudio": "gpt-oss-20b",
    "gpt-oss-lmstudio": "gpt-oss-20b",
    "gpt-oss-20b": "gpt-oss-20b",
    "deepseek_r1_lmstudio": "deepseek-r1-lmstudio",
    "deepseek-r1-lmstudio": "deepseek-r1-lmstudio",
    "piaget_local": "piaget-8b-local",
    "piaget_vllm": "piaget-8b-local",
    "psych_qwen_local": "psych-qwen-32b-local",
    "psych_qwen_vllm": "psych-qwen-32b-local",
    "psyllm_gml_local": "psyllm-gml-local",
    "psyllm_gml_vllm": "psyllm-gml-local",
    "psyllm_lmstudio": "psyllm-lmstudio",
    "psyche_r1_local": "psyche-r1-local",
    "psyche_r1_vllm": "psyche-r1-local",
    "qwen3_lmstudio": "qwen3-lmstudio",
    "qwen3-lmstudio": "qwen3-lmstudio",
    "qwen3-8b-lmstudio": "qwen3-lmstudio",
    "qwq": "qwq",
    "qwq_lmstudio": "qwq",
    "ollama_minimax_m2_5_cloud": "minimax-m2.5-cloud",
    "minimax_m2_5_cloud": "minimax-m2.5-cloud",
}


def ensure_src_on_path(runtime_root: Path) -> None:
    src_dir = runtime_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def resolve_output_dir(runtime_root: Path, output_dir: str | None, *, default_dir_name: str = "results") -> Path:
    if output_dir is None:
        return runtime_root / default_dir_name
    candidate = Path(output_dir)
    if candidate.is_absolute():
        return candidate
    return runtime_root / candidate


def resolve_invariance_output_dir(runtime_root: Path, output_dir: str | None, data_dir: Path) -> Path:
    resolved_output_dir = resolve_output_dir(runtime_root, output_dir, default_dir_name="results_invariance")
    if resolved_output_dir.name in {"base", "variant-family"}:
        return resolved_output_dir

    data_dir_path = data_dir if data_dir.is_absolute() else (runtime_root / data_dir)
    data_dir_path = data_dir_path.resolve()
    controllability_root = (runtime_root / CONTROLLABILITY_VARIANTS_ROOT).resolve()
    base_root = (controllability_root / "base").resolve()

    if data_dir_path == base_root or base_root in data_dir_path.parents:
        return resolved_output_dir / "base"
    if data_dir_path == controllability_root or controllability_root in data_dir_path.parents:
        return resolved_output_dir / "variant-family"
    return resolved_output_dir


def normalize_model_id_for_path(model_id: str, output_dir: Path) -> str:
    """Resolve model IDs to existing results folder names when possible."""

    alias_target = CANONICAL_MODEL_DIRS.get(model_id.lower())
    if alias_target:
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


def normalize_variant_tag(variant_tag: str | None) -> str | None:
    if not variant_tag:
        return None
    slug = re.sub(r"[^a-z0-9]+", "_", variant_tag.strip().lower()).strip("_")
    return slug or None


def default_invariance_cache_path(
    *,
    output_dir: Path,
    model_id: str,
    study_slug: str,
    variant_tag: str | None = None,
) -> Path:
    normalized_variant = normalize_variant_tag(variant_tag)
    filename = f"{study_slug}_generations.jsonl"
    if normalized_variant:
        filename = f"{study_slug}_{normalized_variant}_generations.jsonl"
    return output_dir / model_id / filename
