"""
Pairwise evaluation configuration and manifest loading.

The pairwise layer is notebook-first and manifest-driven: a run config points
to a frozen case manifest and a fixed four-judge manifest, then resolves a
rubric family into the concrete criteria used by the runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, validator


PROHIBITED_CRITERIA = frozenset({
    "clinical_correctness",
    "crisis_safety",
    "hallucination_truthfulness",
    "risk_classification",
    "diagnosis_quality",
})

VALID_LAYERS = frozenset({
    "core",
    "controllability",
    "invariance",
})

VALID_SLICE_IDS = frozenset({
    "study_a",
    "study_a_bias",
    "study_b",
    "study_b_multiturn",
    "study_c",
    "study_a_controllability",
    "study_a_bias_controllability",
    "study_b_controllability",
    "study_b_multiturn_controllability",
    "study_c_controllability",
    "invariance",
    "invariance_under_control",
    "control_under_invariance",
})


class GenerationParams(BaseModel):
    """Generation parameters passed to an LM Studio judge."""

    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.95

    class Config:
        frozen = True


class JudgeManifestEntry(BaseModel):
    """One fixed judge entry in the canonical panel."""

    judge_id: str
    display_name: str
    hf_source: str
    local_model_id: str
    generation_params: GenerationParams = GenerationParams()

    class Config:
        frozen = True


class JudgeManifest(BaseModel):
    """Fixed judge panel metadata."""

    manifest_version: str = "pairwise.judges.v1"
    judges: List[JudgeManifestEntry]

    class Config:
        frozen = True

    @validator("judges")
    def _require_exactly_four_judges(
        cls, value: List[JudgeManifestEntry]
    ) -> List[JudgeManifestEntry]:
        if len(value) != 4:
            raise ValueError(
                f"judge manifest must define exactly 4 judges; got {len(value)}"
            )
        ids = [judge.judge_id for judge in value]
        if len(ids) != len(set(ids)):
            raise ValueError("judge manifest contains duplicate judge_id values")
        return value


class PairwiseConfig(BaseModel):
    """One runnable pairwise configuration."""

    run_id: str
    layer: str
    slice_id: str
    case_manifest_path: str
    judge_manifest_path: str
    rubric_family: str
    allow_ties: bool = True
    orders: List[str] = ["AB", "BA"]
    max_retries_per_invalid_parse: int = 3
    output_root: str = "metric-results/pairwise"
    pooled_requires_all_judges: bool = True

    class Config:
        frozen = True

    @validator("layer")
    def _layer_must_be_known(cls, value: str) -> str:
        if value not in VALID_LAYERS:
            raise ValueError(f"unknown layer: {value}")
        return value

    @validator("slice_id")
    def _slice_id_must_be_known(cls, value: str) -> str:
        if value not in VALID_SLICE_IDS:
            raise ValueError(f"unknown slice_id: {value}")
        return value

    @validator("orders")
    def _orders_must_be_exact(cls, value: List[str]) -> List[str]:
        if set(value) != {"AB", "BA"}:
            raise ValueError("orders must contain exactly AB and BA")
        return value

class PairwiseRunSpec(BaseModel):
    """Resolved runtime spec combining config, judges, and criteria."""

    config: PairwiseConfig
    judge_manifest: JudgeManifest
    criteria: List[str]

    class Config:
        frozen = True

    @validator("criteria")
    def _criteria_not_empty(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("resolved criteria must not be empty")
        violations = PROHIBITED_CRITERIA.intersection(value)
        if violations:
            raise ValueError(
                f"resolved criteria include prohibited fields: {sorted(violations)}"
            )
        return value


def load_pairwise_run_spec(config_path: str | Path) -> PairwiseRunSpec:
    """Load a run config, its judge manifest, and rubric criteria."""

    config_file = Path(config_path).resolve()
    config_payload = json.loads(config_file.read_text(encoding="utf-8"))
    for field in ("case_manifest_path", "judge_manifest_path", "output_root"):
        raw_path = config_payload.get(field)
        if raw_path and not Path(raw_path).is_absolute():
            config_payload[field] = str((config_file.parent / raw_path).resolve())
    config = PairwiseConfig.parse_obj(config_payload)

    for field in ("case_manifest_path", "judge_manifest_path"):
        path = Path(getattr(config, field))
        if not path.exists():
            raise FileNotFoundError(f"{field} does not exist: {path}")

    judge_payload = json.loads(
        Path(config.judge_manifest_path).read_text(encoding="utf-8")
    )
    judge_manifest = JudgeManifest.parse_obj(judge_payload)

    # Import lazily to avoid a config/rubric import cycle.
    from .rubric import criteria_for_family

    criteria = criteria_for_family(config.rubric_family)

    if config.pooled_requires_all_judges and len(judge_manifest.judges) != 4:
        raise ValueError(
            "pooled mode requires all four canonical judges to be present"
        )

    return PairwiseRunSpec(
        config=config,
        judge_manifest=judge_manifest,
        criteria=criteria,
    )
