"""Pipeline orchestrator — plan -> retrieve -> edit -> validate -> regenerate."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .anchor import extract_anchors
from .edit_plan import plan_edit, EditPlan
from .generation import generate_candidates, EditCandidate
from .provenance import ProvenanceRecord, ProvenanceType, build_provenance
from .validation import validate_candidate, ValidationVerdict

if TYPE_CHECKING:
    from ..models.base import ModelRunner
    from ..utils.ner import MedicalNER
    from ..utils.nli import NLIModel
    from .config import PipelineConfig
    from .retrieval import RetrievalIndex

logger = logging.getLogger(__name__)


def run_pipeline(
    source_row: Dict[str, Any],
    operator: str,
    config: "PipelineConfig",
    model: "ModelRunner",
    ner: "MedicalNER",
    nli_model: "NLIModel",
    *,
    source_text_key: str = "prompt",
    source_id_key: str = "id",
    retrieval_index: Optional["RetrievalIndex"] = None,
    prev_turn_text: Optional[str] = None,
    multi_turn: bool = False,
    # Operator-specific kwargs forwarded to plan_edit
    insertion_angle: str = "",
    bias_feature: str = "",
    bias_label: str = "",
    pressure_type: str = "",
    escalation_level: int = 0,
    previous_context: str = "",
) -> Tuple[Optional[Dict[str, Any]], ProvenanceRecord]:
    """Run the full hardening pipeline on a single source row.

    Returns
    -------
    (output_row, provenance)
        *output_row* is ``None`` if no candidate passes validation after
        ``config.max_regen_attempts`` cycles (fail closed).
    """
    source_text = str(source_row.get(source_text_key, ""))
    source_id = str(source_row.get(source_id_key, ""))
    source_openr1_ids = _extract_source_ids(source_row)

    # 1. Extract anchors
    anchor = extract_anchors(source_text, ner, config)

    # 2. Retrieve support rows (optional)
    retrieval_hits = []
    if retrieval_index is not None:
        retrieval_hits = retrieval_index.query(
            source_text,
            anchor.entities,
            exclude_ids={source_id},
        )

    # 3. Plan edit
    plan = plan_edit(
        source_text,
        anchor,
        operator,
        config,
        source_row_id=source_id,
        insertion_angle=insertion_angle,
        bias_feature=bias_feature,
        bias_label=bias_label,
        pressure_type=pressure_type,
        escalation_level=escalation_level,
        retrieval_hits=retrieval_hits,
    )

    # 4. Generate + validate loop
    best_candidate: Optional[EditCandidate] = None
    best_verdict: Optional[ValidationVerdict] = None
    best_cosine: float = -1.0

    for attempt in range(config.max_regen_attempts):
        candidates = generate_candidates(
            source_text, plan, model, config,
            previous_context=previous_context,
        )

        for candidate in candidates:
            verdict = validate_candidate(
                source_text,
                candidate.text,
                anchor,
                plan,
                config,
                ner,
                nli_model,
                prev_turn_text=prev_turn_text,
                multi_turn=multi_turn,
            )

            if verdict.passed and verdict.cosine_similarity > best_cosine:
                best_candidate = candidate
                best_verdict = verdict
                best_cosine = verdict.cosine_similarity

        if best_candidate is not None:
            break

        logger.info(
            "Attempt %d/%d for %s: no passing candidate (best cosine %.4f)",
            attempt + 1, config.max_regen_attempts, source_id, best_cosine,
        )

    # 5. Build provenance
    if best_candidate is not None:
        prov_type = (
            ProvenanceType.RETRIEVED_COMPOSED
            if retrieval_hits
            else ProvenanceType.SOURCE_ANCHORED_GENERATED
        )
        provenance = build_provenance(
            source_openr1_ids,
            prov_type,
            plan=plan,
            verdict=best_verdict,
        )

        # Build output row — copy source metadata, replace text
        output_row = dict(source_row)
        output_row[source_text_key] = best_candidate.text

        return output_row, provenance

    # Fail closed
    logger.warning(
        "Row %s failed after %d attempts — dropping (fail closed)",
        source_id, config.max_regen_attempts,
    )
    provenance = build_provenance(
        source_openr1_ids,
        ProvenanceType.DIRECT_SOURCE,
        plan=plan,
    )
    return None, provenance


def _extract_source_ids(row: Dict[str, Any]) -> List[int]:
    """Extract source OpenR1 IDs from a row's metadata."""
    # Try various field locations
    meta = row.get("metadata", {})
    if isinstance(meta, dict):
        sid = meta.get("source_openr1_id")
        if sid is not None:
            return [int(sid)]
        sids = meta.get("source_openr1_ids")
        if isinstance(sids, list):
            return [int(s) for s in sids]
    # Top-level fallback
    sids = row.get("source_openr1_ids")
    if isinstance(sids, list):
        return [int(s) for s in sids]
    sid = row.get("source_openr1_id")
    if sid is not None:
        return [int(sid)]
    return []
