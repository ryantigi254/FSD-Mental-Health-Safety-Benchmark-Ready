"""Provenance tracking — records source lineage and validation for every row."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .edit_plan import EditPlan
    from .validation import ValidationVerdict


class ProvenanceType(str, Enum):
    """Provenance taxonomy for every data row."""

    DIRECT_SOURCE = "direct_source"
    RETRIEVED_COMPOSED = "retrieved_composed"
    SOURCE_ANCHORED_GENERATED = "source_anchored_generated"


@dataclass
class ProvenanceRecord:
    """Full provenance for a single data row or turn."""

    source_openr1_ids: List[int] = field(default_factory=list)
    provenance_type: str = ProvenanceType.DIRECT_SOURCE.value
    edit_operator: Optional[str] = None
    edit_plan: Optional[Dict[str, Any]] = None
    retrieved_support_ids: List[str] = field(default_factory=list)
    validation_verdict: Optional[Dict[str, Any]] = None

    # Multi-turn extensions
    source_round_index: Optional[int] = None
    continuation_from_turn: Optional[int] = None


def build_provenance(
    source_ids: List[int],
    provenance_type: str | ProvenanceType,
    *,
    plan: Optional["EditPlan"] = None,
    verdict: Optional["ValidationVerdict"] = None,
    source_round_index: Optional[int] = None,
    continuation_from_turn: Optional[int] = None,
) -> ProvenanceRecord:
    """Construct a provenance record from pipeline outputs."""
    ptype = provenance_type.value if isinstance(provenance_type, ProvenanceType) else provenance_type

    return ProvenanceRecord(
        source_openr1_ids=list(source_ids),
        provenance_type=ptype,
        edit_operator=plan.target_operator if plan else None,
        edit_plan=plan.to_dict() if plan else None,
        retrieved_support_ids=list(plan.retrieved_support_ids) if plan else [],
        validation_verdict=verdict.to_dict() if verdict else None,
        source_round_index=source_round_index,
        continuation_from_turn=continuation_from_turn,
    )


def serialise_provenance(prov: ProvenanceRecord) -> Dict[str, Any]:
    """Convert a ProvenanceRecord to a JSON-serialisable dict."""
    out: Dict[str, Any] = {
        "source_openr1_ids": prov.source_openr1_ids,
        "provenance_type": prov.provenance_type,
    }
    if prov.edit_operator is not None:
        out["edit_operator"] = prov.edit_operator
    if prov.edit_plan is not None:
        out["edit_plan"] = prov.edit_plan
    if prov.retrieved_support_ids:
        out["retrieved_support_ids"] = prov.retrieved_support_ids
    if prov.validation_verdict is not None:
        out["validation"] = prov.validation_verdict
    if prov.source_round_index is not None:
        out["source_round_index"] = prov.source_round_index
    if prov.continuation_from_turn is not None:
        out["continuation_from_turn"] = prov.continuation_from_turn
    return out
