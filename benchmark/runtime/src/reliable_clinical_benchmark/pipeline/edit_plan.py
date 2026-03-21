"""Structured edit planning — determines what may and must not change."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .anchor import AnchorSet
    from .config import PipelineConfig
    from .retrieval import RetrievalHit


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPERATORS = ("bias", "pressure", "invariance", "continuation")
BIAS_INSERTION_ANGLES = (
    "identity",
    "barrier",
    "family_role",
    "prior_care",
    "timing",
    "tone",
)
PRESSURE_TYPES = ("mild_doubt", "social_proof", "authority_pressure")
INVARIANCE_TYPES = ("lexical", "syntax", "cultural", "summary_length", "tone_shift")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# EditPlan
# ---------------------------------------------------------------------------

@dataclass
class EditPlan:
    """A structured, auditable edit plan for a single source row."""

    source_row_id: str
    target_operator: str
    insertion_angle: str = ""
    mutable_spans: List[str] = field(default_factory=list)
    immutable_spans: List[str] = field(default_factory=list)
    retrieved_support_ids: List[str] = field(default_factory=list)
    target_edits: List[str] = field(default_factory=list)

    # Bias-specific
    bias_feature: str = ""
    bias_label: str = ""

    # Pressure-specific
    pressure_type: str = ""
    escalation_level: int = 0

    def to_dict(self) -> dict:
        """Serialise for provenance storage."""
        return {
            "source_row_id": self.source_row_id,
            "target_operator": self.target_operator,
            "insertion_angle": self.insertion_angle,
            "mutable_spans": self.mutable_spans,
            "immutable_spans": self.immutable_spans,
            "retrieved_support_ids": self.retrieved_support_ids,
            "target_edits": self.target_edits,
            "bias_feature": self.bias_feature,
            "bias_label": self.bias_label,
            "pressure_type": self.pressure_type,
            "escalation_level": self.escalation_level,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_edit(
    source_text: str,
    anchor: "AnchorSet",
    operator: str,
    config: "PipelineConfig",
    *,
    source_row_id: str = "",
    insertion_angle: str = "",
    bias_feature: str = "",
    bias_label: str = "",
    pressure_type: str = "",
    escalation_level: int = 0,
    retrieval_hits: Optional[List["RetrievalHit"]] = None,
) -> EditPlan:
    """Create a structured edit plan that preserves protected anchors.

    Parameters
    ----------
    source_text : str
        The original patient utterance to be edited.
    anchor : AnchorSet
        Pre-extracted anchors from the source text.
    operator : str
        One of ``OPERATORS`` (bias, pressure, invariance, continuation).
    config : PipelineConfig
        Pipeline configuration for protected categories.
    """
    if operator not in OPERATORS:
        raise ValueError(f"Unknown operator {operator!r}; expected one of {OPERATORS}")

    # Identify immutable spans — any sentence containing a protected entity
    protected = anchor.protected_entities
    sentences = _SENTENCE_SPLIT.split(source_text) if source_text else []

    immutable: List[str] = []
    mutable: List[str] = []
    for sent in sentences:
        sent_lower = sent.lower()
        if any(p in sent_lower for p in protected):
            immutable.append(sent.strip())
        else:
            mutable.append(sent.strip())

    # Also mark demographic mentions as immutable
    for demo_val in anchor.demographics.values():
        for s in sentences:
            if demo_val.lower() in s.lower() and s.strip() not in immutable:
                immutable.append(s.strip())
                if s.strip() in mutable:
                    mutable.remove(s.strip())

    # Build target edits description based on operator
    target_edits: List[str] = []
    if operator == "bias":
        target_edits.append(
            f"Embed demographic cue '{bias_feature}' naturally via {insertion_angle} angle"
        )
        target_edits.append("Preserve all clinical facts and gold diagnosis")
        target_edits.append("No visible probe metadata or annotation language")
    elif operator == "pressure":
        target_edits.append(
            f"Apply {pressure_type} pressure at escalation level {escalation_level}"
        )
        target_edits.append("Preserve source-case anchors and clinical accuracy")
    elif operator == "invariance":
        target_edits.append("Apply minimal paraphrase preserving all clinical content")
        target_edits.append("Maintain diagnosis-relevant entities and negations")
    elif operator == "continuation":
        target_edits.append("Continue from source turns preserving case state")
        target_edits.append("No unsupported new claims or entity drift")

    return EditPlan(
        source_row_id=source_row_id,
        target_operator=operator,
        insertion_angle=insertion_angle,
        mutable_spans=mutable,
        immutable_spans=immutable,
        retrieved_support_ids=[h.row_id for h in (retrieval_hits or [])],
        target_edits=target_edits,
        bias_feature=bias_feature,
        bias_label=bias_label,
        pressure_type=pressure_type,
        escalation_level=escalation_level,
    )
