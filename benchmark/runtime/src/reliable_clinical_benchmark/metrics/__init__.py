"""Evaluation metrics for Studies A, B, and C."""

from .faithfulness import (
    calculate_faithfulness_gap,
    calculate_step_f1,
    calculate_silent_bias_rate,
    calculate_reasoning_adherence,
    check_reasoning_adherence,
    FaithfulnessResult,
)
from .sycophancy import (
    calculate_sycophancy_probability,
    calculate_flip_rate,
    evidence_hallucination_score,
    calculate_turn_of_flip,
    calculate_controlled_hallucination_rate,
    check_controlled_sycophancy,
    SycophancyResult,
)
from .drift import (
    compute_entity_recall_curve,
    calculate_knowledge_conflict_rate,
    calculate_alignment_score,
    calculate_controlled_entity_recall,
    check_controlled_entity_recall,
    DriftResult,
)
from .controllability import (
    calculate_compliance_rate,
    ControllabilityResult,
    STUDY_A_CONSTRAINT,
    STUDY_B_CONSTRAINT,
    STUDY_C_CONSTRAINT,
)

__all__ = [
    # Study A
    "calculate_faithfulness_gap",
    "calculate_step_f1",
    "calculate_silent_bias_rate",
    "calculate_reasoning_adherence",
    "check_reasoning_adherence",
    "FaithfulnessResult",
    # Study B
    "calculate_sycophancy_probability",
    "calculate_flip_rate",
    "evidence_hallucination_score",
    "calculate_turn_of_flip",
    "calculate_controlled_hallucination_rate",
    "check_controlled_sycophancy",
    "SycophancyResult",
    # Study C
    "compute_entity_recall_curve",
    "calculate_knowledge_conflict_rate",
    "calculate_alignment_score",
    "calculate_controlled_entity_recall",
    "check_controlled_entity_recall",
    "DriftResult",
    # Controllability (shared)
    "calculate_compliance_rate",
    "ControllabilityResult",
    "STUDY_A_CONSTRAINT",
    "STUDY_B_CONSTRAINT",
    "STUDY_C_CONSTRAINT",
]
