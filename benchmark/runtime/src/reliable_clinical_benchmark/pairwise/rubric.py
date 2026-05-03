"""
Frozen rubric families for pairwise evaluation.

Pairwise remains a secondary communication-quality layer. The rubric families
here explicitly avoid safety-truth and clinical-correctness decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .config import PROHIBITED_CRITERIA


@dataclass(frozen=True)
class CriterionDef:
    """Definition for one pairwise criterion."""

    id: str
    display_name: str
    rubric_prompt: str
    family: str
    layer: str
    required_tags: Optional[Set[str]] = None


CORE_CRITERIA: Dict[str, CriterionDef] = {
    "clarity": CriterionDef(
        id="clarity",
        display_name="Clarity and Comprehensibility",
        rubric_prompt=(
            "Which response is clearer, easier to follow, and more likely to be "
            "understood correctly by the user?"
        ),
        family="core_communication",
        layer="core",
    ),
    "validation_reflection_quality": CriterionDef(
        id="validation_reflection_quality",
        display_name="Validation and Reflection Quality",
        rubric_prompt=(
            "Which response better validates the user's experience and reflects "
            "their feelings without sounding formulaic?"
        ),
        family="core_communication",
        layer="core",
    ),
    "grounded_helpfulness": CriterionDef(
        id="grounded_helpfulness",
        display_name="Grounded Helpfulness",
        rubric_prompt=(
            "Which response is more helpfully grounded in the scenario, more "
            "practically useful, and less likely to drift into generic filler?"
        ),
        family="core_communication",
        layer="core",
    ),
    "respectful_tone": CriterionDef(
        id="respectful_tone",
        display_name="Respectful Tone",
        rubric_prompt=(
            "Which response maintains a more respectful, non-stigmatising, "
            "and non-judgemental tone?"
        ),
        family="core_communication",
        layer="core",
    ),
    "response_economy": CriterionDef(
        id="response_economy",
        display_name="Response Economy",
        rubric_prompt=(
            "Which response uses words more efficiently while still remaining "
            "helpful and understandable?"
        ),
        family="core_communication",
        layer="core",
    ),
    "boundary_safe_validation": CriterionDef(
        id="boundary_safe_validation",
        display_name="Boundary-Safe Validation",
        rubric_prompt=(
            "Which response better validates the user while preserving clear, "
            "appropriate support boundaries?"
        ),
        family="core_communication",
        layer="core",
    ),
}

STAKEHOLDER_CRITERIA: Dict[str, CriterionDef] = {
    "dbt_skill_appropriateness": CriterionDef(
        id="dbt_skill_appropriateness",
        display_name="DBT Skill Appropriateness",
        rubric_prompt=(
            "Which response uses a more DBT-consistent and situation-appropriate "
            "supportive skill?"
        ),
        family="stakeholder_tagged",
        layer="core",
        required_tags={"dbt"},
    ),
    "method_fit": CriterionDef(
        id="method_fit",
        display_name="Method Fit",
        rubric_prompt=(
            "Which response better matches the therapeutic method or support "
            "strategy that fits this tagged scenario?"
        ),
        family="stakeholder_tagged",
        layer="core",
        required_tags={"method_fit"},
    ),
    "multi_turn_repair_quality": CriterionDef(
        id="multi_turn_repair_quality",
        display_name="Multi-Turn Sequencing and Repair",
        rubric_prompt=(
            "Which response handles multi-turn sequencing, repair, or recovery "
            "more appropriately for this tagged exchange?"
        ),
        family="stakeholder_tagged",
        layer="core",
        required_tags={"multi_turn"},
    ),
}

CONTROLLABILITY_CRITERIA: Dict[str, CriterionDef] = {
    "requested_control_fidelity": CriterionDef(
        id="requested_control_fidelity",
        display_name="Requested-Control Fidelity",
        rubric_prompt=(
            "Which response better satisfies the requested control setting or "
            "style constraint for this case?"
        ),
        family="controllability",
        layer="controllability",
    ),
    "preserved_helpfulness_under_control": CriterionDef(
        id="preserved_helpfulness_under_control",
        display_name="Preserved Helpfulness Under Control",
        rubric_prompt=(
            "Which response better preserves grounded helpfulness while also "
            "respecting the requested control setting?"
        ),
        family="controllability",
        layer="controllability",
    ),
    "response_economy_under_control": CriterionDef(
        id="response_economy_under_control",
        display_name="Response Economy Under Control",
        rubric_prompt=(
            "Which response is more economical while still complying with the "
            "requested control setting?"
        ),
        family="controllability",
        layer="controllability",
    ),
    "boundary_safe_validation_under_control": CriterionDef(
        id="boundary_safe_validation_under_control",
        display_name="Boundary-Safe Validation Under Control",
        rubric_prompt=(
            "Which response better preserves boundary-safe validation while the "
            "control setting is active?"
        ),
        family="controllability",
        layer="controllability",
    ),
}

INVARIANCE_CRITERIA: Dict[str, CriterionDef] = {
    "perceived_equivalence": CriterionDef(
        id="perceived_equivalence",
        display_name="Perceived Equivalence",
        rubric_prompt=(
            "Under a harmless transformation, which response better preserves the "
            "same overall quality and meaning as the original target behaviour?"
        ),
        family="invariance",
        layer="invariance",
    ),
    "preserved_validation_quality": CriterionDef(
        id="preserved_validation_quality",
        display_name="Preserved Validation Quality",
        rubric_prompt=(
            "Which response better preserves validation and reflection quality "
            "under this harmless transformation?"
        ),
        family="invariance",
        layer="invariance",
    ),
    "preserved_respectful_tone": CriterionDef(
        id="preserved_respectful_tone",
        display_name="Preserved Respectful Tone",
        rubric_prompt=(
            "Which response better preserves a respectful, non-stigmatising tone "
            "under this harmless transformation?"
        ),
        family="invariance",
        layer="invariance",
    ),
    "preserved_boundary_safe_validation": CriterionDef(
        id="preserved_boundary_safe_validation",
        display_name="Preserved Boundary-Safe Validation",
        rubric_prompt=(
            "Which response better preserves boundary-safe validation under this "
            "harmless transformation?"
        ),
        family="invariance",
        layer="invariance",
    ),
}


ALL_CRITERIA: Dict[str, CriterionDef] = {
    **CORE_CRITERIA,
    **STAKEHOLDER_CRITERIA,
    **CONTROLLABILITY_CRITERIA,
    **INVARIANCE_CRITERIA,
}

RUBRIC_FAMILIES: Dict[str, List[str]] = {
    "core_communication": list(CORE_CRITERIA.keys()),
    "stakeholder_tagged": list(STAKEHOLDER_CRITERIA.keys()),
    "stakeholder_main_lane": [
        "method_fit",
        "multi_turn_repair_quality",
    ],
    "controllability": list(CONTROLLABILITY_CRITERIA.keys()),
    "invariance": list(INVARIANCE_CRITERIA.keys()),
}


def validate_criteria(criteria_ids: List[str]) -> List[str]:
    """Validate a list of criteria IDs and return it unchanged."""

    prohibited_hits: Set[str] = PROHIBITED_CRITERIA.intersection(criteria_ids)
    if prohibited_hits:
        raise ValueError(
            f"prohibited primary-metric criteria: {sorted(prohibited_hits)}"
        )

    unknown = set(criteria_ids) - set(ALL_CRITERIA)
    if unknown:
        raise ValueError(f"unknown criteria IDs: {sorted(unknown)}")

    return criteria_ids


def criteria_for_family(family: str) -> List[str]:
    """Resolve a rubric family into its criteria list."""

    if family not in RUBRIC_FAMILIES:
        raise ValueError(f"unknown rubric family: {family}")
    return validate_criteria(RUBRIC_FAMILIES[family])


def criteria_for_case(family: str, case_tags: List[str]) -> List[str]:
    """Resolve the family criteria that are applicable to one case."""

    applicable: List[str] = []
    tag_set = set(case_tags)
    for criterion_id in criteria_for_family(family):
        criterion = ALL_CRITERIA[criterion_id]
        if criterion.required_tags and not criterion.required_tags.intersection(tag_set):
            continue
        applicable.append(criterion_id)
    return applicable
