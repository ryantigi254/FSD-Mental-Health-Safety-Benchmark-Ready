"""Evaluation metrics for Studies A, B, and C."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "calculate_faithfulness_gap": (".faithfulness", "calculate_faithfulness_gap"),
    "calculate_step_f1": (".faithfulness", "calculate_step_f1"),
    "calculate_silent_bias_rate": (".faithfulness", "calculate_silent_bias_rate"),
    "calculate_reasoning_adherence": (".faithfulness", "calculate_reasoning_adherence"),
    "check_reasoning_adherence": (".faithfulness", "check_reasoning_adherence"),
    "FaithfulnessResult": (".faithfulness", "FaithfulnessResult"),
    "calculate_sycophancy_probability": (".sycophancy", "calculate_sycophancy_probability"),
    "calculate_flip_rate": (".sycophancy", "calculate_flip_rate"),
    "evidence_hallucination_score": (".sycophancy", "evidence_hallucination_score"),
    "calculate_turn_of_flip": (".sycophancy", "calculate_turn_of_flip"),
    "calculate_controlled_hallucination_rate": (".sycophancy", "calculate_controlled_hallucination_rate"),
    "check_controlled_sycophancy": (".sycophancy", "check_controlled_sycophancy"),
    "SycophancyResult": (".sycophancy", "SycophancyResult"),
    "compute_entity_recall_curve": (".drift", "compute_entity_recall_curve"),
    "calculate_knowledge_conflict_rate": (".drift", "calculate_knowledge_conflict_rate"),
    "calculate_alignment_score": (".drift", "calculate_alignment_score"),
    "calculate_controlled_entity_recall": (".drift", "calculate_controlled_entity_recall"),
    "check_controlled_entity_recall": (".drift", "check_controlled_entity_recall"),
    "DriftResult": (".drift", "DriftResult"),
    "calculate_compliance_rate": (".controllability", "calculate_compliance_rate"),
    "ControllabilityResult": (".controllability", "ControllabilityResult"),
    "STUDY_A_CONSTRAINT": (".controllability", "STUDY_A_CONSTRAINT"),
    "STUDY_B_CONSTRAINT": (".controllability", "STUDY_B_CONSTRAINT"),
    "STUDY_C_CONSTRAINT": (".controllability", "STUDY_C_CONSTRAINT"),
    "MetricThresholdSpec": (".thresholds", "MetricThresholdSpec"),
    "ThresholdAssessment": (".thresholds", "ThresholdAssessment"),
    "get_metric_threshold": (".thresholds", "get_metric_threshold"),
    "list_metric_thresholds": (".thresholds", "list_metric_thresholds"),
    "evaluate_metric_threshold": (".thresholds", "evaluate_metric_threshold"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
