"""Evaluation orchestration and runtime checks."""

from .results_schema import load_controllability_results, load_study_results
from .threshold_analysis import (
    build_active_threshold_summary,
    build_threshold_analysis_table,
    calibrate_threshold_from_baseline,
    classify_threshold_result,
    derive_drift_slope_threshold,
    extract_metric_observations,
    infer_reference_cohorts,
)


def validate_environment():
    from .runtime_checks import validate_environment as _validate_environment

    return _validate_environment()


def validate_data_files(data_dir: str = "data"):
    from .runtime_checks import validate_data_files as _validate_data_files

    return _validate_data_files(data_dir)


def check_model_availability(*args, **kwargs):
    from .runtime_checks import check_model_availability as _check_model_availability

    return _check_model_availability(*args, **kwargs)


__all__ = [
    "validate_environment",
    "validate_data_files",
    "check_model_availability",
    "load_study_results",
    "load_controllability_results",
    "build_active_threshold_summary",
    "build_threshold_analysis_table",
    "calibrate_threshold_from_baseline",
    "classify_threshold_result",
    "derive_drift_slope_threshold",
    "extract_metric_observations",
    "infer_reference_cohorts",
]
