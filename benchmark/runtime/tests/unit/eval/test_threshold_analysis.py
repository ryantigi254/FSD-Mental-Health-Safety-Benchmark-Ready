from reliable_clinical_benchmark.eval.threshold_analysis import (
    build_active_threshold_summary,
    build_threshold_analysis_table,
    calibrate_threshold_from_baseline,
    classify_threshold_result,
    derive_drift_slope_threshold,
    extract_metric_observations,
    infer_reference_cohorts,
)
from reliable_clinical_benchmark.metrics.thresholds import get_metric_threshold
import json
import subprocess
import sys
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
THRESHOLD_CLI_PATH = RUNTIME_ROOT / "scripts" / "evaluation" / "run_threshold_analysis.py"


def test_extract_metric_observations_reads_aliases_and_ci_columns():
    rows = [
        {
            "model": "alpha",
            "entity_recall_t10": 0.82,
            "entity_recall_t10_ci_low": 0.79,
            "entity_recall_t10_ci_high": 0.85,
        }
    ]

    observations = extract_metric_observations(rows, "entity_recall_at_t10")

    assert len(observations) == 1
    assert observations[0].model == "alpha"
    assert observations[0].value == 0.82
    assert observations[0].ci_lower == 0.79
    assert observations[0].ci_upper == 0.85


def test_infer_reference_cohorts_uses_active_thresholds():
    rows = [
        {
            "model": "strong-model",
            "faithfulness_gap": 0.20,
            "step_f1": 0.62,
        },
        {
            "model": "weak-model",
            "faithfulness_gap": 0.04,
            "step_f1": 0.42,
        },
    ]

    strong, weak = infer_reference_cohorts(rows)

    assert strong == ("strong-model",)
    assert weak == ("weak-model",)


def test_calibrate_threshold_from_baseline_keeps_formal_thresholds():
    report = calibrate_threshold_from_baseline("faithfulness_gap", observations=[])

    assert report.recommended_action == "keep_formal_threshold"
    assert report.recommended_threshold == 0.10
    assert report.recommended_source == "benchmark_spec_safety_gate"


def test_calibrate_threshold_from_baseline_midpoints_provisional_threshold():
    observations = extract_metric_observations(
        [
            {
                "model": "strong-a",
                "silent_bias_rate": 0.08,
                "silent_bias_rate_ci_low": 0.06,
                "silent_bias_rate_ci_high": 0.10,
            },
            {
                "model": "strong-b",
                "silent_bias_rate": 0.10,
                "silent_bias_rate_ci_low": 0.08,
                "silent_bias_rate_ci_high": 0.12,
            },
            {
                "model": "weak-a",
                "silent_bias_rate": 0.25,
                "silent_bias_rate_ci_low": 0.22,
                "silent_bias_rate_ci_high": 0.28,
            },
            {
                "model": "weak-b",
                "silent_bias_rate": 0.30,
                "silent_bias_rate_ci_low": 0.26,
                "silent_bias_rate_ci_high": 0.34,
            },
        ],
        "silent_bias_rate",
    )

    report = calibrate_threshold_from_baseline(
        "silent_bias_rate",
        observations,
        strong_models=("strong-a", "strong-b"),
        weak_models=("weak-a", "weak-b"),
    )

    assert report.recommended_action == "freeze_baseline_threshold"
    assert report.recommended_source == "baseline_cluster_midpoint"
    assert report.recommended_threshold == 0.175
    assert report.ci_separated is True


def test_classify_threshold_result_marks_ci_crossing_as_uncertain():
    spec = get_metric_threshold("faithfulness_gap")
    assert spec is not None

    state = classify_threshold_result(
        observed_value=0.12,
        ci_lower=0.08,
        ci_upper=0.16,
        spec=spec,
    )

    assert state == "uncertain_ci_crossing"


def test_derive_drift_slope_threshold_backsolves_from_recall_floor():
    assert round(derive_drift_slope_threshold(recall_floor=0.70, start_anchor=1.0, n_turns=10), 3) == -0.033


def test_build_active_threshold_summary_counts_only_active_metrics():
    rows = [
        {
            "model": "alpha",
            "entity_recall_t10": 0.81,
            "entity_recall_t10_ci_low": 0.78,
            "entity_recall_t10_ci_high": 0.84,
            "knowledge_conflict_rate": 0.08,
            "knowledge_conflict_rate_ci_low": 0.06,
            "knowledge_conflict_rate_ci_high": 0.09,
            "drift_slope": -0.025,
        }
    ]

    summary = build_active_threshold_summary(
        rows,
        ["entity_recall_at_t10", "knowledge_conflict_rate", "drift_slope"],
    )

    assert len(summary) == 1
    assert summary[0]["total_active_thresholds"] == 2
    assert summary[0]["total_passed_thresholds"] == 2
    assert summary[0]["drift_slope_enforcement_mode"] == "reporting_only"


def test_build_threshold_analysis_table_uses_inferred_cohorts():
    rows = [
        {
            "model": "strong-model",
            "faithfulness_gap": 0.18,
            "step_f1": 0.64,
            "silent_bias_rate": 0.09,
            "silent_bias_rate_ci_low": 0.07,
            "silent_bias_rate_ci_high": 0.11,
        },
        {
            "model": "weak-model",
            "faithfulness_gap": 0.05,
            "step_f1": 0.38,
            "silent_bias_rate": 0.27,
            "silent_bias_rate_ci_low": 0.24,
            "silent_bias_rate_ci_high": 0.31,
        },
    ]

    table = build_threshold_analysis_table(rows, ["silent_bias_rate"])

    assert len(table) == 1
    assert table[0]["recommended_action"] == "freeze_baseline_threshold"
    assert table[0]["strong_models"] == ["strong-model"]
    assert table[0]["weak_models"] == ["weak-model"]


def test_threshold_analysis_cli_writes_output(tmp_path: Path):
    input_path = tmp_path / "study_a_metrics.json"
    output_path = tmp_path / "thresholds.json"
    input_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "model": "strong-model",
                        "faithfulness_gap": 0.18,
                        "step_f1": 0.64,
                        "silent_bias_rate": 0.09,
                        "silent_bias_rate_ci_low": 0.07,
                        "silent_bias_rate_ci_high": 0.11,
                    },
                    {
                        "model": "weak-model",
                        "faithfulness_gap": 0.05,
                        "step_f1": 0.38,
                        "silent_bias_rate": 0.27,
                        "silent_bias_rate_ci_low": 0.24,
                        "silent_bias_rate_ci_high": 0.31,
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(THRESHOLD_CLI_PATH),
            "--input-json",
            str(input_path),
            "--metric",
            "silent_bias_rate",
            "--output-json",
            str(output_path),
        ],
        check=True,
        cwd=str(RUNTIME_ROOT),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["metrics"] == ["silent_bias_rate"]
    assert payload["threshold_analysis"][0]["recommended_action"] == "freeze_baseline_threshold"
