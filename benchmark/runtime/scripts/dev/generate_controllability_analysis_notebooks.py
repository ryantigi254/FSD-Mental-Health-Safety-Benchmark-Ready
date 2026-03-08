#!/usr/bin/env python3
"""Generate consistent controllability analysis notebooks for Studies A, B, C, and summary."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable, List


NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "notebooks"


def markdown_cell(text: str) -> dict:
    cleaned = textwrap.dedent(text).strip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in cleaned.split("\n")]}


def code_cell(source: str) -> dict:
    cleaned = textwrap.dedent(source).strip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cleaned.split("\n")],
    }


COMMON_SETUP = """
import sys
from pathlib import Path

for candidate_root in [Path.cwd(), Path.cwd().parent]:
    src_dir = candidate_root / "src"
    if src_dir.exists() and str(src_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(src_dir.resolve()))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

RESULTS_DIR = next(
    (p for p in [Path("results"), Path("../results"), Path("../../results")] if p.exists()),
    Path("results"),
)
print(f"Using RESULTS_DIR: {RESULTS_DIR.resolve()}")
"""


STUDY_LOADER = """
def load_controllability_study_payload(study_code: str):
    study_file_map = {
        "A": "ctrl_study_a_results.json",
        "B": "ctrl_study_b_results.json",
        "C": "ctrl_study_c_results.json",
    }
    target_file = study_file_map[study_code]

    primary_rows = []
    profile_rows = []
    summary_rows = []

    if not RESULTS_DIR.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for model_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
        study_path = model_dir / target_file
        if study_path.exists():
            payload = json.loads(study_path.read_text(encoding="utf-8"))
            primary_metric = payload.get("primary_metric", {})
            aggregate = payload.get("aggregate", {})
            primary_rows.append(
                {
                    "model": payload.get("model", model_dir.name),
                    "study": payload.get("study", study_code),
                    "primary_metric_name": primary_metric.get("metric_name"),
                    "primary_metric_value": primary_metric.get("value"),
                    "primary_ci_lower": primary_metric.get("ci_lower"),
                    "primary_ci_upper": primary_metric.get("ci_upper"),
                    "primary_threshold_value": (primary_metric.get("threshold") or {}).get("threshold_value"),
                    "primary_threshold_status": (primary_metric.get("threshold") or {}).get("status"),
                    "aggregate_score": aggregate.get("score"),
                    "aggregate_weighting_policy": aggregate.get("weighting_policy"),
                    "aggregate_provisional_components": aggregate.get("provisional_components"),
                }
            )

            for entry in payload.get("controlled_profile", []):
                threshold = entry.get("threshold") or {}
                profile_rows.append(
                    {
                        "model": payload.get("model", model_dir.name),
                        "study": payload.get("study", study_code),
                        "metric_name": entry.get("metric_name"),
                        "classification": entry.get("classification"),
                        "controlled_value": entry.get("controlled_value"),
                        "baseline_value": entry.get("baseline_value"),
                        "delta_from_baseline": entry.get("delta_from_baseline"),
                        "compliance_anchor": entry.get("compliance_anchor"),
                        "outcome_gain": entry.get("outcome_gain"),
                        "control_score": entry.get("control_score"),
                        "included_in_rollup": entry.get("included_in_rollup"),
                        "threshold_value": threshold.get("threshold_value"),
                        "threshold_status": threshold.get("status"),
                        "threshold_enforcement_mode": threshold.get("enforcement_mode"),
                        "threshold_source": threshold.get("threshold_source"),
                        "meets_threshold": threshold.get("meets_threshold"),
                        "notes": entry.get("notes"),
                    }
                )

        summary_path = model_dir / "controllability_summary.json"
        if summary_path.exists():
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            studies = summary_payload.get("studies", {})
            study_payload = studies.get(study_code)
            benchmark_payload = summary_payload.get("benchmark_control", {})
            if study_payload:
                summary_rows.append(
                    {
                        "model": summary_payload.get("model", model_dir.name),
                        "study": study_code,
                        "study_score": (study_payload.get("aggregate") or {}).get("score"),
                        "benchmark_control_score": benchmark_payload.get("score"),
                        "benchmark_provisional_components": benchmark_payload.get("provisional_components"),
                    }
                )

    return pd.DataFrame(primary_rows), pd.DataFrame(profile_rows), pd.DataFrame(summary_rows)
"""


RANKING_CELL = """
primary_df, profile_df, summary_df = load_controllability_study_payload(STUDY_CODE)

if primary_df.empty:
    print(f"No controllability results found yet for Study {STUDY_CODE}.")
else:
    ranking_df = primary_df.sort_values("primary_metric_value", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = range(1, len(ranking_df) + 1)
    ranking_df = ranking_df[
        [
            "rank",
            "model",
            "primary_metric_name",
            "primary_metric_value",
            "primary_ci_lower",
            "primary_ci_upper",
            "aggregate_score",
            "aggregate_provisional_components",
        ]
    ]
    print(f"Study {STUDY_CODE} controllability ranking")
    print("=" * 100)
    display(ranking_df)
"""


PRIMARY_PLOT_CELL = """
if primary_df.empty:
    print("Skipping primary metric plot - no controllability study results found.")
else:
    plot_df = primary_df.sort_values("primary_metric_value", ascending=False).reset_index(drop=True)
    vals = pd.to_numeric(plot_df["primary_metric_value"], errors="coerce").fillna(0.0).values
    ci_low = pd.to_numeric(plot_df["primary_ci_lower"], errors="coerce").fillna(plot_df["primary_metric_value"]).values
    ci_high = pd.to_numeric(plot_df["primary_ci_upper"], errors="coerce").fillna(plot_df["primary_metric_value"]).values
    yerr = np.vstack([vals - ci_low, ci_high - vals])

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(plot_df["model"], vals, yerr=yerr, capsize=6, alpha=0.8, color="#4C72B0")
    ax.set_title(f"Study {STUDY_CODE} Primary Controllability Metric", fontsize=14, fontweight="bold")
    ax.set_ylabel("Primary controllability value")
    ax.set_xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
"""


PROFILE_SCORE_CELL = """
if profile_df.empty:
    print("Skipping control-score profile plot - no controlled metric profile rows found.")
else:
    score_df = profile_df.dropna(subset=["control_score"]).copy()
    if score_df.empty:
        print("No metric-level control scores are populated yet.")
    else:
        fig, ax = plt.subplots(figsize=(16, 7))
        sns.barplot(
            data=score_df,
            x="metric_name",
            y="control_score",
            hue="model",
            ax=ax,
        )
        ax.set_title(f"Study {STUDY_CODE} Metric-Level Control Scores", fontsize=14, fontweight="bold")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Control score")
        plt.xticks(rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
"""


DELTA_CELL = """
if profile_df.empty:
    print("Skipping baseline-delta plot - no controlled metric profile rows found.")
else:
    delta_df = profile_df.dropna(subset=["delta_from_baseline"]).copy()
    if delta_df.empty:
        print("No baseline deltas are populated yet.")
    else:
        fig, ax = plt.subplots(figsize=(16, 7))
        sns.barplot(
            data=delta_df,
            x="metric_name",
            y="delta_from_baseline",
            hue="model",
            ax=ax,
        )
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.4)
        ax.set_title(f"Study {STUDY_CODE} Controlled Metric Shift vs Baseline", fontsize=14, fontweight="bold")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Controlled minus baseline")
        plt.xticks(rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
"""


AGGREGATE_CELL = """
if primary_df.empty:
    print("Skipping study aggregate plot - no controllability study results found.")
else:
    agg_df = primary_df.dropna(subset=["aggregate_score"]).sort_values("aggregate_score", ascending=False)
    if agg_df.empty:
        print("No study-level controllability aggregate scores are available yet.")
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(agg_df["model"], agg_df["aggregate_score"], color="#55A868", alpha=0.85)
        ax.set_title(f"Study {STUDY_CODE} Aggregate Controllability", fontsize=14, fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("Aggregate control score")
        plt.xticks(rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
"""


THRESHOLD_TABLE_CELL = """
if profile_df.empty:
    print("Skipping threshold/provenance table - no controlled profile rows found.")
else:
    threshold_cols = [
        "model",
        "metric_name",
        "classification",
        "control_score",
        "threshold_value",
        "threshold_status",
        "threshold_enforcement_mode",
        "threshold_source",
        "meets_threshold",
    ]
    display(profile_df[threshold_cols].sort_values(["metric_name", "model"]).reset_index(drop=True))
"""


OVERALL_CELL = """
if summary_df.empty:
    print("No benchmark-level controllability summary files were found yet.")
else:
    overall_df = summary_df.sort_values("benchmark_control_score", ascending=False).reset_index(drop=True)
    display(overall_df)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(overall_df["model"], overall_df["benchmark_control_score"], color="#C44E52", alpha=0.85)
    ax.set_title("Overall Benchmark Controllability", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Benchmark control score")
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
"""


def write_notebook(path: Path, cells: Iterable[dict]) -> None:
    payload = {
        "cells": list(cells),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def study_notebook(
    *,
    title: str,
    filename: str,
    study_code: str,
    primary_label: str,
    metric_labels: List[str],
) -> None:
    markdown = f"""
    # {title}

    This notebook is the dedicated controllability analysis for Study {study_code}.

    It focuses on:
    1. the study primary controllability metric (`{primary_label}`)
    2. metric-level control performance across the study metrics
    3. study-level aggregate controllability
    4. benchmark-level controllability context where the summary files are available

    ## Controlled metric profile

    The grouped profile rows are expected to show:
    - the controlled value for each metric
    - the baseline value when available
    - the baseline delta
    - the metric-level `control_score`
    - threshold provenance (`formal`, `provisional`, `derived`)
    - whether a metric is active or reporting-only

    Study metrics tracked here:
    {chr(10).join(f"- `{label}`" for label in metric_labels)}
    """

    cells = [
        markdown_cell(markdown),
        code_cell(COMMON_SETUP),
        code_cell(f'STUDY_CODE = "{study_code}"\nPRIMARY_LABEL = "{primary_label}"'),
        code_cell(STUDY_LOADER),
        markdown_cell("## Ranking"),
        code_cell(RANKING_CELL),
        markdown_cell("## Primary Metric"),
        code_cell(PRIMARY_PLOT_CELL),
        markdown_cell("## Metric-Level Control Scores"),
        code_cell(PROFILE_SCORE_CELL),
        markdown_cell("## Controlled vs Baseline Shift"),
        code_cell(DELTA_CELL),
        markdown_cell("## Study Aggregate"),
        code_cell(AGGREGATE_CELL),
        markdown_cell("## Threshold Provenance"),
        code_cell(THRESHOLD_TABLE_CELL),
        markdown_cell("## Overall Benchmark Controllability"),
        code_cell(OVERALL_CELL),
    ]
    write_notebook(NOTEBOOK_DIR / filename, cells)


def generation_first_notebook(
    *,
    title: str,
    filename: str,
    study_label: str,
    expected_cache_names: List[str],
    structured_result_name: str,
) -> None:
    cache_list_literal = "[" + ", ".join(repr(name) for name in expected_cache_names) + "]"
    cells = [
        markdown_cell(
            f"""
            # {title}

            This notebook is the dedicated controllability analysis for the **{study_label}** variant.

            Unlike Studies A, B single-turn, and C, this path is still **generation-first** in the repo today.
            So the notebook does two things:
            1. if a future structured result JSON exists, it will load and show it
            2. otherwise it falls back to the per-model generation caches and summarises coverage / exploratory control proxies

            This keeps the notebook aligned with controllability work without pretending the metric helper already exists.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(
            f"""
            EXPECTED_CACHE_NAMES = {cache_list_literal}
            STRUCTURED_RESULT_NAME = "{structured_result_name}"
            """
        ),
        code_cell(
            """
            def load_generation_first_variant():
                structured_rows = []
                cache_rows = []

                if not RESULTS_DIR.exists():
                    return pd.DataFrame(), pd.DataFrame()

                for model_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
                    structured_path = model_dir / STRUCTURED_RESULT_NAME
                    if structured_path.exists():
                        payload = json.loads(structured_path.read_text(encoding="utf-8"))
                        structured_rows.append(
                            {
                                "model": payload.get("model", model_dir.name),
                                "study": payload.get("study"),
                                "primary_metric": (payload.get("primary_metric") or {}).get("metric_name"),
                                "primary_value": (payload.get("primary_metric") or {}).get("value"),
                                "aggregate_score": (payload.get("aggregate") or {}).get("score"),
                            }
                        )

                    cache_path = next((model_dir / name for name in EXPECTED_CACHE_NAMES if (model_dir / name).exists()), None)
                    if cache_path is None:
                        continue

                    status_ok = 0
                    total_rows = 0
                    bias_feature_mentions = 0
                    bias_label_mentions = 0
                    disagreement_markers = 0
                    by_case = {}

                    with cache_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            total_rows += 1
                            if row.get("status") == "ok":
                                status_ok += 1
                            output_text = str(row.get("output_text") or row.get("response_text") or "")
                            output_lower = output_text.lower()

                            bias_feature = str(row.get("bias_feature") or "").lower().strip()
                            bias_label = str(row.get("bias_label") or "").lower().strip()
                            if bias_feature and bias_feature in output_lower:
                                bias_feature_mentions += 1
                            if bias_label and bias_label in output_lower:
                                bias_label_mentions += 1
                            if any(token in output_lower for token in ["disagree", "cannot agree", "i disagree", "not agree"]):
                                disagreement_markers += 1

                            case_id = str(row.get("id") or row.get("case_id") or f"row_{total_rows}")
                            by_case.setdefault(case_id, 0)
                            by_case[case_id] += 1

                    cache_rows.append(
                        {
                            "model": model_dir.name,
                            "cache_file": cache_path.name,
                            "total_rows": total_rows,
                            "ok_rows": status_ok,
                            "unique_cases": len(by_case),
                            "mean_rows_per_case": (sum(by_case.values()) / len(by_case)) if by_case else 0.0,
                            "bias_feature_mention_rate": (bias_feature_mentions / total_rows) if total_rows else None,
                            "bias_label_mention_rate": (bias_label_mentions / total_rows) if total_rows else None,
                            "disagreement_marker_rate": (disagreement_markers / total_rows) if total_rows else None,
                        }
                    )

                return pd.DataFrame(structured_rows), pd.DataFrame(cache_rows)

            structured_df, cache_df = load_generation_first_variant()
            """
        ),
        markdown_cell("## Structured Results (if available)"),
        code_cell(
            """
            if structured_df.empty:
                print("No standalone structured controllability result JSON exists for this variant yet.")
            else:
                display(structured_df.sort_values("primary_value", ascending=False).reset_index(drop=True))
            """
        ),
        markdown_cell("## Cache Coverage / Exploratory Proxy View"),
        code_cell(
            """
            if cache_df.empty:
                print("No controllability cache files were found for this variant yet.")
            else:
                display(cache_df.sort_values("ok_rows", ascending=False).reset_index(drop=True))
            """
        ),
        markdown_cell("## Exploratory Proxy Plot"),
        code_cell(
            """
            if cache_df.empty:
                print("Skipping exploratory plot - no cache rows found.")
            else:
                plot_col = next(
                    (
                        col
                        for col in [
                            "disagreement_marker_rate",
                            "bias_feature_mention_rate",
                            "bias_label_mention_rate",
                            "mean_rows_per_case",
                        ]
                        if col in cache_df.columns and cache_df[col].notna().any()
                    ),
                    None,
                )
                if plot_col is None:
                    print("No numeric exploratory proxy is available yet.")
                else:
                    plot_df = cache_df.sort_values(plot_col, ascending=False)
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.bar(plot_df["model"], plot_df[plot_col], color="#4C72B0", alpha=0.85)
                    ax.set_title(f"{title}: exploratory proxy view", fontsize=14, fontweight="bold")
                    ax.set_xlabel("Model")
                    ax.set_ylabel(plot_col)
                    plt.xticks(rotation=45, ha="right")
                    ax.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    plt.show()
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / filename, cells)


def summary_notebook() -> None:
    cells = [
        markdown_cell(
            """
            # Controllability Summary Analysis

            This notebook compares controllability across all studies at once.

            It is meant to answer:
            - how each model performs on Study A, B, and C controllability
            - how the benchmark-level controllability score compares across models
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(
            """
            def load_controllability_summary_rows():
                rows = []
                if not RESULTS_DIR.exists():
                    return pd.DataFrame()
                for model_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
                    summary_path = model_dir / "controllability_summary.json"
                    if not summary_path.exists():
                        continue
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                    studies = payload.get("studies", {})
                    rows.append(
                        {
                            "model": payload.get("model", model_dir.name),
                            "benchmark_control_score": (payload.get("benchmark_control") or {}).get("score"),
                            "study_a_control": ((studies.get("A") or {}).get("aggregate") or {}).get("score"),
                            "study_b_control": ((studies.get("B") or {}).get("aggregate") or {}).get("score"),
                            "study_c_control": ((studies.get("C") or {}).get("aggregate") or {}).get("score"),
                        }
                    )
                return pd.DataFrame(rows)

            summary_df = load_controllability_summary_rows()
            if summary_df.empty:
                print("No controllability summary files found yet.")
            else:
                display(summary_df.sort_values("benchmark_control_score", ascending=False).reset_index(drop=True))
            """
        ),
        markdown_cell("## Benchmark Control"),
        code_cell(
            """
            if summary_df.empty:
                print("Skipping benchmark control plot - no summary rows found.")
            else:
                plot_df = summary_df.sort_values("benchmark_control_score", ascending=False)
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.bar(plot_df["model"], plot_df["benchmark_control_score"], color="#8172B2", alpha=0.85)
                ax.set_title("Benchmark-Level Controllability", fontsize=14, fontweight="bold")
                ax.set_xlabel("Model")
                ax.set_ylabel("Benchmark control score")
                plt.xticks(rotation=45, ha="right")
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.show()
            """
        ),
        markdown_cell("## Study Aggregate Comparison"),
        code_cell(
            """
            if summary_df.empty:
                print("Skipping study aggregate comparison - no summary rows found.")
            else:
                long_df = summary_df.melt(
                    id_vars=["model"],
                    value_vars=["study_a_control", "study_b_control", "study_c_control"],
                    var_name="study",
                    value_name="aggregate_score",
                ).dropna(subset=["aggregate_score"])

                fig, ax = plt.subplots(figsize=(16, 7))
                sns.barplot(data=long_df, x="study", y="aggregate_score", hue="model", ax=ax)
                ax.set_title("Study-Level Controllability Aggregates", fontsize=14, fontweight="bold")
                ax.set_xlabel("Study")
                ax.set_ylabel("Aggregate control score")
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.show()
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "controllability_summary_analysis.ipynb", cells)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    study_notebook(
        title="Study A Controllability Analysis",
        filename="study_a_controllability_analysis.ipynb",
        study_code="A",
        primary_label="RA / Reasoning Adherence",
        metric_labels=["faithfulness_gap", "step_f1", "silent_bias_rate", "controlled_accuracy"],
    )
    generation_first_notebook(
        title="Study A Bias Controllability Analysis",
        filename="study_a_bias_controllability_analysis.ipynb",
        study_label="Study A Bias controllability",
        expected_cache_names=["ctrl_study_a_bias_generations.jsonl", "study_a_bias_generations.jsonl"],
        structured_result_name="ctrl_study_a_bias_results.json",
    )
    study_notebook(
        title="Study B Controllability Analysis",
        filename="study_b_controllability_analysis.ipynb",
        study_code="B",
        primary_label="CHR / Controlled Hallucination Rate",
        metric_labels=["sycophancy_probability", "evidence_hallucination", "flip_rate", "controlled_accuracy"],
    )
    generation_first_notebook(
        title="Study B Multi-Turn Controllability Analysis",
        filename="study_b_multiturn_controllability_analysis.ipynb",
        study_label="Study B multi-turn controllability",
        expected_cache_names=["ctrl_study_b_multi_turn_generations.jsonl", "study_b_multi_turn_generations.jsonl"],
        structured_result_name="ctrl_study_b_multi_turn_results.json",
    )
    study_notebook(
        title="Study C Controllability Analysis",
        filename="study_c_controllability_analysis.ipynb",
        study_code="C",
        primary_label="CER / Controlled Entity Recall",
        metric_labels=["entity_recall_at_t10", "knowledge_conflict_rate", "session_goal_alignment", "drift_slope"],
    )
    summary_notebook()
    print("Generated controllability analysis notebooks in benchmark/runtime/notebooks")


if __name__ == "__main__":
    main()
