#!/usr/bin/env python3
"""Generate controllability analysis notebooks for the legacy and v2 paths."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable, List


NOTEBOOK_DIR = Path(__file__).resolve().parents[2] / "notebooks" / "controlability"


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


LEGACY_STUDY_LOADER = """
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
                    "aggregate_score": aggregate.get("score"),
                    "aggregate_provisional_components": aggregate.get("provisional_components"),
                }
            )

            for entry in payload.get("controlled_profile", []):
                profile_rows.append(
                    {
                        "model": payload.get("model", model_dir.name),
                        "study": payload.get("study", study_code),
                        "metric_name": entry.get("metric_name"),
                        "classification": entry.get("classification"),
                        "controlled_value": entry.get("controlled_value"),
                        "baseline_value": entry.get("baseline_value"),
                        "delta_from_baseline": entry.get("delta_from_baseline"),
                        "control_score": entry.get("control_score"),
                        "included_in_rollup": entry.get("included_in_rollup"),
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
                    }
                )

    return pd.DataFrame(primary_rows), pd.DataFrame(profile_rows), pd.DataFrame(summary_rows)
"""


V2_STUDY_LOADER = """
def load_controllability_v2_study_payload(study_key: str):
    study_file_map = {
        "A": "ctrl_v2_study_a_results.json",
        "A_bias": "ctrl_v2_study_a_bias_results.json",
        "B": "ctrl_v2_study_b_results.json",
        "B_multi_turn": "ctrl_v2_study_b_multi_turn_results.json",
        "C": "ctrl_v2_study_c_results.json",
    }
    target_file = study_file_map[study_key]

    arm_rows = []
    delta_rows = []
    summary_rows = []

    if not RESULTS_DIR.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for model_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
        study_path = model_dir / target_file
        if study_path.exists():
            payload = json.loads(study_path.read_text(encoding="utf-8"))
            exclusions = payload.get("exclusions") or {}
            notes = payload.get("notes") or []
            for arm_name, arm_payload in sorted((payload.get("arms") or {}).items()):
                primary_metric = arm_payload.get("primary_metric") or {}
                row = {
                    "model": payload.get("model", model_dir.name),
                    "study": payload.get("study", study_key),
                    "arm": arm_name,
                    "primary_metric_name": primary_metric.get("metric_name"),
                    "primary_metric_value": primary_metric.get("value"),
                    "primary_ci_lower": primary_metric.get("ci_lower"),
                    "primary_ci_upper": primary_metric.get("ci_upper"),
                    "notes": " | ".join(arm_payload.get("notes") or []),
                }
                for key, value in (arm_payload.get("task_metrics") or {}).items():
                    row[f"task_{key}"] = value
                for key, value in (arm_payload.get("counts") or {}).items():
                    row[f"count_{key}"] = value
                for key, value in exclusions.items():
                    row[f"exclusion_{key}"] = value
                arm_rows.append(row)

            for delta in payload.get("pairwise_deltas", []):
                row = {
                    "model": payload.get("model", model_dir.name),
                    "study": payload.get("study", study_key),
                    "from_arm": delta.get("from_arm"),
                    "to_arm": delta.get("to_arm"),
                    "n_pairs": delta.get("n_pairs"),
                    "notes": " | ".join(delta.get("notes") or []),
                }
                for key, value in (delta.get("metrics") or {}).items():
                    row[f"delta_{key}"] = value
                delta_rows.append(row)

        summary_path = model_dir / "controllability_v2_summary.json"
        if summary_path.exists():
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            study_payload = (summary_payload.get("studies") or {}).get(study_key)
            if study_payload:
                summary_rows.append(
                    {
                        "model": summary_payload.get("model", model_dir.name),
                        "study": study_key,
                        "arm_count": len(study_payload.get("arms") or {}),
                    }
                )

    return pd.DataFrame(arm_rows), pd.DataFrame(delta_rows), pd.DataFrame(summary_rows)
"""


V2_SUMMARY_LOADER = """
def load_controllability_v2_summary_rows():
    study_rows = []
    if not RESULTS_DIR.exists():
        return pd.DataFrame()

    for model_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
        summary_path = model_dir / "controllability_v2_summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for study_key, study_payload in sorted((payload.get("studies") or {}).items()):
            for arm_name, arm_payload in sorted((study_payload.get("arms") or {}).items()):
                primary_metric = arm_payload.get("primary_metric") or {}
                study_rows.append(
                    {
                        "model": payload.get("model", model_dir.name),
                        "study": study_key,
                        "arm": arm_name,
                        "primary_metric_name": primary_metric.get("metric_name"),
                        "primary_metric_value": primary_metric.get("value"),
                    }
                )

    return pd.DataFrame(study_rows)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def legacy_study_notebook(
    *,
    title: str,
    filename: str,
    study_code: str,
    primary_label: str,
) -> None:
    cells = [
        markdown_cell(
            f"""
            # {title}

            Legacy controllability notebook for Study {study_code}.

            This notebook reads the original `ctrl_study_*` structured outputs.
            Use the `v2` notebooks beside it for the same-case three-arm evaluation path.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(f'STUDY_CODE = "{study_code}"\nPRIMARY_LABEL = "{primary_label}"'),
        code_cell(LEGACY_STUDY_LOADER),
        code_cell(
            """
            primary_df, profile_df, summary_df = load_controllability_study_payload(STUDY_CODE)
            display(primary_df.sort_values("primary_metric_value", ascending=False).reset_index(drop=True))
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / filename, cells)


def v2_study_notebook(
    *,
    title: str,
    filename: str,
    study_key: str,
    primary_label: str,
    task_metric_labels: List[str],
    notes: List[str],
) -> None:
    metric_columns_literal = "[" + ", ".join(repr(f"task_{label}") for label in task_metric_labels) + "]"
    cells = [
        markdown_cell(
            f"""
            # {title}

            Arm-aware notebook for the controllability `v2` path.

            This notebook reads the same-case three-arm results:
            - `spontaneous`
            - `generic_control`
            - `explicit_control`

            Primary study view: `{primary_label}`

            Notes:
            {chr(10).join(f"- {note}" for note in notes)}
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(f'STUDY_KEY = "{study_key}"\nPRIMARY_LABEL = "{primary_label}"\nTASK_METRIC_COLUMNS = {metric_columns_literal}'),
        code_cell(V2_STUDY_LOADER),
        code_cell(
            """
            arm_df, delta_df, summary_df = load_controllability_v2_study_payload(STUDY_KEY)
            if arm_df.empty:
                print(f"No controllability v2 results found yet for {STUDY_KEY}.")
            else:
                display(arm_df.sort_values(["arm", "primary_metric_value"], ascending=[True, False]).reset_index(drop=True))
            """
        ),
        markdown_cell("## Primary Metric by Arm"),
        code_cell(
            """
            if arm_df.empty:
                print("Skipping primary metric plot - no v2 study results found.")
            else:
                plot_df = arm_df.sort_values(["arm", "model"]).reset_index(drop=True)
                fig, ax = plt.subplots(figsize=(16, 7))
                sns.barplot(data=plot_df, x="model", y="primary_metric_value", hue="arm", ax=ax)
                ax.set_title(f"{STUDY_KEY} primary controllability metric by arm", fontsize=14, fontweight="bold")
                ax.set_xlabel("Model")
                ax.set_ylabel("Primary metric value")
                plt.xticks(rotation=45, ha="right")
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.show()
            """
        ),
        markdown_cell("## Task Metrics by Arm"),
        code_cell(
            """
            if arm_df.empty:
                print("Skipping task-metric view - no v2 study results found.")
            else:
                available_columns = [col for col in TASK_METRIC_COLUMNS if col in arm_df.columns and arm_df[col].notna().any()]
                if not available_columns:
                    print("No requested task metrics are populated yet.")
                else:
                    metric_df = arm_df[["model", "arm"] + available_columns].copy()
                    long_df = metric_df.melt(id_vars=["model", "arm"], value_vars=available_columns, var_name="metric", value_name="value").dropna(subset=["value"])
                    display(long_df.sort_values(["metric", "arm", "model"]).reset_index(drop=True))
                    fig, ax = plt.subplots(figsize=(16, 7))
                    sns.barplot(data=long_df, x="metric", y="value", hue="arm", ax=ax)
                    ax.set_title(f"{STUDY_KEY} task metrics by arm", fontsize=14, fontweight="bold")
                    ax.set_xlabel("Metric")
                    ax.set_ylabel("Value")
                    plt.xticks(rotation=35, ha="right")
                    ax.grid(axis="y", alpha=0.3)
                    plt.tight_layout()
                    plt.show()
            """
        ),
        markdown_cell("## Pairwise Deltas"),
        code_cell(
            """
            if delta_df.empty:
                print("No pairwise arm deltas are available yet.")
            else:
                display(delta_df.sort_values(["from_arm", "to_arm", "model"]).reset_index(drop=True))
            """
        ),
        markdown_cell("## Coverage Summary"),
        code_cell(
            """
            if summary_df.empty:
                print("No v2 summary rows were found for this study yet.")
            else:
                display(summary_df.sort_values("model").reset_index(drop=True))
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / filename, cells)


def v2_summary_notebook() -> None:
    cells = [
        markdown_cell(
            """
            # Controllability V2 Summary Analysis

            This notebook compares the arm-aware `v2` controllability outputs across studies and models.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(V2_SUMMARY_LOADER),
        code_cell(
            """
            summary_df = load_controllability_v2_summary_rows()
            if summary_df.empty:
                print("No controllability v2 summary files found yet.")
            else:
                display(summary_df.sort_values(["study", "arm", "primary_metric_value"], ascending=[True, True, False]).reset_index(drop=True))
            """
        ),
        markdown_cell("## Primary Metrics Across Studies and Arms"),
        code_cell(
            """
            if summary_df.empty:
                print("Skipping summary plot - no v2 summary rows found.")
            else:
                fig, ax = plt.subplots(figsize=(18, 8))
                sns.barplot(data=summary_df, x="study", y="primary_metric_value", hue="arm", ax=ax)
                ax.set_title("Controllability v2 primary metrics by study and arm", fontsize=14, fontweight="bold")
                ax.set_xlabel("Study")
                ax.set_ylabel("Primary metric value")
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()
                plt.show()
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "controllability_v2_summary_analysis.ipynb", cells)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    legacy_study_notebook(
        title="Study A Controllability Analysis",
        filename="study_a_controllability_analysis.ipynb",
        study_code="A",
        primary_label="RA / Reasoning Adherence",
    )
    legacy_study_notebook(
        title="Study B Controllability Analysis",
        filename="study_b_controllability_analysis.ipynb",
        study_code="B",
        primary_label="CHR / Controlled Hallucination Rate",
    )
    legacy_study_notebook(
        title="Study C Controllability Analysis",
        filename="study_c_controllability_analysis.ipynb",
        study_code="C",
        primary_label="CER / Controlled Entity Recall",
    )

    v2_study_notebook(
        title="Study A Controllability V2 Analysis",
        filename="study_a_controllability_v2_analysis.ipynb",
        study_key="A",
        primary_label="RA / Reasoning Adherence",
        task_metric_labels=["step_f1", "accuracy"],
        notes=["Use the same-case three-arm results here rather than comparing cross-split baselines."],
    )
    v2_study_notebook(
        title="Study A Bias Controllability V2 Analysis",
        filename="study_a_bias_controllability_v2_analysis.ipynb",
        study_key="A_bias",
        primary_label="Silent Bias Rate",
        task_metric_labels=["biased_outcome_rate", "feature_mention_rate"],
        notes=[
            "Use spontaneous-vs-generic_control as the cleanest silent-bias comparison.",
            "Treat explicit_control as a transparency-focused arm because it asks for feature acknowledgement.",
        ],
    )
    v2_study_notebook(
        title="Study B Controllability V2 Analysis",
        filename="study_b_controllability_v2_analysis.ipynb",
        study_key="B",
        primary_label="CHR / Controlled Hallucination Rate",
        task_metric_labels=["sycophancy_probability", "flip_rate", "accuracy_injected"],
        notes=["The arm applies to both control and injected variants in the same case set."],
    )
    v2_study_notebook(
        title="Study B Multi-Turn Controllability V2 Analysis",
        filename="study_b_multiturn_controllability_v2_analysis.ipynb",
        study_key="B_multi_turn",
        primary_label="No-Flip Rate",
        task_metric_labels=["turn_of_flip_censored", "per_turn_agreement_rate"],
        notes=["Turn-of-flip is censored at T+1 when the model never flips."],
    )
    v2_study_notebook(
        title="Study C Controllability V2 Analysis",
        filename="study_c_controllability_v2_analysis.ipynb",
        study_key="C",
        primary_label="CER / Controlled Entity Recall",
        task_metric_labels=["recall_at_t10", "mean_entity_recall"],
        notes=["Study C v2 applies the arm to summaries only; dialogue remains outside the v2 comparison."],
    )
    v2_summary_notebook()
    print(f"Generated controllability notebooks in {NOTEBOOK_DIR}")


if __name__ == "__main__":
    main()
