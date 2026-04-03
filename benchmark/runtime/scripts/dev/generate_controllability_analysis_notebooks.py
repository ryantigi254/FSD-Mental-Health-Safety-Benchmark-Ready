#!/usr/bin/env python3
"""Generate canonical arm-aware controllability analysis notebooks."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable


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

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

RESULTS_DIR = next(
    (p for p in [Path("results"), Path("../results"), Path("../../results")] if p.exists()),
    Path("results"),
)
print(f"Using RESULTS_DIR: {RESULTS_DIR.resolve()}")
"""


STUDY_LOADER = """
def load_controllability_study_payload(study_key: str):
    study_file_map = {
        "A": "ctrl_study_a_results.json",
        "A_bias": "ctrl_study_a_bias_results.json",
        "B": "ctrl_study_b_results.json",
        "B_multi_turn": "ctrl_study_b_multi_turn_results.json",
        "C": "ctrl_study_c_results.json",
    }
    target_file = study_file_map[study_key]

    arm_rows = []
    delta_rows = []
    summary_rows = []

    if not RESULTS_DIR.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for model_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
        study_path = model_dir / target_file
        if not study_path.exists():
            legacy_v2_path = model_dir / target_file.replace("ctrl_study_", "ctrl_v2_study_")
            study_path = legacy_v2_path
        if study_path.exists():
            payload = json.loads(study_path.read_text(encoding="utf-8"))
            exclusions = payload.get("exclusions") or {}
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

        summary_path = model_dir / "controllability_summary.json"
        if not summary_path.exists():
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


SUMMARY_LOADER = """
def load_controllability_summary_rows():
    rows = []
    if not RESULTS_DIR.exists():
        return pd.DataFrame()

    for model_dir in sorted(path for path in RESULTS_DIR.iterdir() if path.is_dir()):
        summary_path = model_dir / "controllability_summary.json"
        if not summary_path.exists():
            summary_path = model_dir / "controllability_v2_summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for study_key, study_payload in sorted((payload.get("studies") or {}).items()):
            for arm_name, arm_payload in sorted((study_payload.get("arms") or {}).items()):
                primary_metric = arm_payload.get("primary_metric") or {}
                rows.append(
                    {
                        "model": payload.get("model", model_dir.name),
                        "study": study_key,
                        "arm": arm_name,
                        "primary_metric_name": primary_metric.get("metric_name"),
                        "primary_metric_value": primary_metric.get("value"),
                    }
                )

    return pd.DataFrame(rows)
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


def study_notebook(*, title: str, filename: str, study_key: str, metric_hint: str) -> None:
    cells = [
        markdown_cell(
            f"""
            # {title}

            Canonical arm-aware controllability notebook for Study {study_key}.

            The notebook reads the canonical `ctrl_study_*` outputs. It will also
            fall back to the older `ctrl_v2_study_*` aliases when you are reviewing
            legacy runs.

            Canonical arms:

            - `spontaneous`
            - `generic_control`
            - `explicit_control`
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(STUDY_LOADER),
        code_cell(
            f"""
            STUDY_KEY = "{study_key}"
            arm_df, delta_df, summary_df = load_controllability_study_payload(STUDY_KEY)
            print("Arms:", len(arm_df))
            print("Pairwise deltas:", len(delta_df))
            print("Summary rows:", len(summary_df))
            arm_df.head()
            """
        ),
        code_cell(
            f"""
            if not arm_df.empty:
                display_cols = ["model", "arm", "primary_metric_value"]
                extra_cols = [col for col in arm_df.columns if col.startswith("task_")]
                display(arm_df[display_cols + extra_cols].sort_values(["model", "arm"]))
            """
        ),
        code_cell(
            """
            if not arm_df.empty:
                plot_df = arm_df.copy()
                plt.figure(figsize=(12, 6))
                sns.barplot(data=plot_df, x="model", y="primary_metric_value", hue="arm")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.show()
            """
        ),
        code_cell(
            """
            if not delta_df.empty:
                display(delta_df.sort_values(["model", "from_arm", "to_arm"]))
            """
        ),
    ]
    if study_key == "B_multi_turn":
        cells.extend(
            [
                markdown_cell(
                    """
                    ## Diagnostic companions

                    These scalar pressure-response diagnostics refine the multi-turn picture without
                    replacing `no_flip_rate` as the governing controllability metric.
                    """
                ),
                code_cell(
                    """
                    if not arm_df.empty:
                        diagnostic_cols = [
                            "model",
                            "arm",
                            "task_stance_shift_slope_mean",
                            "task_sycophancy_auc_mean",
                            "task_soften_before_flip_rate",
                            "count_n_cases_scored",
                            "count_n_cases_flipped",
                        ]
                        available_cols = [col for col in diagnostic_cols if col in arm_df.columns]
                        display(arm_df[available_cols].sort_values(["model", "arm"]))

                        for metric_col in [
                            "task_stance_shift_slope_mean",
                            "task_sycophancy_auc_mean",
                            "task_soften_before_flip_rate",
                        ]:
                            if metric_col not in arm_df.columns:
                                continue
                            metric_df = arm_df.dropna(subset=[metric_col]).sort_values([metric_col], ascending=False)
                            if metric_df.empty:
                                continue
                            print(metric_col)
                            display(metric_df[["model", "arm", metric_col]])
                    """
                ),
            ]
        )
    cells.append(markdown_cell(f"Primary metric hint: `{metric_hint}`."))
    write_notebook(NOTEBOOK_DIR / filename, cells)


def summary_notebook() -> None:
    cells = [
        markdown_cell(
            """
            # Controllability Summary Analysis

            Canonical cross-study arm-aware summary notebook.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(SUMMARY_LOADER),
        code_cell(
            """
            summary_df = load_controllability_summary_rows()
            print("Summary rows:", len(summary_df))
            summary_df.head()
            """
        ),
        code_cell(
            """
            if not summary_df.empty:
                plt.figure(figsize=(12, 6))
                sns.barplot(data=summary_df, x="study", y="primary_metric_value", hue="arm")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.show()
            """
        ),
    ]
    write_notebook(NOTEBOOK_DIR / "controllability_summary_analysis.ipynb", cells)


def main() -> None:
    study_notebook(
        title="Study A Controllability Analysis",
        filename="study_a_controllability_analysis.ipynb",
        study_key="A",
        metric_hint="reasoning_adherence",
    )
    study_notebook(
        title="Study A Bias Controllability Analysis",
        filename="study_a_bias_controllability_analysis.ipynb",
        study_key="A_bias",
        metric_hint="silent_bias_rate",
    )
    study_notebook(
        title="Study B Controllability Analysis",
        filename="study_b_controllability_analysis.ipynb",
        study_key="B",
        metric_hint="controlled_hallucination_rate",
    )
    study_notebook(
        title="Study B Multi-turn Controllability Analysis",
        filename="study_b_multiturn_controllability_analysis.ipynb",
        study_key="B_multi_turn",
        metric_hint="no_flip_rate",
    )
    study_notebook(
        title="Study C Controllability Analysis",
        filename="study_c_controllability_analysis.ipynb",
        study_key="C",
        metric_hint="controlled_entity_recall",
    )
    summary_notebook()


if __name__ == "__main__":
    main()
