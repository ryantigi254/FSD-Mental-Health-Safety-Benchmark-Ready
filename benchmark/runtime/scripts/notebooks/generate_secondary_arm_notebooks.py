#!/usr/bin/env python3
"""Generate expanded secondary-arm analysis notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = RUNTIME_ROOT / "notebooks"

STUDIES = {
    "study_a": "Study A",
    "study_a_bias": "Study A Bias",
    "study_b": "Study B",
    "study_b_multi_turn": "Study B Multi-turn",
    "study_c": "Study C",
}

ARM_NOTEBOOKS = [
    {
        "folder": NOTEBOOK_ROOT / "controllability",
        "filename_template": "{study}_controllability_analysis.ipynb",
        "title_template": "{title} Controllability Analysis",
        "lanes": ["controllability"],
        "description": "Explicit-control responsiveness against the spontaneous control arm.",
    },
    {
        "folder": NOTEBOOK_ROOT / "invariance" / "main",
        "filename_template": "{study}_invariance_analysis.ipynb",
        "title_template": "{title} Main Invariance Analysis",
        "lanes": ["invariance"],
        "description": "Raw/main generations compared with harmless invariance perturbations.",
    },
    {
        "folder": NOTEBOOK_ROOT / "invariance" / "ctrl_invariance",
        "filename_template": "{study}_ctrl_invariance_analysis.ipynb",
        "title_template": "{title} Ctrl-Invariance Analysis",
        "lanes": ["ctrl-invariance"],
        "description": "Explicit-control generations compared with controlled invariance perturbations.",
    },
    {
        "folder": NOTEBOOK_ROOT / "invariance" / "invariance_ctrl",
        "filename_template": "{study}_invariance_ctrl_analysis.ipynb",
        "title_template": "{title} Invariance-Control Analysis",
        "lanes": ["ctrl-invariance"],
        "description": "The same normalised ctrl-invariance metric lane viewed as invariance under control.",
    },
]

SETUP_TEMPLATE = """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

NOTEBOOK_UTILS_DIR = Path("benchmark/runtime/notebooks").resolve()
if not NOTEBOOK_UTILS_DIR.exists():
    NOTEBOOK_UTILS_DIR = Path.cwd().resolve()
    while NOTEBOOK_UTILS_DIR.name != "notebooks" and NOTEBOOK_UTILS_DIR.parent != NOTEBOOK_UTILS_DIR:
        NOTEBOOK_UTILS_DIR = NOTEBOOK_UTILS_DIR.parent
if str(NOTEBOOK_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_UTILS_DIR))

from notebook_utils import (
    find_runtime_root,
    load_secondary_coverage,
    load_secondary_flat,
    plot_secondary_arm_values,
    plot_secondary_coverage_heatmap,
    plot_secondary_delta_ci,
    plot_secondary_model_heatmap,
    plot_secondary_per_model_heatmaps,
    secondary_metric_headline,
    secondary_missing_table,
    secondary_not_measurable_table,
    secondary_threshold_audit,
    setup_notebook_style,
)

setup_notebook_style()
runtime_root = find_runtime_root()
metric_root = runtime_root / "metric-results" / "secondary_branch_metrics"
study = "{study}"
lanes = {lanes!r}
flat_all = load_secondary_flat(metric_root)
coverage_all = load_secondary_coverage(metric_root)
df = flat_all[(flat_all["study"] == study) & (flat_all["lane"].isin(lanes))].copy()
coverage = coverage_all[(coverage_all["study"] == study) & (coverage_all["lane"].isin(lanes))].copy()
print("Runtime root:", runtime_root)
print("Rows:", len(df), "Coverage rows:", len(coverage))
"""


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook(*, study: str, title: str, lanes: list[str], description: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        markdown_cell(f"# {title}"),
        markdown_cell(description),
        code_cell(SETUP_TEMPLATE.format(study=study, lanes=lanes)),
        markdown_cell("## Provenance and coverage"),
        code_cell(
            """
display(coverage.sort_values(["lane", "model", "study"]))
display(secondary_missing_table(coverage))
display(secondary_not_measurable_table(df))
fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
plot_secondary_coverage_heatmap(coverage, ax=ax, title=f"{study}: coverage status")
plt.show()
"""
        ),
        markdown_cell("## Metric headline table"),
        code_cell(
            """
headline = secondary_metric_headline(df)
display(headline.round(4))
audit = secondary_threshold_audit(df)
display(audit.round(4))
"""
        ),
        markdown_cell("## Paired-delta confidence intervals"),
        code_cell(
            """
fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * max(len(df), 8))), constrained_layout=True)
plot_secondary_delta_ci(df, ax=ax, title=f"{study}: largest paired deltas", max_rows=30)
plt.show()
"""
        ),
        markdown_cell("## Base versus variant values"),
        code_cell(
            """
fig, ax = plt.subplots(figsize=(12, max(5, 0.38 * max(len(df), 8))), constrained_layout=True)
plot_secondary_arm_values(df, ax=ax, title=f"{study}: base and variant metric values", max_rows=28)
plt.show()
"""
        ),
        markdown_cell("## Model sensitivity heatmap"),
        code_cell(
            """
fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
plot_secondary_model_heatmap(df, ax=ax, title=f"{study}: median paired delta by model")
plt.show()
"""
        ),
        markdown_cell("## Model-separated metric heatmaps"),
        code_cell(
            """
models_with_rows = sorted(df.loc[(df["status"] == "ok") & df["delta"].notna(), "model"].unique())
height = max(4.5, 1.75 * max(len(models_with_rows), 1))
fig, axes = plt.subplots(max(len(models_with_rows), 1), 1, figsize=(13, height), constrained_layout=True)
plot_secondary_per_model_heatmaps(df, axes=axes, title=f"{study}: paired deltas separated by model")
plt.show()
"""
        ),
        markdown_cell("## Per-model sensitivity ranking"),
        code_cell(
            """
measured = df[(df["status"] == "ok") & df["delta"].notna() & (df["n_pairs"] > 0)].copy()
ranking = (
    measured.assign(abs_delta=measured["delta"].abs())
    .groupby(["lane", "model"], dropna=False)
    .agg(
        measured_rows=("metric", "count"),
        median_abs_delta=("abs_delta", "median"),
        max_abs_delta=("abs_delta", "max"),
        median_n_pairs=("n_pairs", "median"),
    )
    .reset_index()
    .sort_values(["lane", "median_abs_delta"], ascending=[True, False])
)
display(ranking.round(4))
"""
        ),
        markdown_cell(
            """
## Interpretation

Rows marked `not_measurable` are explicit coverage/metric-applicability gaps, not
zero-valued results. Treat CIs crossing zero as descriptive only. Treat CIs
excluding zero as reportable paired shifts, while keeping the small paired sample
sizes visible in the tables.
"""
        ),
    ]
    return nb


def main() -> int:
    for arm in ARM_NOTEBOOKS:
        folder = Path(arm["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        for study, study_title in STUDIES.items():
            path = folder / str(arm["filename_template"]).format(study=study)
            nb = build_notebook(
                study=study,
                title=str(arm["title_template"]).format(title=study_title),
                lanes=list(arm["lanes"]),
                description=str(arm["description"]),
            )
            nbf.write(nb, path)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
