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

STUDY_LENSES = {
    "study_a": {
        "question": "diagnostic accuracy and reasoning quality under the arm change",
        "primary_metrics": ["acc_cot", "step_f1"],
        "context_metrics": ["faithfulness_gap", "acc_early"],
        "interpretation": (
            "For Study A, read the secondary arm as a perturbation check on the "
            "core diagnosis/reasoning result. Accuracy and Step-F1 are the "
            "primary lens; faithfulness gap and early accuracy are supporting "
            "diagnostics where the branch metric can measure them."
        ),
    },
    "study_a_bias": {
        "question": "whether the arm change alters silent-bias behaviour",
        "primary_metrics": ["silent_bias_rate"],
        "context_metrics": [],
        "interpretation": (
            "For Study A Bias, the relevant outcome is silent-bias rate. A "
            "zero delta here means the paired arm did not change the measured "
            "silent-bias rate for the shared cases."
        ),
    },
    "study_b": {
        "question": "sycophancy and evidence/stance behaviour under the arm change",
        "primary_metrics": ["sycophancy_probability", "injected_agreement_rate", "turn_of_flip_proxy"],
        "context_metrics": ["control_agreement_rate"],
        "interpretation": (
            "For Study B, the central question is whether the arm changes "
            "sycophancy-like agreement and flip behaviour. Agreement-rate "
            "metrics are relevant as stance diagnostics, not as generic model "
            "quality scores."
        ),
    },
    "study_b_multi_turn": {
        "question": "multi-turn softening and flip dynamics under the arm change",
        "primary_metrics": ["sycophancy_auc", "turn_of_flip", "soften_before_flip"],
        "context_metrics": ["stance_shift_slope"],
        "interpretation": (
            "For Study B Multi-turn, the relevant view is temporal: whether the "
            "arm changes accumulated sycophancy, the turn of flip, or softening "
            "before a flip. Stance-shift slope is a supporting trend diagnostic."
        ),
    },
    "study_c": {
        "question": "longitudinal recall and conflict behaviour under the arm change",
        "primary_metrics": ["entity_recall_t10", "knowledge_conflict_rate"],
        "context_metrics": [],
        "interpretation": (
            "For Study C, the relevant outcomes are entity recall and NLI-backed "
            "knowledge-conflict rate. Treat conflict deltas as the safety signal "
            "and recall deltas as the continuity/retention signal."
        ),
    },
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
    load_reasoning_control_coverage,
    load_reasoning_control_flat,
    load_secondary_coverage,
    load_secondary_flat,
    plot_reasoning_adherence_tiles,
    plot_reasoning_arm_values,
    plot_reasoning_control_delta_ci,
    plot_reasoning_endpoint_scatter,
    plot_secondary_arm_values,
    plot_secondary_coverage_heatmap,
    plot_secondary_delta_ci,
    plot_secondary_model_heatmap,
    plot_secondary_per_model_heatmaps,
    reasoning_control_headline,
    secondary_metric_headline,
    secondary_missing_table,
    secondary_not_measurable_table,
    secondary_saturation_notes,
    secondary_threshold_audit,
    setup_notebook_style,
)

setup_notebook_style()
runtime_root = find_runtime_root()
metric_root = runtime_root / "metric-results" / "secondary_branch_metrics"
reasoning_root = runtime_root / "metric-results" / "reasoning_control"
study = "{study}"
lanes = {lanes!r}
primary_metrics = {primary_metrics!r}
context_metrics = {context_metrics!r}
flat_all = load_secondary_flat(metric_root)
coverage_all = load_secondary_coverage(metric_root)
reasoning_all = load_reasoning_control_flat(reasoning_root)
reasoning_coverage_all = load_reasoning_control_coverage(reasoning_root)
df = flat_all[(flat_all["study"] == study) & (flat_all["lane"].isin(lanes))].copy()
coverage = coverage_all[(coverage_all["study"] == study) & (coverage_all["lane"].isin(lanes))].copy()
reasoning_df = reasoning_all[(reasoning_all["study"] == study) & (reasoning_all["lane"].isin(lanes))].copy()
reasoning_coverage = reasoning_coverage_all[(reasoning_coverage_all["study"] == study) & (reasoning_coverage_all["lane"].isin(lanes))].copy()
primary_df = df[df["metric"].isin(primary_metrics)].copy()
context_df = df[df["metric"].isin(context_metrics)].copy()
print("Runtime root:", runtime_root)
print("Rows:", len(df), "Coverage rows:", len(coverage))
print("Reasoning-control rows:", len(reasoning_df), "Coverage rows:", len(reasoning_coverage))
print("Primary study metrics:", ", ".join(primary_metrics))
if context_metrics:
    print("Context diagnostics:", ", ".join(context_metrics))
"""


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook(*, study: str, title: str, lanes: list[str], description: str) -> nbf.NotebookNode:
    lens = STUDY_LENSES[study]
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        markdown_cell(f"# {title}"),
        markdown_cell(description),
        markdown_cell(
            f"""
## Study Lens

This notebook treats the secondary metric arm as a perturbation of the study's
own endpoint: **{lens["question"]}**.

Primary plotted metrics: `{", ".join(lens["primary_metrics"])}`.

Supporting diagnostics: `{", ".join(lens["context_metrics"]) if lens["context_metrics"] else "none"}`.
"""
        ),
        code_cell(
            SETUP_TEMPLATE.format(
                study=study,
                lanes=lanes,
                primary_metrics=lens["primary_metrics"],
                context_metrics=lens["context_metrics"],
            )
        ),
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
        markdown_cell("## Primary study endpoint deltas"),
        code_cell(
            """
display(secondary_metric_headline(primary_df).round(4))
display(secondary_threshold_audit(primary_df).round(4))
fig, ax = plt.subplots(figsize=(12, max(5, 0.55 * max(len(primary_df), 6))), constrained_layout=True)
plot_secondary_delta_ci(primary_df, ax=ax, title=f"{study}: primary endpoint paired deltas", max_rows=24)
plt.show()
"""
        ),
        markdown_cell("## Primary endpoint base versus variant"),
        code_cell(
            """
fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * max(len(primary_df), 6))), constrained_layout=True)
plot_secondary_arm_values(primary_df, ax=ax, title=f"{study}: primary endpoint base and variant values", max_rows=24)
plt.show()
"""
        ),
        markdown_cell("## Reasoning and output control"),
        code_cell(
            """
display(reasoning_control_headline(reasoning_df).round(4))
display(reasoning_df[reasoning_df["status"].fillna("") != "ok"].sort_values(["lane", "model", "variant", "metric"]))

fig, ax = plt.subplots(figsize=(12, max(5, 0.48 * max(len(reasoning_df), 8))), constrained_layout=True)
plot_reasoning_control_delta_ci(
    reasoning_df,
    ax=ax,
    title=f"{study}: visible reasoning and output-control paired deltas",
    metrics=["visible_reasoning_tokens", "output_tokens", "reasoning_share", "final_answer_marker_present"],
    max_rows=26,
)
plt.show()

fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * max(len(reasoning_df), 8))), constrained_layout=True)
plot_reasoning_arm_values(
    reasoning_df,
    ax=ax,
    title=f"{study}: visible reasoning/output base versus variant",
    metrics=["visible_reasoning_tokens", "output_tokens", "reasoning_share"],
    max_rows=22,
)
plt.show()
"""
        ),
        markdown_cell("## Marker and control-adherence audit"),
        code_cell(
            """
fig, ax = plt.subplots(figsize=(13, 5.6), constrained_layout=True)
plot_reasoning_adherence_tiles(reasoning_df, ax=ax, title=f"{study}: marker and control-adherence rates")
plt.show()

fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
plot_reasoning_endpoint_scatter(
    reasoning_df,
    ax=ax,
    title=f"{study}: reasoning-share shift versus primary endpoint shift",
    metric="reasoning_share",
)
plt.show()
"""
        ),
        markdown_cell("## Baseline and saturation notes"),
        code_cell(
            """
saturation = secondary_saturation_notes(primary_df, coverage)
if saturation.empty:
    print("No measured ceiling/floor saturation rows detected for the primary endpoint metrics.")
else:
    display(saturation.round(4))

if study == "study_c" and "controllability" in lanes:
    measured_models = sorted(coverage.loc[coverage["status"].fillna("") == "ok", "model"].unique())
    missing_models = sorted(coverage.loc[coverage["status"].fillna("") != "ok", "model"].unique())
    print("Study C controllability note:")
    print("Measured models:", ", ".join(measured_models) if measured_models else "none")
    print("Missing-cache models:", ", ".join(missing_models) if missing_models else "none")
    print(
        "Interpret neutral Study C deltas as ceiling/floor behaviour on the measurable subset "
        "when recall is saturated at 1.0 or conflict is saturated at 0.0; do not overclaim broad robustness."
    )
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
plot_secondary_model_heatmap(primary_df, ax=ax, title=f"{study}: primary metric median paired delta by model")
plt.show()
"""
        ),
        markdown_cell("## Model-separated metric heatmaps"),
        code_cell(
            """
models_with_rows = sorted(primary_df.loc[(primary_df["status"] == "ok") & primary_df["delta"].notna(), "model"].unique())
height = max(4.5, 1.75 * max(len(models_with_rows), 1))
fig, axes = plt.subplots(max(len(models_with_rows), 1), 1, figsize=(13, height), constrained_layout=True)
plot_secondary_per_model_heatmaps(primary_df, axes=axes, title=f"{study}: primary paired deltas separated by model")
plt.show()
"""
        ),
        markdown_cell("## Supporting diagnostics"),
        code_cell(
            """
if context_df.empty:
    print(
        "No additional supporting diagnostics are defined for this study arm. "
        "Interpret this notebook through the primary endpoint metrics above, "
        "plus coverage, n-pairs, and not-measurable reasons."
    )
else:
    display(secondary_metric_headline(context_df).round(4))
    fig, ax = plt.subplots(figsize=(12, max(4, 0.55 * max(len(context_df), 4))), constrained_layout=True)
    plot_secondary_delta_ci(context_df, ax=ax, title=f"{study}: supporting diagnostic deltas", max_rows=18)
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

{lens["interpretation"]}
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
