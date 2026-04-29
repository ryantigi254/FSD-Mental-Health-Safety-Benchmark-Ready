#!/usr/bin/env python3
"""Generate study-level cross-benchmark arm analysis notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = RUNTIME_ROOT / "notebooks" / "cross_arm"

STUDIES = {
    "study_a": {
        "title": "Study A Cross-Arm Analysis",
        "filename": "study_a_cross_arm_analysis.ipynb",
        "main_metrics": ["faithfulness_gap", "step_f1", "acc_cot", "acc_early"],
        "headline_metric": "faithfulness_gap",
    },
    "study_a_bias": {
        "title": "Study A Bias Cross-Arm Analysis",
        "filename": "study_a_bias_cross_arm_analysis.ipynb",
        "main_metrics": ["silent_bias_rate"],
        "headline_metric": "silent_bias_rate",
    },
    "study_b": {
        "title": "Study B Cross-Arm Analysis",
        "filename": "study_b_cross_arm_analysis.ipynb",
        "main_metrics": [
            "sycophancy_probability",
            "evidence_hallucination",
            "turn_of_flip_proxy",
        ],
        "headline_metric": "sycophancy_probability",
    },
    "study_b_multi_turn": {
        "title": "Study B Multi-turn Cross-Arm Analysis",
        "filename": "study_b_multiturn_cross_arm_analysis.ipynb",
        "main_metrics": [
            "mean_turn_of_flip_raw",
            "never_flip_rate",
            "median_turn_of_flip_raw",
        ],
        "headline_metric": "mean_turn_of_flip_raw",
    },
    "study_c": {
        "title": "Study C Cross-Arm Analysis",
        "filename": "study_c_cross_arm_analysis.ipynb",
        "main_metrics": [
            "entity_recall_t10",
            "knowledge_conflict_rate",
            "continuity_score",
        ],
        "headline_metric": "entity_recall_t10",
    },
}


SETUP_TEMPLATE = """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

NOTEBOOK_UTILS_DIR = Path("benchmark/runtime/notebooks").resolve()
if not NOTEBOOK_UTILS_DIR.exists():
    NOTEBOOK_UTILS_DIR = Path.cwd().resolve().parent
if str(NOTEBOOK_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_UTILS_DIR))

from notebook_utils import (
    SECONDARY_LANES,
    CROSS_ARM_STUDY_CONFIG,
    load_cross_arm_data,
    plot_cross_arm_heatmap,
    plot_cross_arm_trends,
    plot_main_metric_ci,
    plot_secondary_delta_ci,
    secondary_missing_table,
    secondary_threshold_audit,
    find_runtime_root,
    setup_notebook_style,
    summarise_cross_arm_effects,
)

setup_notebook_style()
runtime_root = find_runtime_root()
metric_root = runtime_root / "metric-results"
study = "{study}"
main_metrics = {main_metrics!r}
headline_metric = "{headline_metric}"
main_long, secondary, coverage = load_cross_arm_data(
    metric_root=metric_root,
    runtime_root=runtime_root,
    study=study,
)
secondary_summary = summarise_cross_arm_effects(secondary)
print("Runtime root:", runtime_root)
print("Main rows:", len(main_long), "Secondary rows:", len(secondary), "Coverage rows:", len(coverage))
"""


def markdown_cell(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code_cell(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build_notebook(study: str, config: dict[str, object]) -> nbf.NotebookNode:
    title = str(config["title"])
    main_metrics = list(config["main_metrics"])
    headline_metric = str(config["headline_metric"])

    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    nb.cells = [
        markdown_cell(f"# {title}"),
        code_cell(
            SETUP_TEMPLATE.format(
                study=study,
                main_metrics=main_metrics,
                headline_metric=headline_metric,
            )
        ),
        markdown_cell(
            """
## Provenance and coverage

This notebook joins deterministic main metrics with secondary branch metrics only.
Pairwise judge outputs are intentionally excluded. Secondary Study B and Study C
rows should retain `use_nli=True` and `nli_stride=1` where generation coverage exists.
"""
        ),
        code_cell(
            """
coverage_display = coverage[["lane", "model", "study", "status", "use_nli", "nli_stride", "data_root"]].copy()
display(coverage_display.sort_values(["lane", "model"]))
missing = secondary_missing_table(coverage)
display(missing[["lane", "model", "study", "status"]])
"""
        ),
        markdown_cell("## Main performance"),
        code_cell(
            """
main_table = main_long.sort_values(["metric", "model"]).copy()
display(main_table.round(4))

fig, axes = plt.subplots(
    len(main_metrics),
    1,
    figsize=(11, max(4, 3.2 * len(main_metrics))),
    constrained_layout=True,
)
if len(main_metrics) == 1:
    axes = [axes]
for ax, metric in zip(axes, main_metrics):
    plot_main_metric_ci(
        main_long,
        ax=ax,
        metric=metric,
        title=f"{CROSS_ARM_STUDY_CONFIG[study]['title']}: {metric}",
    )
plt.show()
"""
        ),
        markdown_cell(
            """
## Controllability

The zero line is the no-effect threshold. Rows whose 95% CI crosses zero are
descriptive only; rows whose CI excludes zero are reportable paired shifts.
"""
        ),
        code_cell(
            """
ctrl = secondary[secondary["lane"] == "controllability"].copy()
display(secondary_threshold_audit(ctrl).round(4))
fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * max(len(ctrl), 8))), constrained_layout=True)
plot_secondary_delta_ci(ctrl, ax=ax, title=f"{CROSS_ARM_STUDY_CONFIG[study]['title']}: controllability deltas", max_rows=26)
plt.show()
"""
        ),
        markdown_cell("## Invariance and ctrl-invariance"),
        code_cell(
            """
fig, axes = plt.subplots(1, 2, figsize=(18, max(5, 0.34 * max(len(secondary), 10))), constrained_layout=True)
for ax, lane in zip(axes, ["invariance", "ctrl-invariance"]):
    lane_df = secondary[secondary["lane"] == lane].copy()
    display(secondary_threshold_audit(lane_df).round(4))
    plot_secondary_delta_ci(
        lane_df,
        ax=ax,
        title=f"{CROSS_ARM_STUDY_CONFIG[study]['title']}: {lane} deltas",
        max_rows=22,
    )
plt.show()
"""
        ),
        markdown_cell("## Cross-arm trend view"),
        code_cell(
            """
display(secondary_summary.round(4))

fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
plot_cross_arm_heatmap(
    secondary_summary,
    ax=axes[0],
    title=f"{CROSS_ARM_STUDY_CONFIG[study]['title']}: median absolute secondary delta",
)
plot_cross_arm_trends(
    main_long,
    secondary_summary,
    main_metric=headline_metric,
    ax=axes[1],
    title=f"{headline_metric} vs secondary-arm sensitivity",
)
plt.show()
"""
        ),
        markdown_cell(
            """
## Interpretation

Use the main-performance panel for the baseline result, then treat the secondary
arms as stress-test evidence. Controllability reflects responsiveness to explicit
control. Invariance reflects robustness under harmless perturbation. Ctrl-invariance
asks whether the explicit-control effect itself remains robust under perturbation.
Trend panels are descriptive because the model count is small; rely on paired CIs
before making reportable claims.
"""
        ),
    ]
    return nb


def main() -> int:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    for study, config in STUDIES.items():
        path = NOTEBOOK_ROOT / str(config["filename"])
        nbf.write(build_notebook(study, config), path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
