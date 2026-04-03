#!/usr/bin/env python3
"""
Generate the dedicated pairwise notebook family.
"""

from __future__ import annotations

import json
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK_ROOT = RUNTIME_ROOT / "notebooks" / "pairwise"
REPORT_ROOT = RUNTIME_ROOT / "metric-results" / "pairwise" / "reports"

CORE_SLICES = [
    "study_a",
    "study_a_bias",
    "study_b",
    "study_b_multiturn",
    "study_c",
]
CONTROLLABILITY_SLICES = [
    "study_a_controllability",
    "study_a_bias_controllability",
    "study_b_controllability",
    "study_b_multiturn_controllability",
    "study_c_controllability",
]
INVARIANCE_SLICES = [
    "invariance",
    "invariance_under_control",
    "control_under_invariance",
]


def main() -> int:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)

    for slice_id in CORE_SLICES:
        _write_notebook(
            NOTEBOOK_ROOT / f"core_{slice_id}_pairwise.ipynb",
            title=f"Pairwise Core Analysis: {slice_id}",
            slice_id=slice_id,
            report_glob=f"{REPORT_ROOT}/pairwise_{slice_id}_v1/{slice_id}__pairwise_secondary_results.json",
        )

    for slice_id in CONTROLLABILITY_SLICES:
        _write_notebook(
            NOTEBOOK_ROOT / f"controllability_{slice_id}_pairwise.ipynb",
            title=f"Pairwise Controllability Analysis: {slice_id}",
            slice_id=slice_id,
            report_glob=f"{REPORT_ROOT}/pairwise_{slice_id}_v1/{slice_id}__pairwise_secondary_results.json",
        )

    for slice_id in INVARIANCE_SLICES:
        _write_notebook(
            NOTEBOOK_ROOT / f"invariance_{slice_id}_pairwise.ipynb",
            title=f"Pairwise Invariance Analysis: {slice_id}",
            slice_id=slice_id,
            report_glob=f"{REPORT_ROOT}/pairwise_{slice_id}_v1/{slice_id}__pairwise_secondary_results.json",
        )

    _write_summary_notebook(
        NOTEBOOK_ROOT / "pairwise_cross_layer_summary.ipynb",
        title="Pairwise Cross-Layer Summary",
        summary_mode="cross_layer",
    )
    _write_summary_notebook(
        NOTEBOOK_ROOT / "pairwise_judge_diagnostics.ipynb",
        title="Pairwise Judge Diagnostics",
        summary_mode="judge_diagnostics",
    )
    return 0


def _write_notebook(path: Path, *, title: str, slice_id: str, report_glob: str) -> None:
    notebook = {
        "cells": [
            _markdown_cell(f"# {title}\n\nThis notebook reads the canonical pairwise report for `{slice_id}` only."),
            _code_cell(
                "from pathlib import Path\n"
                "import json\n\n"
                f"report_path = Path(r'''{report_glob}''')\n"
                "if not report_path.exists():\n"
                "    print(f'Report not found yet: {report_path}')\n"
                "else:\n"
                "    report = json.loads(report_path.read_text(encoding='utf-8'))\n"
                "    print('run_id:', report.get('run_id'))\n"
                "    print('slice_id:', report.get('slice_id'))\n"
                "    print('pooled_complete:', report.get('pooled_complete'))\n"
                "    print('boundary_notes:', report.get('boundary_notes', []))\n"
            ),
            _code_cell(
                "if 'report' in globals():\n"
                "    print('\\nPer-judge summaries')\n"
                "    for judge_id, payload in report.get('per_judge', {}).items():\n"
                "        print(judge_id, payload.get('summary', {}))\n"
            ),
            _code_cell(
                "if 'report' in globals():\n"
                "    print('\\nPooled win rates')\n"
                "    for row in report.get('pooled', {}).get('win_rates', [])[:20]:\n"
                "        print(row)\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def _write_summary_notebook(path: Path, *, title: str, summary_mode: str) -> None:
    notebook = {
        "cells": [
            _markdown_cell(f"# {title}\n\nReads canonical reports only."),
            _code_cell(
                "from pathlib import Path\n"
                "import json\n\n"
                f"report_root = Path(r'''{REPORT_ROOT}''')\n"
                "report_paths = sorted(report_root.glob('*/*__pairwise_secondary_results.json'))\n"
                "print('reports found:', len(report_paths))\n"
                "reports = [json.loads(path.read_text(encoding='utf-8')) for path in report_paths]\n"
            ),
            _code_cell(
                "if reports:\n"
                "    for report in reports:\n"
                "        print(report.get('slice_id'), report.get('pooled_complete'), report.get('case_manifest_status'))\n"
            ),
            _code_cell(
                "if reports and '" + summary_mode + "' == 'judge_diagnostics':\n"
                "    for report in reports:\n"
                "        print('\\n', report.get('slice_id'))\n"
                "        print(report.get('judge_agreement', {}))\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def _markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


if __name__ == "__main__":
    raise SystemExit(main())
