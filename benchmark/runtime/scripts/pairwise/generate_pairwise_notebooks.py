#!/usr/bin/env python3
"""Generate the dedicated pairwise notebook family."""

from __future__ import annotations

import json
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK_ROOT = RUNTIME_ROOT / "notebooks" / "pairwise"
REPORT_ROOT = RUNTIME_ROOT / "metric-results" / "pairwise" / "reports"
JUDGE_AUDIT_REPORT_ROOT = (
    RUNTIME_ROOT / "metric-results" / "pairwise" / "judge_audit" / "reports"
)
PAIRWISE_RUN_VERSION = "v3"

CORE_NOTEBOOKS = [
    (
        "study_a",
        NOTEBOOK_ROOT / "core" / "core_study_a_pairwise.ipynb",
    ),
    (
        "study_a_bias",
        NOTEBOOK_ROOT / "core" / "core_study_a_bias_pairwise.ipynb",
    ),
    (
        "study_b",
        NOTEBOOK_ROOT / "core" / "core_study_b_pairwise.ipynb",
    ),
    (
        "study_b_multiturn",
        NOTEBOOK_ROOT / "core" / "core_study_b_multiturn_pairwise.ipynb",
    ),
    (
        "study_c",
        NOTEBOOK_ROOT / "core" / "core_study_c_pairwise.ipynb",
    ),
]
STUDY_PARTS = [
    "study_a",
    "study_a_bias",
    "study_b",
    "study_b_multiturn",
    "study_c",
]
CONTROLLABILITY_NOTEBOOKS = [
    (
        "study_a_controllability",
        NOTEBOOK_ROOT
        / "controllability"
        / "study_a_controllability_pairwise.ipynb",
    ),
    (
        "study_a_bias_controllability",
        NOTEBOOK_ROOT
        / "controllability"
        / "study_a_bias_controllability_pairwise.ipynb",
    ),
    (
        "study_b_controllability",
        NOTEBOOK_ROOT
        / "controllability"
        / "study_b_controllability_pairwise.ipynb",
    ),
    (
        "study_b_multiturn_controllability",
        NOTEBOOK_ROOT
        / "controllability"
        / "study_b_multiturn_controllability_pairwise.ipynb",
    ),
    (
        "study_c_controllability",
        NOTEBOOK_ROOT
        / "controllability"
        / "study_c_controllability_pairwise.ipynb",
    ),
]
INVARIANCE_ARMS = [
    (
        "invariance",
        "invariance",
        "invariance_pairwise.ipynb",
    ),
    (
        "invariance_under_control",
        "invariance-controlability",
        "invariance_under_control_pairwise.ipynb",
    ),
    (
        "control_under_invariance",
        "controlability-invariance",
        "control_under_invariance_pairwise.ipynb",
    ),
]


def main() -> int:
    NOTEBOOK_ROOT.mkdir(parents=True, exist_ok=True)

    _write_slice_notebooks("Core", CORE_NOTEBOOKS)
    _write_slice_notebooks("Controllability", CONTROLLABILITY_NOTEBOOKS)
    _write_invariance_notebooks()

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
    _write_summary_notebook(
        NOTEBOOK_ROOT / "pairwise_stacked_execution_summary.ipynb",
        title="Pairwise Stacked Execution Summary",
        summary_mode="stacked_execution",
    )
    _write_summary_notebook(
        NOTEBOOK_ROOT / "pairwise_escalation_diagnostics.ipynb",
        title="Pairwise Escalation and Disagreement Diagnostics",
        summary_mode="escalation",
    )
    _write_judge_audit_notebook(
        NOTEBOOK_ROOT / "pairwise_judge_audit_summary.ipynb",
        title="Pairwise Judge Audit Summary",
    )
    return 0


def _write_slice_notebooks(title_prefix: str, notebook_specs: list[tuple[str, Path]]) -> None:
    for slice_id, output_path in notebook_specs:
        _write_notebook(
            output_path,
            title=f"Pairwise {title_prefix} Analysis: {slice_id}",
            slice_id=slice_id,
            report_glob=f"{REPORT_ROOT}/pairwise_{slice_id}_{PAIRWISE_RUN_VERSION}/{slice_id}__pairwise_secondary_results.json",
        )


def _write_invariance_notebooks() -> None:
    for study_id in STUDY_PARTS:
        for arm_slice_id, folder_name, notebook_name in INVARIANCE_ARMS:
            _write_invariance_notebook(
                NOTEBOOK_ROOT / "invariance" / study_id / folder_name / notebook_name,
                arm_slice_id=arm_slice_id,
                study_id=study_id,
                title=f"Pairwise Invariance Analysis: {study_id} / {arm_slice_id}",
            )


def _write_notebook(path: Path, *, title: str, slice_id: str, report_glob: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            _markdown_cell(
                f"# {title}\n\nThis notebook reads the canonical pairwise report for `{slice_id}` only."
            ),
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
                "    print('run_mode:', report.get('run_mode'))\n"
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
                "    print('\\nExecution summary')\n"
                "    print(report.get('execution_summary', {}))\n"
            ),
            _code_cell(
                "if 'report' in globals():\n"
                "    print('\\nPooled win rates (secondary view)')\n"
                "    for row in report.get('pooled', {}).get('win_rates', [])[:20]:\n"
                "        print(row)\n"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def _write_invariance_notebook(
    path: Path,
    *,
    arm_slice_id: str,
    study_id: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report_glob = (
        f"{REPORT_ROOT}/pairwise_{arm_slice_id}_{PAIRWISE_RUN_VERSION}/"
        f"{arm_slice_id}__pairwise_secondary_results.json"
    )
    manifest_path = (
        RUNTIME_ROOT
        / "metric-results"
        / "pairwise"
        / "manifests"
        / f"{arm_slice_id}_case_manifest.json"
    )
    notebook = {
        "cells": [
            _markdown_cell(
                f"# {title}\n\nThis notebook inspects the `{study_id}` slice inside the `{arm_slice_id}` invariance arm."
            ),
            _code_cell(
                "from collections import Counter\n"
                "from pathlib import Path\n"
                "import json\n\n"
                f"arm_slice_id = '{arm_slice_id}'\n"
                f"study_id = '{study_id}'\n"
                f"report_path = Path(r'''{report_glob}''')\n"
                f"manifest_path = Path(r'''{manifest_path}''')\n\n"
                "if report_path.exists():\n"
                "    report = json.loads(report_path.read_text(encoding='utf-8'))\n"
                "    print('report_run_id:', report.get('run_id'))\n"
                "    print('report_slice_id:', report.get('slice_id'))\n"
                "    print('report_run_mode:', report.get('run_mode'))\n"
                "    print('report_pooled_complete:', report.get('pooled_complete'))\n"
                "else:\n"
                "    print(f'Report not found yet: {report_path}')\n\n"
                "if manifest_path.exists():\n"
                "    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))\n"
                "    cases = [\n"
                "        case\n"
                "        for case in manifest.get('cases', [])\n"
                "        if case.get('case_meta', {}).get('study') == study_id\n"
                "    ]\n"
                "    print('manifest_slice_id:', manifest.get('slice_id'))\n"
                "    print('study_id:', study_id)\n"
                "    print('matching_cases:', len(cases))\n"
                "    print('manifest_boundary_notes:', manifest.get('boundary_notes', []))\n"
                "else:\n"
                "    print(f'Manifest not found: {manifest_path}')\n"
            ),
            _code_cell(
                "if 'cases' in globals():\n"
                "    print('\\nModel counts in matching cases')\n"
                "    print(Counter(case.get('case_meta', {}).get('model_id') for case in cases))\n"
            ),
            _code_cell(
                "if 'cases' in globals():\n"
                "    print('\\nSample matching cases')\n"
                "    for case in cases[:10]:\n"
                "        print(case.get('case_id'), case.get('case_meta', {}))\n"
            ),
            _code_cell(
                "if 'report' in globals():\n"
                "    print('\\nPer-judge summaries')\n"
                "    for judge_id, payload in report.get('per_judge', {}).items():\n"
                "        print(judge_id, payload.get('summary', {}))\n"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
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
                "        summary = report.get('execution_summary', {})\n"
                "        print(report.get('slice_id'), report.get('pooled_complete'), report.get('case_manifest_status'), summary.get('routine_two_judge_results', {}), summary.get('escalated_four_judge_results', {}))\n"
            ),
            _code_cell(
                "if reports and '" + summary_mode + "' == 'judge_diagnostics':\n"
                "    for report in reports:\n"
                "        print('\\n', report.get('slice_id'))\n"
                "        print(report.get('judge_agreement', {}))\n"
            ),
            _code_cell(
                "if reports and '" + summary_mode + "' == 'stacked_execution':\n"
                "    for report in reports:\n"
                "        print('\\n', report.get('slice_id'))\n"
                "        print(report.get('execution_summary', {}))\n"
            ),
            _code_cell(
                "if reports and '" + summary_mode + "' == 'escalation':\n"
                "    for report in reports:\n"
                "        summary = report.get('execution_summary', {})\n"
                "        print('\\n', report.get('slice_id'))\n"
                "        print('escalation_summary:', summary.get('escalation_summary', {}))\n"
                "        print('persistent_disagreement_cases:', summary.get('persistent_disagreement_cases', [])[:10])\n"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def _write_judge_audit_notebook(path: Path, *, title: str) -> None:
    notebook = {
        "cells": [
            _markdown_cell(f"# {title}\n\nReads canonical judge-audit reports only."),
            _code_cell(
                "from pathlib import Path\n"
                "import json\n\n"
                f"report_root = Path(r'''{JUDGE_AUDIT_REPORT_ROOT}''')\n"
                "report_paths = sorted(report_root.glob('*.json'))\n"
                "print('judge-audit reports found:', len(report_paths))\n"
                "reports = [json.loads(path.read_text(encoding='utf-8')) for path in report_paths]\n"
            ),
            _code_cell(
                "if reports:\n"
                "    for report in reports:\n"
                "        print('\\nreport:', report.get('report_name'))\n"
                "        for judge_id, metrics in report.get('per_judge', {}).items():\n"
                "            print(judge_id, {key: value for key, value in metrics.items() if key != 'item_summaries'})\n"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
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
