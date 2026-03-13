#!/usr/bin/env python3
"""Generate distribution summaries for controllability split variants."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyse_clinical_distribution import classify_category, classify_severity


BASE = Path(__file__).resolve().parent.parent
DEFAULT_SMALL_DATA = BASE / "data" / "controllability_splits"
DEFAULT_OUT = Path(__file__).resolve().parent / "controllability_distribution_analysis.json"

STUDY_LABELS = {
    "A": "Study A Bias",
    "B": "Study B Multi-turn",
    "C": "Study C",
    "D": "Study A",
    "E": "Study B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse controllability split distributions.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_SMALL_DATA,
        help="Root directory containing the controllability split artefacts.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUT,
        help="Path for the generated analysis JSON.",
    )
    parser.add_argument(
        "--profile",
        choices=["combined", "study_a", "study_b"],
        default="combined",
        help="Which controllability-oriented profile to analyse.",
    )
    parser.add_argument(
        "--suite-label",
        default="",
        help="Optional human-readable label for the suite variant, such as 'Small Scale' or 'Scaled Large Resolved'.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalise_label(text: str | None) -> str:
    return str(text or "Unknown").strip().lower()


def _normalise_items(payload: Any, preferred_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unable to normalise payload using keys={preferred_keys}")


def load_study_a_bias(data_root: Path) -> list[dict[str, str]]:
    raw = _load_json(data_root / "study_a_bias_controllability_test.json")
    rows: list[dict[str, str]] = []
    for case in _normalise_items(raw, ("cases", "samples", "items")):
        rows.append(
            {
                "id": str(case.get("id") or ""),
                "study": "A",
                "condition": _normalise_label(case.get("bias_label")),
            }
        )
    return rows


def load_study_a_main(data_root: Path, *, study_key: str = "A") -> list[dict[str, str]]:
    raw = _load_json(data_root / "study_a_controllability_test.json")
    labels_raw = _load_json(data_root / "ctrl_gold_diagnosis_labels.json")
    labels = labels_raw.get("labels", {}) if isinstance(labels_raw, dict) else {}

    rows: list[dict[str, str]] = []
    for sample in _normalise_items(raw, ("samples", "cases", "items")):
        sample_id = str(sample.get("id") or "")
        rows.append(
            {
                "id": sample_id,
                "study": study_key,
                "condition": _normalise_label(labels.get(sample_id)),
            }
        )
    return rows


def load_study_b_multi_turn(data_root: Path) -> list[dict[str, str]]:
    raw = _load_json(data_root / "study_b_multi_turn_controllability_test.json")
    rows: list[dict[str, str]] = []
    for case in _normalise_items(raw, ("cases", "multi_turn_cases", "items")):
        metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
        condition = metadata.get("condition_phrase") or case.get("gold_answer")
        rows.append(
            {
                "id": str(case.get("id") or ""),
                "study": "B",
                "condition": _normalise_label(condition),
            }
        )
    return rows


def load_study_b_main(data_root: Path, *, study_key: str = "B") -> list[dict[str, str]]:
    raw = _load_json(data_root / "study_b_controllability_test.json")
    rows: list[dict[str, str]] = []
    for case in _normalise_items(raw, ("samples", "items")):
        rows.append(
            {
                "id": str(case.get("id") or ""),
                "study": study_key,
                "condition": _normalise_label(case.get("gold_answer")),
            }
        )
    return rows


def load_study_c(data_root: Path) -> list[dict[str, str]]:
    raw = _load_json(data_root / "study_c_controllability_test.json")
    rows: list[dict[str, str]] = []
    for case in _normalise_items(raw, ("cases", "samples", "items")):
        critical_entities = case.get("critical_entities") or []
        condition = critical_entities[0] if critical_entities else "unknown"
        rows.append(
            {
                "id": str(case.get("id") or ""),
                "study": "C",
                "condition": _normalise_label(condition),
            }
        )
    return rows


def classify_controllability_category(study: str, condition: str) -> str:
    if study == "A":
        return "Bias / Adversarial"
    return classify_category(condition)


def classify_controllability_severity(study: str, condition: str) -> str:
    if study == "A":
        return "Unknown"
    return classify_severity(condition)


def _base_radar_config() -> dict[str, Any]:
    return {
        "exclude_categories": ["Bias / Adversarial", "Other", "Unclassified"],
        "max_value": 100,
        "tick_step": 20,
        "tick_label_mode": "fraction",
        "manual_tick_labels": True,
        "tick_label_angle": -8,
    }


def _build_analysis_from_rows(
    rows_by_study: dict[str, list[dict[str, str]]],
    *,
    profile: str,
    profile_label: str,
    suite_label: str,
    study_labels: dict[str, str],
    category_fn,
    severity_fn,
    radar: dict[str, Any] | None = None,
    readme_notes: list[str] | None = None,
    omit_figures: list[str] | None = None,
) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_study.values() for row in rows]

    category_counts = Counter()
    category_per_study = defaultdict(Counter)
    condition_counts = Counter()
    condition_per_study = defaultdict(Counter)
    severity_counts = Counter()
    severity_per_study = defaultdict(Counter)

    for row in all_rows:
        study = row["study"]
        condition = row["condition"]
        category = category_fn(study, condition)
        severity = severity_fn(study, condition)
        category_counts[category] += 1
        category_per_study[study][category] += 1
        condition_counts[condition] += 1
        condition_per_study[study][condition] += 1
        severity_counts[severity] += 1
        severity_per_study[study][severity] += 1

    total_rows = len(all_rows)
    study_ids = list(rows_by_study)
    primary_study_for_condition = {
        condition: max(
            study_ids,
            key=lambda study: condition_per_study[study][condition],
        )
        for condition in condition_counts
    }

    analysis = {
        "profile": profile,
        "profile_label": profile_label,
        "suite_label": suite_label,
        "study_labels": study_labels,
        "radar": radar or {},
        "readme_notes": readme_notes or [],
        "omit_figures": omit_figures or [],
        "total_samples": total_rows,
        "per_study": {study: len(rows_by_study[study]) for study in study_ids},
        "diagnostic_categories": {
            category: {
                "total": count,
                "pct": round((100.0 * count / total_rows), 2) if total_rows else 0.0,
                "per_study": {
                    study: category_per_study[study][category]
                    for study in study_ids
                },
            }
            for category, count in category_counts.most_common()
        },
        "specific_conditions": {
            condition: {
                "total": count,
                "category": category_fn(primary_study_for_condition[condition], condition),
                "severity": severity_fn(primary_study_for_condition[condition], condition),
                "per_study": {
                    study: condition_per_study[study][condition]
                    for study in study_ids
                },
            }
            for condition, count in condition_counts.most_common()
        },
        "severity": {
            severity: {
                "total": count,
                "pct": round((100.0 * count / total_rows), 2) if total_rows else 0.0,
                "per_study": {
                    study: severity_per_study[study][severity]
                    for study in study_ids
                },
            }
            for severity, count in severity_counts.most_common()
        },
        "therapeutic_modalities": {},
        "unique_conditions_per_study": {
            study: len({row["condition"] for row in rows_by_study[study]})
            for study in study_ids
        },
        "total_unique_conditions": len(condition_counts),
    }
    return analysis


def _default_suite_label(data_root: Path) -> str:
    mapping = {
        "controllability_splits": "Small Scale",
        "controllability_splits_large_resolved": "Scaled Large Resolved",
    }
    return mapping.get(data_root.name, data_root.name.replace("_", " ").title())


def build_analysis(data_root: Path, profile: str = "combined", suite_label: str = "") -> dict[str, Any]:
    resolved_suite_label = suite_label or _default_suite_label(data_root)

    if profile == "study_a":
        return _build_analysis_from_rows(
            {"A": load_study_a_main(data_root)},
            profile="controllability_study_a",
            profile_label=f"Controllability Study A — {resolved_suite_label}",
            suite_label=resolved_suite_label,
            study_labels={"A": "Study A"},
            category_fn=lambda _study, condition: classify_category(condition),
            severity_fn=lambda _study, condition: classify_severity(condition),
            readme_notes=[
                f"This profile isolates the Study A controllability split for the {resolved_suite_label.lower()} suite.",
            ],
            omit_figures=["fig2_per_study_breakdown", "fig3_therapeutic_modalities"],
        )

    if profile == "study_b":
        return _build_analysis_from_rows(
            {"B": load_study_b_main(data_root)},
            profile="controllability_study_b",
            profile_label=f"Controllability Study B — {resolved_suite_label}",
            suite_label=resolved_suite_label,
            study_labels={"B": "Study B"},
            category_fn=lambda _study, condition: classify_category(condition),
            severity_fn=lambda _study, condition: classify_severity(condition),
            readme_notes=[
                f"This profile isolates the Study B single-turn controllability split for the {resolved_suite_label.lower()} suite.",
            ],
            omit_figures=["fig2_per_study_breakdown", "fig3_therapeutic_modalities"],
        )

    combined = _build_analysis_from_rows(
        {
            "A": load_study_a_bias(data_root),
            "B": load_study_b_multi_turn(data_root),
            "C": load_study_c(data_root),
            "D": load_study_a_main(data_root, study_key="D"),
            "E": load_study_b_main(data_root, study_key="E"),
        },
        profile="controllability",
        profile_label=f"Controllability Splits — {resolved_suite_label}",
        suite_label=resolved_suite_label,
        study_labels=STUDY_LABELS,
        category_fn=classify_controllability_category,
        severity_fn=classify_controllability_severity,
        radar=_base_radar_config(),
        readme_notes=[
            "Study A Bias contributes adversarial labels rather than gold clinical diagnoses, so category charts include a dedicated `Bias / Adversarial` bucket.",
            "Severity is marked as `Unknown` where the controllability suite does not expose a defensible clinical-severity mapping.",
        ],
        omit_figures=["fig3_therapeutic_modalities"],
    )
    combined["radar_series"] = [
        {"key": "A", "label": "Study A Bias", "style": "study_a_bias_dotted"},
        {"key": "overall", "label": "Overall", "style": "overall"},
        {"key": "E", "label": "Study B", "style": "study_b_controllability"},
        {"key": "B", "label": "Study B Multi-turn", "style": "study_b_multiturn_controllability"},
        {"key": "C", "label": "Study C", "style": "study_c"},
        {"key": "D", "label": "Study A", "style": "study_a"},
    ]
    return combined


def main() -> None:
    args = parse_args()
    analysis = build_analysis(args.data_root, profile=args.profile, suite_label=args.suite_label)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
