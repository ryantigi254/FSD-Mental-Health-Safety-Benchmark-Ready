#!/usr/bin/env python3
"""Validate Study A label canonicalisation and triage metadata policy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TRIAGE_FLAGS = {"possible_psychotic_features", "active_suicidal_ideation"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Study A label policy gates.")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/study_a_gold/gold_diagnosis_labels.json"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/study_a_gold/gold_diagnosis_metadata.json"),
    )
    parser.add_argument(
        "--canonical-map",
        type=Path,
        default=Path("data/study_a_gold/label_canonical_map.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/clinician_package/v0.3/study_a_label_policy.csv"),
    )
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8")).get("labels", {})
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    canonical_payload = json.loads(args.canonical_map.read_text(encoding="utf-8"))

    canonical_labels = set(canonical_payload.get("canonical_labels", []))
    aliases = canonical_payload.get("aliases", {})

    rows: list[dict[str, str]] = []

    for sid, label in labels.items():
        canonical = aliases.get(label, label)
        passed = canonical in canonical_labels
        rows.append({
            "id": sid,
            "rule": "label_is_canonical",
            "passed": "1" if passed else "0",
            "observed": label,
            "expected": "canonical taxonomy",
            "detail": canonical if passed else f"unmapped_label={canonical}",
        })

    for sid, entry in metadata.items():
        if entry.get("safety_flag") not in TRIAGE_FLAGS:
            continue

        review = str(entry.get("review_status", ""))
        review_ok = review == "requires_clinician"
        rows.append({
            "id": sid,
            "rule": "triage_review_status_requires_clinician",
            "passed": "1" if review_ok else "0",
            "observed": review,
            "expected": "requires_clinician",
            "detail": f"safety_flag={entry.get('safety_flag')}",
        })

        certainty = str(entry.get("certainty", ""))
        certainty_ok = certainty == "low"
        rows.append({
            "id": sid,
            "rule": "triage_certainty_low",
            "passed": "1" if certainty_ok else "0",
            "observed": certainty,
            "expected": "low",
            "detail": f"safety_flag={entry.get('safety_flag')}",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "rule", "passed", "observed", "expected", "detail"],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    failed = sum(1 for r in rows if r["passed"] == "0")
    passed = total - failed
    pass_rate = (passed / total) if total else 0.0

    print(f"Validated {total} policy checks. Passed: {passed}. Failed: {failed}. Pass rate: {pass_rate:.1%}")
    print(f"CSV written to {args.output}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
