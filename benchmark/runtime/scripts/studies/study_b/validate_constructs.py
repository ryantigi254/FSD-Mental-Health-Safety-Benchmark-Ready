#!/usr/bin/env python3
"""Deterministic construct checks for Study B single-turn items."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DURATION_MARKERS = (
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "daily",
    "for ",
)

IMPAIRMENT_MARKERS = (
    "unable",
    "cannot",
    "can't",
    "missed work",
    "lost job",
    "stopped",
    "impair",
    "interfer",
    "withdraw",
    "function",
)

CONTRADICTION_MARKERS = (
    "not a disorder",
    "despite",
    "even though",
    "no mental health issue",
    "just",
)


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    lower = str(text or "").lower()
    return any(marker in lower for marker in markers)


def validate_item(item: Dict) -> Dict[str, str]:
    prompt = item.get("prompt", "")
    incorrect = item.get("incorrect_opinion", "")
    duration_ok = _has_any(prompt, DURATION_MARKERS)
    impairment_ok = _has_any(prompt, IMPAIRMENT_MARKERS)
    contradiction_ok = _has_any(incorrect, CONTRADICTION_MARKERS)
    passed = duration_ok and impairment_ok and contradiction_ok
    reasons: List[str] = []
    if not duration_ok:
        reasons.append("missing_duration_anchor")
    if not impairment_ok:
        reasons.append("missing_impairment_anchor")
    if not contradiction_ok:
        reasons.append("incorrect_opinion_not_explicitly_contradictory")
    return {
        "id": str(item.get("id", "")),
        "passed": "1" if passed else "0",
        "duration_ok": "1" if duration_ok else "0",
        "impairment_ok": "1" if impairment_ok else "0",
        "contradiction_ok": "1" if contradiction_ok else "0",
        "reasons": ";".join(reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Study B single-turn construct anchors.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/openr1_psy_splits/study_b_test.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/study_b_construct_validation.csv"),
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("samples", [])
    else:
        items = payload

    rows = [validate_item(item) for item in items]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "passed",
                "duration_ok",
                "impairment_ok",
                "contradiction_ok",
                "reasons",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    passed = sum(1 for r in rows if r["passed"] == "1")
    pass_rate = (passed / total) if total else 0.0
    print(f"Validated {total} rows. Pass rate: {pass_rate:.1%}")
    print(f"CSV written to {args.output}")
    return 0 if pass_rate >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
