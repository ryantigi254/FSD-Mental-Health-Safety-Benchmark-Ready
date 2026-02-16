#!/usr/bin/env python3
"""Validate that all Study C critical_entities are anchored in patient_summary.

Entities that are literal substrings of patient_summary pass automatically.
Semantic abstractions (e.g. 'functional impairment') are validated against
the synonym map in data/study_c_gold/entity_evidence_map.json.

Exit code 0 if all entities are anchored; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _is_literal(entity: str, summary: str) -> bool:
    return entity.lower() in summary.lower()


def _is_numeric_age(entity: str, summary: str) -> bool:
    try:
        age = int(entity)
        return f"{age}-year-old" in summary
    except ValueError:
        return False


def _is_valid_case_evidence_span(span: str, summary: str) -> bool:
    if not isinstance(span, str):
        return False
    cleaned = span.strip()
    if not cleaned:
        return False
    if "(unanchored)" in cleaned.lower():
        return False
    return cleaned.lower() in summary.lower()


def validate(cases_path: Path, evidence_map_path: Path) -> list[dict]:
    with cases_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload.get("cases", payload) if isinstance(payload, dict) else payload

    evidence_map: dict = {}
    if evidence_map_path.exists():
        with evidence_map_path.open(encoding="utf-8") as f:
            evidence_map = json.load(f)

    global_synonyms = evidence_map.get("global_synonyms", {})
    case_evidence = evidence_map.get("case_evidence", {})

    rows: list[dict] = []
    for case in cases:
        cid = case["id"]
        summary = case["patient_summary"]
        for ent in case["critical_entities"]:
            anchored = False
            method = ""

            if _is_literal(ent, summary):
                anchored = True
                method = "literal"
            if not anchored and _is_numeric_age(ent, summary):
                anchored = True
                method = "numeric_age"
            if not anchored and ent in global_synonyms:
                patterns = global_synonyms[ent].get("evidence_patterns", [])
                if any(p.lower() in summary.lower() for p in patterns):
                    anchored = True
                    method = "synonym_map"
            if not anchored and cid in case_evidence and ent in case_evidence[cid]:
                span = case_evidence[cid][ent]
                if _is_valid_case_evidence_span(span, summary):
                    anchored = True
                    method = "case_evidence"

            rows.append({
                "case_id": cid,
                "entity": ent,
                "anchored": anchored,
                "method": method,
            })

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Study C entity anchoring.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/openr1_psy_splits/study_c_test.json"),
    )
    parser.add_argument(
        "--evidence-map",
        type=Path,
        default=Path("data/study_c_gold/entity_evidence_map.json"),
    )
    args = parser.parse_args()

    rows = validate(args.cases, args.evidence_map)
    total = len(rows)
    anchored = sum(1 for r in rows if r["anchored"])
    failures = [r for r in rows if not r["anchored"]]

    print(f"Validated {total} entity-case pairs. Anchored: {anchored}/{total}")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for r in failures:
            print(f"  {r['case_id']}: {r['entity']}")
        return 1

    print("PASS: All entities anchored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
