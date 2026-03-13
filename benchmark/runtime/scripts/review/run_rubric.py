#!/usr/bin/env python3
"""Run lightweight safety/anchor checks over generated invariance variant roots."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _text_blob_for_study(row: Dict[str, Any], study: str) -> str:
    if study == "study_a":
        return str(row.get("prompt", "") or "")
    if study == "study_b":
        metadata = row.get("metadata", {}) or {}
        return "\n".join(
            [
                str(row.get("prompt", "") or ""),
                str(metadata.get("injected_prompt_override", "") or ""),
                str(row.get("incorrect_opinion", "") or ""),
            ]
        )
    if study == "study_b_multi_turn":
        return "\n".join(str(turn.get("message", "") or "") for turn in row.get("turns", []))
    if study == "study_c":
        turns = "\n".join(str(turn.get("message", "") or "") for turn in row.get("turns", []))
        return "\n".join([str(row.get("patient_summary", "") or ""), turns])
    return ""


def _load_rows(root: Path, study: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    filename_map = {
        "study_a": "study_a_test.json",
        "study_b": "study_b_test.json",
        "study_b_multi_turn": "study_b_multi_turn_test.json",
        "study_c": "study_c_test.json",
    }
    payload = _load_json(root / filename_map[study])
    if study == "study_a":
        return payload, payload.get("samples", [])
    if study == "study_c":
        return payload, payload.get("cases", [])
    return payload, payload if isinstance(payload, list) else payload.get("multi_turn_cases", [])


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def _risk_groups(text: str, rules: Dict[str, Any]) -> set[str]:
    text_lower = text.lower()
    hits = set()
    for group, tokens in (rules.get("risk_marker_groups", {}) or {}).items():
        if any(token.lower() in text_lower for token in tokens):
            hits.add(group)
    return hits


def _unsafe_hits(text: str, rules: Dict[str, Any]) -> List[str]:
    text_lower = text.lower()
    return [pattern for pattern in (rules.get("unsafe_patterns", []) or []) if pattern.lower() in text_lower]


def _validate_row(
    *,
    base_row: Dict[str, Any],
    variant_row: Dict[str, Any],
    study: str,
    rules: Dict[str, Any],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if study == "study_c":
        if base_row.get("critical_entities") != variant_row.get("critical_entities"):
            reasons.append("critical_entities changed")
    if study in {"study_b", "study_b_multi_turn", "study_c"}:
        if (base_row.get("metadata", {}) or {}).get("source_openr1_ids") != (variant_row.get("metadata", {}) or {}).get("source_openr1_ids"):
            reasons.append("source_openr1_ids changed")
    if study == "study_b" and base_row.get("incorrect_opinion") != variant_row.get("incorrect_opinion"):
        reasons.append("incorrect_opinion changed")

    base_text = _text_blob_for_study(base_row, study)
    variant_text = _text_blob_for_study(variant_row, study)
    if not variant_text.strip():
        reasons.append("variant text empty")

    missing_numbers = sorted(_numbers(base_text) - _numbers(variant_text))
    if missing_numbers:
        reasons.append(f"numbers missing: {missing_numbers[:5]}")

    base_risk = _risk_groups(base_text, rules)
    variant_risk = _risk_groups(variant_text, rules)
    if base_risk - variant_risk:
        reasons.append(f"risk markers dropped: {sorted(base_risk - variant_risk)}")

    unsafe = _unsafe_hits(variant_text, rules)
    if unsafe:
        reasons.append(f"unsafe patterns: {unsafe}")

    return ("Acceptable" if not reasons else "Invalid", reasons)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic variant rubric checks.")
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Variant root to validate.")
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("benchmark/runtime/data/frozen_splits/v5_invariance_samples"),
        help="Base sampled split root to compare against.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rules = _load_json(args.rules)
    metadata = _load_json(args.input / "variant_metadata.json")
    study = str(metadata.get("study", "") or "")
    _, base_rows = _load_rows(args.base_root, study)
    _, variant_rows = _load_rows(args.input, study)
    base_by_id = {str(row.get("id", "")): row for row in base_rows if str(row.get("id", ""))}
    variant_by_id = {str(row.get("id", "")): row for row in variant_rows if str(row.get("id", ""))}

    verdicts = []
    acceptable = 0
    for row_id in metadata.get("changed_ids", []):
        base_row = base_by_id.get(row_id)
        variant_row = variant_by_id.get(row_id)
        if base_row is None or variant_row is None:
            verdicts.append({"id": row_id, "verdict": "Invalid", "reasons": ["missing row pairing"]})
            continue
        verdict, reasons = _validate_row(base_row=base_row, variant_row=variant_row, study=study, rules=rules)
        if verdict == "Acceptable":
            acceptable += 1
        verdicts.append({"id": row_id, "verdict": verdict, "reasons": reasons})

    payload = {
        "study": study,
        "variant_tag": metadata.get("variant_tag"),
        "variant_root": str(args.input.resolve()),
        "base_root": str(args.base_root.resolve()),
        "rule_version": rules.get("rule_version"),
        "n_total": len(verdicts),
        "n_acceptable": acceptable,
        "overall_verdict": "Acceptable" if acceptable == len(verdicts) else "Invalid",
        "verdicts": verdicts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(payload["overall_verdict"])
    return 0 if payload["overall_verdict"] == "Acceptable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
