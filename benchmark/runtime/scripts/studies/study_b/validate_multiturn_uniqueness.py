#!/usr/bin/env python3
"""Validate that Study B multi-turn conversations have unique full-turn signatures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _normalise(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _signature(case: dict) -> str:
    turns = case.get("turns", [])
    joined = "||".join(_normalise(t.get("message", "")) for t in turns)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Study B multi-turn uniqueness.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/openr1_psy_splits/study_b_multi_turn_test.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/clinician_package/v0.3/study_b_multiturn_uniqueness.csv"),
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cases = payload.get("cases", []) if isinstance(payload, dict) else payload

    seen: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    duplicates = 0
    for case in cases:
        cid = str(case.get("id", ""))
        sig = _signature(case)
        duplicate_of = seen.get(sig, "")
        if duplicate_of:
            duplicates += 1
        else:
            seen[sig] = cid
        rows.append({
            "id": cid,
            "signature_sha256": sig,
            "duplicate_of": duplicate_of,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "signature_sha256", "duplicate_of"])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    unique = total - duplicates
    print(f"Validated {total} cases. Unique signatures: {unique}. Duplicates: {duplicates}")
    print(f"CSV written to {args.output}")
    return 0 if duplicates == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
