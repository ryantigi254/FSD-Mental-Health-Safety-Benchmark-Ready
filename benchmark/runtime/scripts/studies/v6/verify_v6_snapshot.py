#!/usr/bin/env python3
"""Verify source-backed provenance for the frozen `v6` snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = RUNTIME_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_clinical_benchmark.data.source_backed_snapshot import verify_source_backed_snapshot  # noqa: E402


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _case_counts(path: Path) -> dict[str, int]:
    payload = _load_json(path)
    cases = payload if isinstance(payload, list) else payload.get("cases", [])
    counts: dict[str, int] = {}
    for case in cases:
        for turn in case.get("turns", []):
            key = str(turn.get("provenance_type", "") or "")
            counts[key] = counts.get(key, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=RUNTIME_ROOT / "data" / "frozen_splits" / "v6",
        help="Frozen v6 root to verify.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNTIME_ROOT / "data" / "verification" / "v6" / "source_backed_snapshot_verification.json",
        help="Verification report output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify_source_backed_snapshot(root=args.root)
    report = {
        "root": str(args.root),
        "ok": bool(result.get("ok")),
        "error_count": len(result.get("errors", [])),
        "errors": result.get("errors", []),
        "study_b_multi_turn_provenance_counts": _case_counts(args.root / "study_b_multi_turn_test.json"),
        "study_c_provenance_counts": _case_counts(args.root / "study_c_test.json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
