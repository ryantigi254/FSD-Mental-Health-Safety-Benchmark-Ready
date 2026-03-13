#!/usr/bin/env python3
"""Flatten invariance comparison JSON files into analysis-friendly rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = RUNTIME_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_clinical_benchmark.invariance_analysis import summarize_invariance_result_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize invariance comparison JSON files.")
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Directory to scan for invariance comparison JSON files.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    args = parser.parse_args()

    rows = summarize_invariance_result_files(args.results_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
