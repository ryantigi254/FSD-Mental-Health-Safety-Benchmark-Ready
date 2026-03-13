#!/usr/bin/env python3
"""Summarise clinical strata for invariance sampling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = RUNTIME_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_clinical_benchmark.invariance import (
    DEFAULT_V5_ROOT,
    analyse_distribution,
    study_cli_choices,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse clinical strata for invariance sampling.")
    parser.add_argument(
        "--study",
        required=True,
        choices=list(study_cli_choices()),
        help="Study to summarise.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_V5_ROOT,
        help="Frozen v5 root or compatible release root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    summary = analyse_distribution(args.study, args.data_root.resolve())
    payload = json.dumps(summary, indent=2) + "\n"

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Wrote {args.output}")
        return 0

    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
