#!/usr/bin/env python3
"""Export per-case invariance deltas for granular notebooks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = RUNTIME_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_clinical_benchmark.invariance import DEFAULT_V5_ROOT, study_cli_choices, study_metric_names
from reliable_clinical_benchmark.invariance_analysis import build_invariance_case_delta_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Export per-case invariance deltas.")
    parser.add_argument("--study", required=True, choices=list(study_cli_choices()))
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--variant-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_V5_ROOT)
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument("--use-nli", action="store_true")
    parser.add_argument("--nli-stride", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = build_invariance_case_delta_rows(
        study=args.study,
        base_cache=args.base_cache.resolve(),
        variant_cache=args.variant_cache.resolve(),
        data_root=args.data_root.resolve(),
        metrics=args.metric or list(study_metric_names(args.study)),
        use_nli=args.use_nli,
        nli_stride=args.nli_stride,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
