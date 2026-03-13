#!/usr/bin/env python3
"""Run paired invariance comparisons for cached generations."""

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
    compare_invariance_runs,
    study_cli_choices,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare base and variant caches with paired bootstrap deltas.")
    parser.add_argument(
        "--study",
        required=True,
        choices=list(study_cli_choices()),
        help="Study to compare.",
    )
    parser.add_argument("--base-cache", type=Path, required=True, help="Baseline cache JSONL path.")
    parser.add_argument("--variant-cache", type=Path, required=True, help="Variant cache JSONL path.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_V5_ROOT,
        help="Frozen v5 root or compatible release root.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Number of paired bootstrap resamples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed.")
    parser.add_argument(
        "--use-nli",
        action="store_true",
        help="Enable Study C knowledge-conflict scoring with NLI.",
    )
    parser.add_argument(
        "--nli-stride",
        type=int,
        default=2,
        help="Study C NLI stride when --use-nli is enabled.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    args = parser.parse_args()

    try:
        comparison = compare_invariance_runs(
            study=args.study,
            base_cache=args.base_cache.resolve(),
            variant_cache=args.variant_cache.resolve(),
            data_root=args.data_root.resolve(),
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
            use_nli=args.use_nli,
            nli_stride=args.nli_stride,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    for metric_name, summary in comparison["metrics"].items():
        print(
            f"{metric_name}: base={summary['base']:.4f} "
            f"variant={summary['variant']:.4f} delta={summary['delta']:.4f} "
            f"95% CI [{summary['ci_low']:.4f}, {summary['ci_high']:.4f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
