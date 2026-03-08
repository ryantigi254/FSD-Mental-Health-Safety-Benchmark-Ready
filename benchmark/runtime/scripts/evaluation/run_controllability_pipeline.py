#!/usr/bin/env python3
"""Run controllability evaluation and write structured result JSON files."""
# pylint: disable=import-error,wrong-import-position

from __future__ import annotations

import argparse
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from reliable_clinical_benchmark.pipelines.controllability import (  # noqa: E402
    DEFAULT_CTRL_DIR,
    DEFAULT_RESULTS_DIR,
    run_controllability_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate controllability caches and write structured results.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model result directory name under benchmark/runtime/results/",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory containing per-model benchmark results.",
    )
    parser.add_argument(
        "--ctrl-dir",
        type=Path,
        default=DEFAULT_CTRL_DIR,
        help="Directory containing controllability split and gold artefacts.",
    )
    parser.add_argument(
        "--skip-nli",
        action="store_true",
        help="Skip NLI-backed metrics such as H_Ev and K_Conflict when dependencies are unavailable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_controllability_pipeline(
        model_name=args.model,
        results_dir=args.results_dir,
        ctrl_dir=args.ctrl_dir,
        use_nli=not args.skip_nli,
    )
    print(f"Model: {summary.model}")
    print(f"Benchmark control score: {summary.benchmark_control.score}")
    print(f"Wrote controllability outputs under: {Path(args.results_dir) / args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
