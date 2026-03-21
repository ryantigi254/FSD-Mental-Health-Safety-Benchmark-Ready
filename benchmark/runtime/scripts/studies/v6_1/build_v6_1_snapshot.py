#!/usr/bin/env python3
"""Build the strict no-generation `v6.1` frozen snapshot from the canonical `v5` root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = RUNTIME_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_clinical_benchmark.data.strict_snapshot_v6_1 import (  # noqa: E402
    build_v6_1_snapshot,
)
from reliable_clinical_benchmark.data.source_backed_snapshot import (  # noqa: E402
    DEFAULT_MULTI_TURN_TARGET,
    DEFAULT_SINGLE_TURN_TARGET,
    DEFAULT_STUDY_C_TARGET,
    DEFAULT_TURNS_PER_CASE,
    SEED,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v5-root",
        type=Path,
        default=RUNTIME_ROOT / "data" / "frozen_splits" / "v5",
        help="Input frozen v5 root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RUNTIME_ROOT / "data" / "frozen_splits" / "v6_1",
        help="Output frozen v6.1 root.",
    )
    parser.add_argument(
        "--single-turn-target",
        type=int,
        default=DEFAULT_SINGLE_TURN_TARGET,
        help="Target row count for Study B single-turn.",
    )
    parser.add_argument(
        "--multi-turn-target",
        type=int,
        default=DEFAULT_MULTI_TURN_TARGET,
        help="Target case count for Study B multi-turn.",
    )
    parser.add_argument(
        "--study-c-target",
        type=int,
        default=DEFAULT_STUDY_C_TARGET,
        help="Target case count for Study C.",
    )
    parser.add_argument(
        "--turns-per-case",
        type=int,
        default=DEFAULT_TURNS_PER_CASE,
        help="Target patient-turn count for rebuilt multi-turn cases.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Deterministic random seed.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output root before rebuilding.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_v6_1_snapshot(
        v5_root=args.v5_root,
        output_root=args.output_root,
        single_turn_target=args.single_turn_target,
        multi_turn_target=args.multi_turn_target,
        study_c_target=args.study_c_target,
        turns_per_case=args.turns_per_case,
        seed=args.seed,
        clean=args.clean,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("verification", {}).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
