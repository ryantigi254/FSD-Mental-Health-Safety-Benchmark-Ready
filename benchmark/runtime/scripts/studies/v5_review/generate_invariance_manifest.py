#!/usr/bin/env python3
"""Generate deterministic v5 invariance sampling manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = RUNTIME_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_clinical_benchmark.invariance import DEFAULT_V5_ROOT, build_invariance_manifest


DEFAULT_OUTPUTS = {
    "study_a": RUNTIME_ROOT / "data" / "frozen_splits" / "v5_invariance_samples" / "study_a_manifest.json",
    "study_b": RUNTIME_ROOT / "data" / "frozen_splits" / "v5_invariance_samples" / "study_b_manifest.json",
    "study_b_multi_turn": RUNTIME_ROOT / "data" / "frozen_splits" / "v5_invariance_samples" / "study_b_multi_turn_manifest.json",
    "study_c": RUNTIME_ROOT / "data" / "frozen_splits" / "v5_invariance_samples" / "study_c_manifest.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic invariance sampling manifests.")
    parser.add_argument(
        "--study",
        required=True,
        choices=list(DEFAULT_OUTPUTS),
        help="Study to sample.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
        help="Diagnostic subset size in frozen evaluation units (heuristic budget, not a spec threshold).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_V5_ROOT,
        help="Frozen v5 root or compatible release root.",
    )
    parser.add_argument(
        "--min-high-risk",
        type=int,
        default=5,
        help="Minimum allocation per high-risk stratum when feasible within the diagnostic budget.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output manifest path.",
    )
    args = parser.parse_args()

    manifest = build_invariance_manifest(
        args.study,
        root=args.data_root.resolve(),
        sample_size=args.sample_size,
        seed=args.seed,
        min_high_risk=args.min_high_risk,
    )
    output_path = args.out or DEFAULT_OUTPUTS[args.study]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Sampled {manifest['sample_size']} records for {manifest['study']} with seed={manifest['seed']}")
    print(f"Coverage axes: {', '.join(manifest['coverage_axes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
