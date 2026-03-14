#!/usr/bin/env python3
"""Generate deterministic invariance sampling manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = RUNTIME_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_clinical_benchmark.invariance import (
    DEFAULT_V5_ROOT,
    build_invariance_manifest,
    default_invariance_output_root,
    default_invariance_sample_sizes,
    resolve_invariance_sample_profile,
    study_cli_choices,
)

def main() -> int:
    parser = argparse.ArgumentParser(description="Create deterministic invariance sampling manifests.")
    parser.add_argument(
        "--study",
        required=True,
        choices=list(study_cli_choices()),
        help="Study to sample.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Diagnostic subset size in evaluation units. Defaults from the selected profile.",
    )
    parser.add_argument(
        "--sample-profile",
        choices=("auto", "v5", "controllability"),
        default="auto",
        help="Default sample-budget profile. `auto` infers from --data-root.",
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
    resolved_profile = resolve_invariance_sample_profile(args.sample_profile, args.data_root.resolve())
    sample_size = args.sample_size or default_invariance_sample_sizes(resolved_profile)[args.study]

    manifest = build_invariance_manifest(
        args.study,
        root=args.data_root.resolve(),
        sample_size=sample_size,
        seed=args.seed,
        min_high_risk=args.min_high_risk,
    )
    default_output_dir = default_invariance_output_root(resolved_profile)
    output_path = args.out or (default_output_dir / f"{args.study}_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Sampled {manifest['sample_size']} records for {manifest['study']} with seed={manifest['seed']}")
    print(f"Coverage axes: {', '.join(manifest['coverage_axes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
