#!/usr/bin/env python3
"""Build deterministic frozen v5 invariance manifests and split files."""

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
    DEFAULT_INVARIANCE_SAMPLE_SIZES,
    DEFAULT_V5_ROOT,
    build_invariance_manifest,
    materialize_invariance_split_root,
    study_cli_choices,
)


DEFAULT_OUTPUT_ROOT = RUNTIME_ROOT / "data" / "frozen_splits" / "v5_invariance_samples"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate manifests and materialised split files for frozen v5 invariance runs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_V5_ROOT,
        help="Frozen v5 source root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Target root for manifests and materialised sampled split files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--min-high-risk",
        type=int,
        default=5,
        help="Minimum allocation per high-risk stratum when feasible.",
    )
    parser.add_argument(
        "--study",
        action="append",
        choices=list(study_cli_choices()),
        help="Optional subset of studies to build. Default is all invariance studies.",
    )
    args = parser.parse_args()

    selected_studies = args.study or list(study_cli_choices())
    args.output_root.mkdir(parents=True, exist_ok=True)

    for study_name in selected_studies:
        manifest = build_invariance_manifest(
            study_name,
            root=args.data_root.resolve(),
            sample_size=DEFAULT_INVARIANCE_SAMPLE_SIZES[study_name],
            seed=args.seed,
            min_high_risk=args.min_high_risk,
        )
        manifest_path = args.output_root / f"{study_name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote manifest {manifest_path}")

    summary = materialize_invariance_split_root(
        source_root=args.data_root.resolve(),
        output_root=args.output_root.resolve(),
        manifest_dir=args.output_root.resolve(),
        studies=selected_studies,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
