#!/usr/bin/env python3
"""Build a full invariance variant matrix from one fixed sampled split root."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _variant_common import write_json
from variant_catalog import matrix_manifest, resolve_specs, study_choices, variant_choices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialise a variant-family matrix from one frozen invariance sample root."
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("benchmark/runtime/data/frozen_splits/v5_invariance_samples"),
        help="Frozen sampled invariance root to fan out into variant families.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Parent directory for per-study/per-variant roots.",
    )
    parser.add_argument(
        "--study",
        action="append",
        choices=list(study_choices()),
        help="Optional subset of studies. Defaults to the full variant matrix.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=list(variant_choices()),
        help="Optional subset of concrete variants across the selected studies.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    specs = resolve_specs(studies=args.study, variants=args.variant)
    if not specs:
        raise SystemExit("No variant specs selected.")

    base_root = args.base_root.resolve()
    output_root = args.output_root.resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        variant_root = output_root / spec.study / spec.variant
        command = [
            sys.executable,
            str(spec.script_path),
            "--study",
            spec.study,
            "--variant",
            spec.variant,
            "--base-root",
            str(base_root),
            "--output-root",
            str(variant_root),
            "--seed",
            str(args.seed),
        ]
        subprocess.run(command, check=True, cwd=str(spec.script_path.parent))

    manifest = matrix_manifest(
        base_root=base_root,
        output_root=output_root,
        seed=args.seed,
        specs=specs,
    )
    write_json(output_root / "variant_matrix_manifest.json", manifest)
    print(f"Wrote variant matrix {output_root}")
    print(f"Variants materialised: {len(specs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
