#!/usr/bin/env python3
"""
Build one frozen case manifest for pairwise secondary evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_RUNTIME_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_RUNTIME_ROOT / "src"))

from reliable_clinical_benchmark.pairwise.manifest_builder import build_case_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a pairwise case manifest for one slice.")
    parser.add_argument("--slice-id", required=True, help="Slice id to build.")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output JSON path. Defaults to metric-results/pairwise/manifests/<slice>_case_manifest.json",
    )
    parser.add_argument(
        "--systems",
        default="",
        help="Optional comma-separated candidate system ids to keep in the manifest.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if the slice cannot be built cleanly.")
    args = parser.parse_args()
    include_systems = [
        item.strip()
        for item in str(args.systems).split(",")
        if item.strip()
    ]

    manifest = build_case_manifest(
        slice_id=args.slice_id,
        runtime_root=_RUNTIME_ROOT,
        output_path=args.output_path,
        include_systems=include_systems or None,
        strict=args.strict,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
