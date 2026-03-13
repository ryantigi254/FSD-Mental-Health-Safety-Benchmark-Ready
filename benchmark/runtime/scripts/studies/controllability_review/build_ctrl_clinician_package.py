#!/usr/bin/env python3
"""Build the controllability clinician package."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_readiness = _load_module(
    "ctrl_controllability_clinical_readiness_package",
    SRC_DIR
    / "reliable_clinical_benchmark"
    / "review"
    / "controllability_clinical_readiness.py",
)

DEFAULT_CTRL_DIR = _readiness.DEFAULT_CTRL_DIR
DEFAULT_PACKAGE_DIR = _readiness.DEFAULT_PACKAGE_DIR
DEFAULT_VERIFICATION_DIR = _readiness.DEFAULT_VERIFICATION_DIR
build_clinician_package = _readiness.build_clinician_package


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the controllability clinician package.")
    parser.add_argument("--ctrl-dir", type=Path, default=DEFAULT_CTRL_DIR)
    parser.add_argument("--verification-dir", type=Path, default=DEFAULT_VERIFICATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_clinician_package(
        ctrl_dir=args.ctrl_dir,
        verification_dir=args.verification_dir,
        output_dir=args.output_dir,
    )
    print(f"Package release status: {result['release_status']}")
    print(f"Manifest written to {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
