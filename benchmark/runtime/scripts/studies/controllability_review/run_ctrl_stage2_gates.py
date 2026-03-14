#!/usr/bin/env python3
"""Run controllability-specific stage-2 hardening gates."""

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
    "ctrl_controllability_clinical_readiness_stage2",
    SRC_DIR
    / "reliable_clinical_benchmark"
    / "review"
    / "controllability_clinical_readiness.py",
)

DEFAULT_CTRL_DIR = _readiness.DEFAULT_CTRL_DIR
DEFAULT_SMALL_CTRL_DIR = _readiness.DEFAULT_SMALL_CTRL_DIR
DEFAULT_VERIFICATION_DIR = _readiness.DEFAULT_VERIFICATION_DIR
EXPECTED_CTRL_COUNTS = _readiness.EXPECTED_CTRL_COUNTS
run_stage2_gates = _readiness.run_stage2_gates
write_json = _readiness.write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controllability stage-2 hardening gates.")
    parser.add_argument("--ctrl-dir", type=Path, default=DEFAULT_CTRL_DIR)
    parser.add_argument("--small-ctrl-dir", type=Path, default=DEFAULT_SMALL_CTRL_DIR)
    parser.add_argument("--verification-dir", type=Path, default=DEFAULT_VERIFICATION_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expect-study-a", type=int, default=EXPECTED_CTRL_COUNTS["study_a"])
    parser.add_argument(
        "--expect-study-a-bias", type=int, default=EXPECTED_CTRL_COUNTS["study_a_bias"]
    )
    parser.add_argument(
        "--expect-study-b-single", type=int, default=EXPECTED_CTRL_COUNTS["study_b_single"]
    )
    parser.add_argument(
        "--expect-study-b-multi", type=int, default=EXPECTED_CTRL_COUNTS["study_b_multi"]
    )
    parser.add_argument("--expect-study-c", type=int, default=EXPECTED_CTRL_COUNTS["study_c"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output or args.verification_dir / "stage2_gate_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = run_stage2_gates(
        ctrl_dir=args.ctrl_dir,
        small_ctrl_dir=args.small_ctrl_dir,
        expected_counts={
            "study_a": args.expect_study_a,
            "study_a_bias": args.expect_study_a_bias,
            "study_b_single": args.expect_study_b_single,
            "study_b_multi": args.expect_study_b_multi,
            "study_c": args.expect_study_c,
        },
    )
    write_json(output_path, report)

    passed = sum(1 for gate in report["gates"] if gate["passed"])
    total = len(report["gates"])
    print(f"Controllability stage-2 gates: {passed}/{total} passed")
    print(f"Report written to {output_path}")
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
