#!/usr/bin/env python3
"""Run Stage 2 clinician send-off hardening gates and write a summary report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    return {
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "passed": proc.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 2 hardening gates.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/clinician_package/v0.3/stage2_gate_report.json"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]

    gates = [
        _run(
            [
                sys.executable,
                "scripts/studies/study_b/validate_multiturn_uniqueness.py",
                "--output",
                "docs/reports/clinician_package/v0.3/study_b_multiturn_uniqueness.csv",
            ],
            root,
        ),
        _run(
            [
                sys.executable,
                "scripts/studies/study_b/validate_constructs.py",
                "--output",
                "docs/reports/clinician_package/v0.3/study_b_construct_validation.csv",
            ],
            root,
        ),
        _run(
            [
                sys.executable,
                "scripts/studies/study_a/validate_label_policy.py",
                "--output",
                "docs/reports/clinician_package/v0.3/study_a_label_policy.csv",
            ],
            root,
        ),
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "v0.3_stage2_gates",
        "overall_passed": all(g["passed"] for g in gates),
        "gates": gates,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    passed = sum(1 for g in gates if g["passed"])
    total = len(gates)
    print(f"Stage 2 gates: {passed}/{total} passed")
    print(f"Report written to {args.output}")
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
