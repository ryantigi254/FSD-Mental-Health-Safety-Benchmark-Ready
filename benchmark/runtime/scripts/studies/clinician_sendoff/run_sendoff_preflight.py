#!/usr/bin/env python3
"""Run end-to-end clinician send-off preflight checks for v0.3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


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
    parser = argparse.ArgumentParser(description="Run clinician send-off v0.3 preflight checks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/clinician_package/v0.3/sendoff_preflight_report.json"),
    )
    args = parser.parse_args()

    checks = [
        _run(
            [
                "pytest",
                "tests/unit/eval/test_runtime_checks.py",
                "-q",
            ],
            ROOT,
        ),
        _run(
            [
                sys.executable,
                "scripts/studies/clinician_sendoff/run_stage2_gates.py",
            ],
            ROOT,
        ),
        _run(
            [
                sys.executable,
                "scripts/studies/clinician_sendoff/build_clinician_package.py",
            ],
            ROOT,
        ),
        _run(
            [
                "pytest",
                "tests/unit/data/test_frozen_snapshot_v03_manifest.py",
                "-q",
            ],
            ROOT,
        ),
        _run(
            [
                "pytest",
                "tests/unit/data/test_clinician_package_v03_manifest.py",
                "-q",
            ],
            ROOT,
        ),
        _run(
            [
                "pytest",
                "tests/unit/data/test_clinician_package_v03_schema.py",
                "-q",
            ],
            ROOT,
        ),
        _run(
            [
                sys.executable,
                "scripts/studies/clinician_sendoff/rebuild_release_manifest.py",
                "--verify",
            ],
            ROOT,
        ),
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "v0.3_sendoff_preflight",
        "overall_passed": all(check["passed"] for check in checks),
        "checks": checks,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    passed = sum(1 for check in checks if check["passed"])
    total = len(checks)
    print(f"Preflight checks: {passed}/{total} passed")
    print(f"Report written to {output_path}")

    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
