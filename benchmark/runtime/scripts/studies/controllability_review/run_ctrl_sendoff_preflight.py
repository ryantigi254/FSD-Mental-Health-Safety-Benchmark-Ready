#!/usr/bin/env python3
"""Run end-to-end preflight for controllability clinician-readiness."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"


def _supports_modules(python_bin: str, modules: list[str]) -> bool:
    probe = "import " + ", ".join(modules)
    proc = subprocess.run([python_bin, "-c", probe], capture_output=True, text=True)
    return proc.returncode == 0


def _find_compatible_python() -> str:
    candidates = [
        sys.executable,
        shutil.which("python"),
        shutil.which("python3"),
        "/opt/homebrew/Caskroom/miniforge/base/bin/python3",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists() and _supports_modules(candidate, ["pytest"]):
            return candidate
    raise RuntimeError("No compatible Python interpreter with pytest available.")


_COMPATIBLE_PYTHON = _find_compatible_python()
if Path(sys.executable).resolve() != Path(_COMPATIBLE_PYTHON).resolve():
    os.execv(_COMPATIBLE_PYTHON, [_COMPATIBLE_PYTHON, __file__, *sys.argv[1:]])


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_readiness = _load_module(
    "ctrl_controllability_clinical_readiness_preflight",
    SRC_DIR
    / "reliable_clinical_benchmark"
    / "review"
    / "controllability_clinical_readiness.py",
)

DEFAULT_CTRL_DIR = _readiness.DEFAULT_CTRL_DIR
DEFAULT_PACKAGE_DIR = _readiness.DEFAULT_PACKAGE_DIR
DEFAULT_VERIFICATION_DIR = _readiness.DEFAULT_VERIFICATION_DIR
EXPECTED_CTRL_COUNTS = _readiness.EXPECTED_CTRL_COUNTS
REVIEW_FILENAMES = _readiness.REVIEW_FILENAMES
build_clinician_package = _readiness.build_clinician_package
review_blockers = _readiness.review_blockers
review_summary_rows = _readiness.review_summary_rows
verify_package_manifest = _readiness.verify_package_manifest
write_json = _readiness.write_json


def _run(cmd: list[str], cwd: Path) -> dict[str, object]:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controllability clinician-readiness preflight."
    )
    parser.add_argument("--ctrl-dir", type=Path, default=DEFAULT_CTRL_DIR)
    parser.add_argument("--verification-dir", type=Path, default=DEFAULT_VERIFICATION_DIR)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Preflight report path. Defaults to <package-dir>/sendoff_preflight_report.json.",
    )
    parser.add_argument("--rules", type=Path, default=ROOT / "data" / "verification" / "v4" / "rubric_rules_v2.json")
    return parser.parse_args(argv)


def _required_review_files(verification_dir: Path) -> list[Path]:
    return [
        verification_dir / REVIEW_FILENAMES["study_a"],
        verification_dir / REVIEW_FILENAMES["study_a_bias"],
        verification_dir / REVIEW_FILENAMES["study_b_single"],
        verification_dir / REVIEW_FILENAMES["study_b_multi"],
        verification_dir / REVIEW_FILENAMES["study_c"],
        verification_dir / REVIEW_FILENAMES["summary"],
        verification_dir / REVIEW_FILENAMES["metadata"],
        verification_dir / "stage2_gate_report.json",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output or (args.package_dir / "sendoff_preflight_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.verification_dir.mkdir(parents=True, exist_ok=True)
    args.package_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, object]] = []

    review_cmd = [
        _COMPATIBLE_PYTHON,
        "scripts/studies/controllability_review/run_ctrl_cross_study_review.py",
        "--ctrl-dir",
        str(args.ctrl_dir),
        "--out-dir",
        str(args.verification_dir),
        "--rules",
        str(args.rules),
        "--clean",
        "--expect-study-a",
        str(EXPECTED_CTRL_COUNTS["study_a"]),
        "--expect-study-a-bias",
        str(EXPECTED_CTRL_COUNTS["study_a_bias"]),
        "--expect-study-b-single",
        str(EXPECTED_CTRL_COUNTS["study_b_single"]),
        "--expect-study-b-multi",
        str(EXPECTED_CTRL_COUNTS["study_b_multi"]),
        "--expect-study-c",
        str(EXPECTED_CTRL_COUNTS["study_c"]),
    ]
    checks.append(_run(review_cmd, ROOT))

    gates_cmd = [
        _COMPATIBLE_PYTHON,
        "scripts/studies/controllability_review/run_ctrl_stage2_gates.py",
        "--ctrl-dir",
        str(args.ctrl_dir),
        "--verification-dir",
        str(args.verification_dir),
    ]
    checks.append(_run(gates_cmd, ROOT))

    package_cmd = [
        _COMPATIBLE_PYTHON,
        "scripts/studies/controllability_review/build_ctrl_clinician_package.py",
        "--ctrl-dir",
        str(args.ctrl_dir),
        "--verification-dir",
        str(args.verification_dir),
        "--output-dir",
        str(args.package_dir),
    ]
    checks.append(_run(package_cmd, ROOT))

    pytest_cmd = [
        _COMPATIBLE_PYTHON,
        "-m",
        "pytest",
        "tests/unit/review/test_ctrl_clinical_readiness.py",
        "tests/unit/data/test_ctrl_clinician_package_manifest.py",
        "-q",
    ]
    checks.append(_run(pytest_cmd, ROOT))

    missing_artifacts = [
        str(path) for path in _required_review_files(args.verification_dir) if not path.exists()
    ]

    review_summary_path = args.verification_dir / REVIEW_FILENAMES["summary"]
    blockers = review_blockers(review_summary_rows(review_summary_path)) if review_summary_path.exists() else ["missing_review_summary"]

    overall_passed = (
        all(check["passed"] for check in checks)
        and not missing_artifacts
        and not blockers
    )
    preflight_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "controllability_v0.1_large_resolved_preflight",
        "ctrl_dir": str(args.ctrl_dir.resolve()),
        "verification_dir": str(args.verification_dir.resolve()),
        "package_dir": str(args.package_dir.resolve()),
        "overall_passed": overall_passed,
        "release_status": "ready" if overall_passed else "blocked",
        "missing_artifacts": missing_artifacts,
        "review_blockers": blockers,
        "checks": checks,
    }
    write_json(output_path, preflight_payload)

    build_clinician_package(
        ctrl_dir=args.ctrl_dir,
        verification_dir=args.verification_dir,
        output_dir=args.package_dir,
        preflight_report_payload=preflight_payload,
    )

    package_manifest_check = verify_package_manifest(args.package_dir)
    manifest_check = {
        "command": "verify_package_manifest",
        "exit_code": 0 if package_manifest_check["passed"] else 1,
        "stdout": "",
        "stderr": "; ".join(package_manifest_check["issues"]),
        "passed": package_manifest_check["passed"],
    }
    checks.append(manifest_check)
    preflight_payload["checks"] = checks
    preflight_payload["overall_passed"] = (
        all(check["passed"] for check in checks)
        and not missing_artifacts
        and not blockers
    )
    preflight_payload["release_status"] = (
        "ready" if preflight_payload["overall_passed"] else "blocked"
    )
    write_json(output_path, preflight_payload)
    build_clinician_package(
        ctrl_dir=args.ctrl_dir,
        verification_dir=args.verification_dir,
        output_dir=args.package_dir,
        preflight_report_payload=preflight_payload,
    )

    print(
        "Controllability preflight: "
        f"{sum(1 for check in checks if check['passed'])}/{len(checks)} checks passed"
    )
    print(f"Report written to {output_path}")
    return 0 if preflight_payload["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
