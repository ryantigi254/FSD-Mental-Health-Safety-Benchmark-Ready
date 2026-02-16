"""Integrity checks for clinician package v0.3 manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
PACKAGE_DIR = BASE_DIR / "docs" / "reports" / "clinician_package" / "v0.3"
MANIFEST_PATH = PACKAGE_DIR / "manifest.json"


@pytest.mark.unit
def test_clinician_package_v03_manifest_declared_files_exist():
    assert MANIFEST_PATH.exists(), f"Missing clinician package manifest: {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    missing = []
    for entry in manifest.get("files", []):
        file_name = entry["file"]
        file_path = PACKAGE_DIR / file_name
        if not file_path.exists():
            missing.append(file_name)

    assert not missing, f"{len(missing)} missing clinician package files: {missing}"


@pytest.mark.unit
def test_clinician_package_v03_manifest_hashes_match_files():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    mismatches = []
    for entry in manifest.get("files", []):
        file_name = entry["file"]
        expected_sha = entry["sha256"]
        file_path = PACKAGE_DIR / file_name
        actual_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            mismatches.append(
                f"{file_name}: expected={expected_sha} actual={actual_sha}"
            )

    assert not mismatches, (
        f"{len(mismatches)} clinician package hash mismatches:\n" + "\n".join(mismatches)
    )
