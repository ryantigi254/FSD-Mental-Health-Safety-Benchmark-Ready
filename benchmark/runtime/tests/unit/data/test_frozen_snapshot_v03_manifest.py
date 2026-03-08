"""Integrity checks for v0.3 clinician-audit frozen snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = BASE_DIR / "data" / "frozen_splits" / "v0.3_postclinician_audit"
MANIFEST_PATH = SNAPSHOT_DIR / "manifest.json"


@pytest.mark.unit
def test_v03_manifest_declared_files_exist():
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    missing = []
    for entry in manifest.get("files", []):
        name = entry["file"]
        path = SNAPSHOT_DIR / name
        if not path.exists():
            missing.append(name)
    assert not missing, f"{len(missing)} missing snapshot files: {missing}"


@pytest.mark.unit
def test_v03_manifest_sha256_matches_snapshot_files():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mismatches = []
    for entry in manifest.get("files", []):
        name = entry["file"]
        expected = entry["sha256"]
        path = SNAPSHOT_DIR / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(f"{name}: expected={expected} actual={actual}")
    assert not mismatches, (
        f"{len(mismatches)} snapshot hash mismatches:\n" + "\n".join(mismatches)
    )
