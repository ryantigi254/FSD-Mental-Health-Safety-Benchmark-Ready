"""Contract checks for adversarial bias inclusion in v0.3 frozen/release artefacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = BASE_DIR / "data" / "frozen_splits" / "v0.3_postclinician_audit"
RELEASE_DIR = BASE_DIR / "data" / "releases" / "clinician_readiness_v0.3_2026-02-16"

SNAPSHOT_MANIFEST = SNAPSHOT_DIR / "manifest.json"
RELEASE_MANIFEST = RELEASE_DIR / "manifest.json"

EXPECTED_BIAS_FILES = {
    "adversarial_bias/biased_vignettes.json",
    "adversarial_bias/biased_vignettes_legacy_2016.json",
    "adversarial_bias/BIAS_DIMENSIONS.md",
    "adversarial_bias/README.md",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.unit
def test_v03_adversarial_bias_files_exist_in_snapshot_and_release():
    for rel in sorted(EXPECTED_BIAS_FILES):
        assert (SNAPSHOT_DIR / rel).exists(), f"Missing snapshot adversarial file: {rel}"
        assert (RELEASE_DIR / rel).exists(), f"Missing release adversarial file: {rel}"


@pytest.mark.unit
def test_v03_adversarial_bias_manifest_entries_and_hashes():
    snapshot_manifest = json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
    release_manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))

    snapshot_entries = {
        entry["file"]: entry for entry in snapshot_manifest.get("files", [])
        if str(entry.get("file", "")).startswith("adversarial_bias/")
    }
    release_entries = {
        entry["path"]: entry for entry in release_manifest.get("files", [])
        if str(entry.get("path", "")).startswith("adversarial_bias/")
    }

    assert EXPECTED_BIAS_FILES.issubset(snapshot_entries.keys()), (
        f"Snapshot manifest missing adversarial entries: "
        f"{sorted(EXPECTED_BIAS_FILES - set(snapshot_entries))}"
    )
    assert EXPECTED_BIAS_FILES.issubset(release_entries.keys()), (
        f"Release manifest missing adversarial entries: "
        f"{sorted(EXPECTED_BIAS_FILES - set(release_entries))}"
    )

    for rel in sorted(EXPECTED_BIAS_FILES):
        snap_path = SNAPSHOT_DIR / rel
        rel_path = RELEASE_DIR / rel
        assert snapshot_entries[rel]["sha256"] == _sha256(snap_path)
        assert release_entries[rel]["sha256"] == _sha256(rel_path)


@pytest.mark.unit
def test_v03_adversarial_bias_row_count_matches_cases():
    payload = json.loads((RELEASE_DIR / "adversarial_bias" / "biased_vignettes.json").read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    assert isinstance(cases, list)
    assert len(cases) == 2000

    release_manifest = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
    release_entry = next(
        entry for entry in release_manifest.get("files", [])
        if entry.get("path") == "adversarial_bias/biased_vignettes.json"
    )
    snapshot_manifest = json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
    snapshot_entry = next(
        entry for entry in snapshot_manifest.get("files", [])
        if entry.get("file") == "adversarial_bias/biased_vignettes.json"
    )

    assert release_entry["row_count"] == len(cases) == 2000
    assert snapshot_entry["row_count"] == len(cases) == 2000
