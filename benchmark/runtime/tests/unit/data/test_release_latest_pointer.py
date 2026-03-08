"""Checks for the canonical release pointer document."""

from __future__ import annotations

from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
LATEST_PATH = BASE_DIR / "data" / "releases" / "LATEST.md"
EXPECTED_RELEASE = "clinician_readiness_v0.3_2026-02-16"


@pytest.mark.unit
def test_latest_release_pointer_matches_v03():
    assert LATEST_PATH.exists(), f"Missing latest release pointer: {LATEST_PATH}"
    text = LATEST_PATH.read_text(encoding="utf-8")
    assert EXPECTED_RELEASE in text
