"""Unit tests for Study A label policy validator script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
SCRIPT = BASE_DIR / "scripts" / "studies" / "study_a" / "validate_label_policy.py"


@pytest.mark.unit
def test_validate_label_policy_fails_for_triage_non_low_certainty(tmp_path: Path):
    labels = {"labels": {"a_1": "Adjustment Disorder"}}
    metadata = {
        "a_1": {
            "id": "a_1",
            "safety_flag": "active_suicidal_ideation",
            "certainty": "high",
            "review_status": "requires_clinician",
            "note": "test",
        }
    }
    canonical = {
        "canonical_labels": ["Adjustment Disorder"],
        "aliases": {},
    }
    labels_p = tmp_path / "labels.json"
    meta_p = tmp_path / "meta.json"
    canon_p = tmp_path / "canon.json"
    out_p = tmp_path / "policy.csv"
    labels_p.write_text(json.dumps(labels), encoding="utf-8")
    meta_p.write_text(json.dumps(metadata), encoding="utf-8")
    canon_p.write_text(json.dumps(canonical), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--labels",
            str(labels_p),
            "--metadata",
            str(meta_p),
            "--canonical-map",
            str(canon_p),
            "--output",
            str(out_p),
        ],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert out_p.exists()


@pytest.mark.unit
def test_validate_label_policy_passes_when_rules_satisfied(tmp_path: Path):
    labels = {"labels": {"a_1": "Attention-Deficit/Hyperactivity Disorder"}}
    metadata = {
        "a_1": {
            "id": "a_1",
            "safety_flag": "possible_psychotic_features",
            "certainty": "low",
            "review_status": "requires_clinician",
            "note": "test",
        }
    }
    canonical = {
        "canonical_labels": ["Attention Deficit Hyperactivity Disorder", "Psychotic features suspected"],
        "aliases": {"Attention-Deficit/Hyperactivity Disorder": "Attention Deficit Hyperactivity Disorder"},
    }
    labels_p = tmp_path / "labels.json"
    meta_p = tmp_path / "meta.json"
    canon_p = tmp_path / "canon.json"
    out_p = tmp_path / "policy.csv"
    labels_p.write_text(json.dumps(labels), encoding="utf-8")
    meta_p.write_text(json.dumps(metadata), encoding="utf-8")
    canon_p.write_text(json.dumps(canonical), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--labels",
            str(labels_p),
            "--metadata",
            str(meta_p),
            "--canonical-map",
            str(canon_p),
            "--output",
            str(out_p),
        ],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert out_p.exists()
