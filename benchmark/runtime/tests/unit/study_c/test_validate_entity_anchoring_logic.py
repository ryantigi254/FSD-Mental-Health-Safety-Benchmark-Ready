"""Regression tests for Study C anchoring validator control flow and evidence rules."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]
SCRIPT_PATH = BASE_DIR / "scripts" / "studies" / "study_c" / "validate_entity_anchoring.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_entity_anchoring", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_case_evidence_span_must_exist_in_summary(tmp_path: Path):
    """Non-empty case evidence must still be a real substring of patient_summary."""
    validator = _load_validator_module()

    cases_path = tmp_path / "cases.json"
    evidence_path = tmp_path / "evidence.json"

    cases_payload = {
        "cases": [
            {
                "id": "c_x01",
                "patient_summary": "Patient reports stable mood and good sleep.",
                "critical_entities": ["functional impairment"],
            }
        ]
    }
    evidence_payload = {
        "global_synonyms": {},
        "case_evidence": {
            "c_x01": {
                "functional impairment": "this text is non-empty but not in summary",
            }
        },
    }

    cases_path.write_text(json.dumps(cases_payload), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence_payload), encoding="utf-8")

    rows = validator.validate(cases_path, evidence_path)
    assert len(rows) == 1
    assert rows[0]["anchored"] is False
    assert rows[0]["method"] == ""


@pytest.mark.unit
def test_case_evidence_fallback_runs_when_synonym_patterns_miss(tmp_path: Path):
    """Case evidence fallback must still run when synonym patterns do not match."""
    validator = _load_validator_module()

    cases_path = tmp_path / "cases.json"
    evidence_path = tmp_path / "evidence.json"

    summary_text = "Patient has reduced day-to-day functioning at work or study."
    cases_payload = {
        "cases": [
            {
                "id": "c_x02",
                "patient_summary": summary_text,
                "critical_entities": ["functional impairment"],
            }
        ]
    }
    evidence_payload = {
        "global_synonyms": {
            "functional impairment": {
                "type": "semantic",
                "evidence_patterns": ["pattern that does not occur"],
            }
        },
        "case_evidence": {
            "c_x02": {
                "functional impairment": "reduced day-to-day functioning at work or study",
            }
        },
    }

    cases_path.write_text(json.dumps(cases_payload), encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence_payload), encoding="utf-8")

    rows = validator.validate(cases_path, evidence_path)
    assert len(rows) == 1
    assert rows[0]["anchored"] is True
    assert rows[0]["method"] == "case_evidence"
