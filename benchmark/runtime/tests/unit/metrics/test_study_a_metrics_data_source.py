"""Data source routing tests for Study A metric script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re

import pytest

BASE_DIR = Path(__file__).resolve().parents[3]


def _load_module():
    script_path = (
        BASE_DIR
        / "scripts"
        / "studies"
        / "study_a"
        / "metrics"
        / "calculate_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("study_a_metrics_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _latest_release_name() -> str:
    latest_path = BASE_DIR / "data" / "releases" / "LATEST.md"
    text = latest_path.read_text(encoding="utf-8")
    match = re.search(r"Current canonical release:\s*`([^`]+)`", text)
    assert match, f"Unable to parse release name from {latest_path}"
    return match.group(1)


@pytest.mark.unit
def test_study_a_metrics_defaults_to_latest_release(monkeypatch, tmp_path: Path):
    module = _load_module()
    captured: dict[str, Path] = {}

    def fake_load_study_a_data(data_path: str, gold_diagnosis_labels_path: str = None, **_kwargs):
        captured["study_a_path"] = Path(data_path)
        captured["labels_path"] = Path(gold_diagnosis_labels_path)
        return []

    monkeypatch.setattr(module, "load_study_a_data", fake_load_study_a_data)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "calculate_metrics.py",
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "__missing__",
        ],
    )

    rc = module.main()
    assert rc == 0

    release_name = _latest_release_name()
    expected_root = BASE_DIR / "data" / "releases" / release_name
    assert captured["study_a_path"] == expected_root / "openr1_psy_splits" / "study_a_test.json"
    assert captured["labels_path"] == expected_root / "study_a_gold" / "gold_diagnosis_labels.json"


@pytest.mark.unit
def test_study_a_metrics_can_use_working_data(monkeypatch, tmp_path: Path):
    module = _load_module()
    captured: dict[str, Path] = {}

    def fake_load_study_a_data(data_path: str, gold_diagnosis_labels_path: str = None, **_kwargs):
        captured["study_a_path"] = Path(data_path)
        captured["labels_path"] = Path(gold_diagnosis_labels_path)
        return []

    monkeypatch.setattr(module, "load_study_a_data", fake_load_study_a_data)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "calculate_metrics.py",
            "--data-source",
            "working_data",
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            "__missing__",
        ],
    )

    rc = module.main()
    assert rc == 0

    expected_root = BASE_DIR / "data"
    assert captured["study_a_path"] == expected_root / "openr1_psy_splits" / "study_a_test.json"
    assert captured["labels_path"] == expected_root / "study_a_gold" / "gold_diagnosis_labels.json"
