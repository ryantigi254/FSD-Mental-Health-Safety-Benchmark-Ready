"""Data source routing tests for Study C metric script."""

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
        / "study_c"
        / "metrics"
        / "calculate_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("study_c_metrics_script", script_path)
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


class _DummyMedicalNER:
    def extract_entities(self, _text: str):
        return set()


@pytest.mark.unit
def test_study_c_metrics_defaults_to_latest_release(monkeypatch, tmp_path: Path):
    module = _load_module()
    captured: dict[str, Path] = {}

    def fake_load_gold_data(data_dir: Path):
        captured["gold_data_dir"] = Path(data_dir)
        return {}

    monkeypatch.setattr(module, "load_gold_data", fake_load_gold_data)
    monkeypatch.setattr(module, "MedicalNER", _DummyMedicalNER)
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
    assert captured["gold_data_dir"] == expected_root


@pytest.mark.unit
def test_study_c_metrics_can_use_working_data(monkeypatch, tmp_path: Path):
    module = _load_module()
    captured: dict[str, Path] = {}

    def fake_load_gold_data(data_dir: Path):
        captured["gold_data_dir"] = Path(data_dir)
        return {}

    monkeypatch.setattr(module, "load_gold_data", fake_load_gold_data)
    monkeypatch.setattr(module, "MedicalNER", _DummyMedicalNER)
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
    assert captured["gold_data_dir"] == expected_root
