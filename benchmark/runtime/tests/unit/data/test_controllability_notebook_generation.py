from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_GENERATOR_PATH = (
    RUNTIME_ROOT / "scripts" / "dev" / "generate_controllability_analysis_notebooks.py"
)


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_generate_controllability_analysis_notebooks_writes_canonical_notebooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module("generate_controllability_analysis_notebooks", NOTEBOOK_GENERATOR_PATH)
    monkeypatch.setattr(module, "NOTEBOOK_DIR", tmp_path / "controlability")

    module.main()

    expected = {
        "study_a_controllability_analysis.ipynb",
        "study_a_bias_controllability_analysis.ipynb",
        "study_b_controllability_analysis.ipynb",
        "study_b_multiturn_controllability_analysis.ipynb",
        "study_c_controllability_analysis.ipynb",
        "controllability_summary_analysis.ipynb",
    }

    written = {path.name for path in (tmp_path / "controlability").glob("*.ipynb")}
    assert expected <= written

    payload = json.loads((tmp_path / "controlability" / "study_a_bias_controllability_analysis.ipynb").read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    joined_sources = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
    )
    assert "load_controllability_study_payload" in joined_sources
    assert "spontaneous" in joined_sources
