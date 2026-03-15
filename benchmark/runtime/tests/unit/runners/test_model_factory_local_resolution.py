"""Unit tests for local model path resolution in the model factory."""

from pathlib import Path

import pytest

from reliable_clinical_benchmark.models import factory


@pytest.mark.unit
def test_resolve_local_model_path_prefers_runtime_models_dir(tmp_path, monkeypatch):
    runtime_root = tmp_path / "runtime"
    model_dir = runtime_root / "models" / "PsyLLM"
    model_dir.mkdir(parents=True)

    monkeypatch.setattr(factory, "EXTERNAL_MODELS_ROOT", tmp_path / "external")
    monkeypatch.setattr(factory, "RUNTIME_ROOT", runtime_root)

    resolved = factory._resolve_local_model_path("PsyLLM")

    assert resolved == str(model_dir)


@pytest.mark.unit
def test_resolve_local_model_path_falls_back_to_public_psyllm_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(factory, "EXTERNAL_MODELS_ROOT", tmp_path / "external")
    monkeypatch.setattr(factory, "RUNTIME_ROOT", tmp_path / "runtime")

    resolved = factory._resolve_local_model_path("PsyLLM")

    assert resolved == factory.PSYLLM_GML_HF_REPO_ID
