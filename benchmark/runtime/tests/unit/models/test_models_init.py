from __future__ import annotations

import importlib


def test_models_package_import_is_lightweight():
    module = importlib.import_module("reliable_clinical_benchmark.models")
    assert hasattr(module, "ModelRunner")
    assert hasattr(module, "GenerationConfig")
    assert callable(module.get_model_runner)
