from __future__ import annotations

import json
import runpy
import subprocess
import sys
import types
from pathlib import Path


def test_canonical_scripts_exist() -> None:
    runtime_root = Path(__file__).resolve().parents[2]

    expected = [
        runtime_root / "scripts" / "preprocessing" / "clean_generation_outputs.py",
        runtime_root / "scripts" / "preprocessing" / "extract_predictions.py",
        runtime_root / "scripts" / "preprocessing" / "build_splits.py",
        runtime_root / "scripts" / "evaluation" / "run_metrics_pipeline.py",
        runtime_root / "scripts" / "studies" / "study_a" / "scaling" / "expand_to_2000_samples.py",
    ]

    for script in expected:
        assert script.exists(), f"Missing canonical script: {script}"


def test_removed_scripts_absent() -> None:
    runtime_root = Path(__file__).resolve().parents[2]

    removed = [
        runtime_root / "scripts" / "preprocessing" / "clean_generation_outputs_fast.py",
        runtime_root / "scripts" / "preprocessing" / "clean_generations.py",
        runtime_root / "scripts" / "preprocessing" / "clean_generations_optimized.py",
        runtime_root / "scripts" / "preprocessing" / "step2_extract_predictions.py",
        runtime_root / "scripts" / "studies" / "study_a" / "scale_to_2000.py",
        runtime_root / "scripts" / "reporting" / "generate_final_report.py",
        runtime_root / "scripts" / "reporting" / "update_leaderboard.py",
    ]

    for script in removed:
        assert not script.exists(), f"Removed script still present: {script}"


def test_core_entrypoints_expose_help() -> None:
    runtime_root = Path(__file__).resolve().parents[2]
    scripts = [
        runtime_root / "scripts" / "preprocessing" / "clean_generation_outputs.py",
        runtime_root / "scripts" / "preprocessing" / "extract_predictions.py",
        runtime_root / "scripts" / "evaluation" / "run_metrics_pipeline.py",
    ]

    for script in scripts:
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(runtime_root),
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_study_a_generate_only_has_single_entrypoint() -> None:
    runtime_root = Path(__file__).resolve().parents[2]
    script = runtime_root / "hf-local-scripts" / "run_study_a_generate_only.py"
    source = script.read_text(encoding="utf-8")

    assert source.count('if __name__ == "__main__":') == 1
    assert source.count("def _parse_args()") == 1
    assert "--quantization" in source


def test_study_a_generate_only_accepts_quantization_once(tmp_path: Path, monkeypatch) -> None:
    runtime_root = Path(__file__).resolve().parents[2]
    script = runtime_root / "hf-local-scripts" / "run_study_a_generate_only.py"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "results"
    data_dir.mkdir()
    output_dir.mkdir()
    (data_dir / "study_a_test.json").write_text(json.dumps({"samples": []}), encoding="utf-8")

    calls = {"factory": [], "run_study_a": 0}

    package = types.ModuleType("reliable_clinical_benchmark")
    package.__path__ = []
    models_package = types.ModuleType("reliable_clinical_benchmark.models")
    models_package.__path__ = []
    pipelines_package = types.ModuleType("reliable_clinical_benchmark.pipelines")
    pipelines_package.__path__ = []
    utils_package = types.ModuleType("reliable_clinical_benchmark.utils")
    utils_package.__path__ = []

    base_module = types.ModuleType("reliable_clinical_benchmark.models.base")

    class GenerationConfig:
        def __init__(self, max_tokens):
            self.max_tokens = max_tokens

    base_module.GenerationConfig = GenerationConfig

    factory_module = types.ModuleType("reliable_clinical_benchmark.models.factory")

    def get_model_runner(model_id, config, quantization=None):
        calls["factory"].append(
            {
                "model_id": model_id,
                "max_tokens": config.max_tokens,
                "quantization": quantization,
            }
        )
        return object()

    factory_module.get_model_runner = get_model_runner

    study_a_module = types.ModuleType("reliable_clinical_benchmark.pipelines.study_a")

    def run_study_a(**kwargs):
        calls["run_study_a"] += 1

    study_a_module.run_study_a = run_study_a

    worker_module = types.ModuleType("reliable_clinical_benchmark.utils.worker_runtime")

    def resolve_worker_count(requested_workers, runner, lmstudio_default, non_lm_default):
        return requested_workers or non_lm_default

    worker_module.resolve_worker_count = resolve_worker_count

    monkeypatch.setitem(sys.modules, "reliable_clinical_benchmark", package)
    monkeypatch.setitem(sys.modules, "reliable_clinical_benchmark.models", models_package)
    monkeypatch.setitem(sys.modules, "reliable_clinical_benchmark.models.base", base_module)
    monkeypatch.setitem(sys.modules, "reliable_clinical_benchmark.models.factory", factory_module)
    monkeypatch.setitem(sys.modules, "reliable_clinical_benchmark.pipelines", pipelines_package)
    monkeypatch.setitem(sys.modules, "reliable_clinical_benchmark.pipelines.study_a", study_a_module)
    monkeypatch.setitem(sys.modules, "reliable_clinical_benchmark.utils", utils_package)
    monkeypatch.setitem(sys.modules, "reliable_clinical_benchmark.utils.worker_runtime", worker_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--model-id",
            "psych_qwen_local",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--quantization",
            "4bit",
        ],
    )

    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exit_error:
        assert exit_error.code in (None, 0)

    assert calls == {
        "factory": [
            {
                "model_id": "psych_qwen_local",
                "max_tokens": 32000,
                "quantization": "4bit",
            }
        ],
        "run_study_a": 1,
    }
