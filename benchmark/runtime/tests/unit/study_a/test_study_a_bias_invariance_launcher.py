from __future__ import annotations

import importlib.util
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
AUTO_SCRIPT_PATH = RUNTIME_ROOT / "scripts" / "dev" / "run_generation_auto.py"
BIAS_SCRIPT_PATH = RUNTIME_ROOT / "hf-local-scripts" / "run_study_a_bias_generate_only.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_study_a_bias_invariance_launcher_injects_invariance_defaults(monkeypatch, capsys) -> None:
    module = _load_module("run_generation_auto_test_module", AUTO_SCRIPT_PATH)

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "run_generation_auto.py",
            "--study",
            "study_a_bias_invariance",
            "--model-id",
            "gpt_oss_lmstudio",
            "--env",
            "mh-llm-benchmark-env",
            "--check-only",
        ],
    )

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "run_study_a_bias_generate_only.py" in captured.out
    assert "--study-name study_a_bias_invariance" in captured.out
    assert "--data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json" in captured.out
    assert "--output-dir results_invariance_v5" in captured.out


def test_study_a_bias_invariance_uses_dedicated_cache_name() -> None:
    module = _load_module("run_study_a_bias_generate_only_test_module", BIAS_SCRIPT_PATH)

    assert module._default_cache_name("study_a_bias") == "study_a_bias_generations.jsonl"
    assert (
        module._default_cache_name("study_a_bias_invariance")
        == "study_a_bias_invariance_generations.jsonl"
    )
    assert module._default_max_cases("study_a_bias") is None
    assert module._default_max_cases("study_a_bias_invariance") == 150


def test_study_a_bias_invariance_launcher_handles_missing_data_path_value(monkeypatch, capsys) -> None:
    module = _load_module("run_generation_auto_missing_value_test_module", AUTO_SCRIPT_PATH)

    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "run_generation_auto.py",
            "--study",
            "study_a_bias_invariance",
            "--model-id",
            "gpt_oss_lmstudio",
            "--env",
            "mh-llm-benchmark-env",
            "--data-path",
            "--output-dir",
            "results_invariance_v5",
            "--workers",
            "2",
            "--check-only",
        ],
    )

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "--data-path data/frozen_splits/v5/adversarial_bias/biased_vignettes.json" in captured.out
    assert "--output-dir results_invariance_v5" in captured.out
    assert "--workers 2" in captured.out
    assert "biased_vignettes.json results_invariance_v5 --workers" not in captured.out
