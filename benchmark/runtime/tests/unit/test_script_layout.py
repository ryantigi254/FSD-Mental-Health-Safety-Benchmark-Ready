from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
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


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def test_hf_local_scripts_first_main_guard_exits_before_duplicate_body() -> None:
    runtime_root = Path(__file__).resolve().parents[2]
    scripts = [
        runtime_root / "hf-local-scripts" / "run_study_a_generate_only.py",
        runtime_root / "hf-local-scripts" / "run_study_b_generate_only.py",
        runtime_root / "hf-local-scripts" / "run_study_b_multi_turn_generate_only.py",
        runtime_root / "hf-local-scripts" / "run_study_c_generate_only.py",
    ]

    for script in scripts:
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
        main_guards = [node for node in tree.body if _is_main_guard(node)]
        assert main_guards, f"Missing __main__ guard in {script}"
        first_guard = main_guards[0]
        assert isinstance(first_guard.body[0], ast.Raise), (
            f"First __main__ guard must raise SystemExit before any duplicated body in {script}"
        )


def test_vllm_aliases_do_not_reuse_hf_local_result_folders() -> None:
    runtime_root = Path(__file__).resolve().parents[2]
    scripts = [
        runtime_root / "hf-local-scripts" / "run_study_a_generate_only.py",
        runtime_root / "hf-local-scripts" / "run_study_b_generate_only.py",
        runtime_root / "hf-local-scripts" / "run_study_b_multi_turn_generate_only.py",
        runtime_root / "hf-local-scripts" / "run_study_c_generate_only.py",
    ]
    forbidden_aliases = {
        '"psyllm_gml_vllm": "psyllm-gml-local"',
        '"piaget_vllm": "piaget-8b-local"',
        '"psyche_r1_vllm": "psyche-r1-local"',
        '"psych_qwen_vllm": "psych-qwen-32b-local"',
    }

    for script in scripts:
        source = script.read_text(encoding="utf-8")
        for alias in forbidden_aliases:
            assert alias not in source, f"{script} must not map vLLM runs onto HF-local caches"


def _load_script_module(script: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_study_c_vllm_paths_do_not_partial_match_hf_local_folders(tmp_path: Path) -> None:
    runtime_root = Path(__file__).resolve().parents[2]
    script = runtime_root / "hf-local-scripts" / "run_study_c_generate_only.py"
    module = _load_script_module(script, "run_study_c_generate_only")

    for hf_local_folder in [
        "psyllm-gml-local",
        "piaget-8b-local",
        "psyche-r1-local",
        "psych-qwen-32b-local",
    ]:
        (tmp_path / hf_local_folder).mkdir()

    assert module._normalize_model_id_for_path("psyllm_gml_vllm", tmp_path) == "psyllm-gml-vllm"
    assert module._normalize_model_id_for_path("piaget_vllm", tmp_path) == "piaget-vllm"
    assert module._normalize_model_id_for_path("psyche_r1_vllm", tmp_path) == "psyche-r1-vllm"
    assert module._normalize_model_id_for_path("psych_qwen_vllm", tmp_path) == "psych-qwen-vllm"
