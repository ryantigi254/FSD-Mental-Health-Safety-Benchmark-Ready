from __future__ import annotations

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
