from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_variant_family_matrix_for_selected_study_and_variant(tmp_path: Path):
    base_root = tmp_path / "base"
    out_root = tmp_path / "variants"
    _write_json(
        base_root / "study_c_test.json",
        {
            "cases": [
                {
                    "id": "c_001",
                    "patient_summary": "Sentence one. Sentence two. Sentence three.",
                    "critical_entities": ["major depressive disorder"],
                    "turns": [
                        {"turn": 1, "message": "I am tired most mornings."},
                        {"turn": 2, "message": "I have missed work twice this week."},
                    ],
                    "metadata": {"persona_id": "aisha", "source_openr1_ids": [1], "source_split": "test"},
                }
            ]
        },
    )
    command = [
        sys.executable,
        str(RUNTIME_ROOT / "scripts" / "invariance" / "build_variant_family_matrix.py"),
        "--base-root",
        str(base_root),
        "--output-root",
        str(out_root),
        "--study",
        "study_c",
        "--variant",
        "summary_short",
    ]
    subprocess.run(command, check=True, cwd=str(RUNTIME_ROOT))

    manifest = json.loads((out_root / "variant_matrix_manifest.json").read_text())
    assert manifest["sampling_policy"] == "sample_once_then_fan_out_variants"
    variants = manifest["studies"]["study_c"]["variants"]
    assert [entry["variant"] for entry in variants] == ["summary_short"]
    assert (out_root / "study_c" / "summary_short" / "study_c_test.json").exists()


def test_build_variant_family_matrix_accepts_relative_roots(tmp_path: Path):
    base_root = tmp_path / "base"
    out_root = tmp_path / "variants"
    _write_json(
        base_root / "study_c_test.json",
        {
            "cases": [
                {
                    "id": "c_001",
                    "patient_summary": "Sentence one. Sentence two. Sentence three.",
                    "critical_entities": ["major depressive disorder"],
                    "turns": [
                        {"turn": 1, "message": "I am tired most mornings."},
                        {"turn": 2, "message": "I have missed work twice this week."},
                    ],
                    "metadata": {"persona_id": "aisha", "source_openr1_ids": [1], "source_split": "test"},
                }
            ]
        },
    )
    command = [
        sys.executable,
        str(RUNTIME_ROOT / "scripts" / "invariance" / "build_variant_family_matrix.py"),
        "--base-root",
        os.path.relpath(base_root, RUNTIME_ROOT),
        "--output-root",
        os.path.relpath(out_root, RUNTIME_ROOT),
        "--study",
        "study_c",
        "--variant",
        "summary_short",
    ]
    subprocess.run(command, check=True, cwd=str(RUNTIME_ROOT))

    manifest = json.loads((out_root / "variant_matrix_manifest.json").read_text())
    assert manifest["base_root"] == str(base_root.resolve())
    assert (out_root / "study_c" / "summary_short" / "study_c_test.json").exists()
