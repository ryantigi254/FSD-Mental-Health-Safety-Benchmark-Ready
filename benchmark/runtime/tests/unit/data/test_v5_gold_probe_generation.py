from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_study_a_probe_backend_can_target_v5_paths(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(
        "study_a_populate_probe",
        "scripts/studies/study_a/gold_labels/populate_from_openr1.py",
    )

    study_a_path = tmp_path / "study_a_test.json"
    study_a_path.write_text(json.dumps({"samples": []}), encoding="utf-8")
    weak_labels_path = tmp_path / "gold_diagnosis_labels.json"
    weak_labels_path.write_text(
        json.dumps({"labels": {"a_001": "Adjustment Disorder"}}),
        encoding="utf-8",
    )
    out_path = tmp_path / "gold_diagnosis_labels.out.json"

    monkeypatch.setattr(
        module,
        "load_study_a_data",
        lambda _: [{"id": "a_001", "prompt": "Patient prompt"}],
    )
    monkeypatch.setattr(
        module,
        "run_probe_labeler",
        lambda **kwargs: SimpleNamespace(
            predictions=["Adjustment Disorder"],
            confidences=[0.91],
            agreement_flags=[True],
            meta={"agreement_count": 1, "agreement_rate": 1.0},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "populate_from_openr1.py",
            "--backend",
            "probe",
            "--study-a-path",
            str(study_a_path),
            "--weak-labels",
            str(weak_labels_path),
            "--out",
            str(out_path),
            "--primary-model",
            "primary-model",
            "--secondary-model",
            "secondary-model",
        ],
    )

    assert module.main() == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["labels"] == {"a_001": "Adjustment Disorder"}
    assert payload["meta"]["backend"] == "probe"
    assert payload["meta"]["primary_model"] == "primary-model"
    assert payload["meta"]["secondary_model"] == "secondary-model"
    assert payload["meta"]["weak_labels_path"] == str(weak_labels_path)


def test_study_c_probe_backend_can_target_v5_paths(tmp_path: Path, monkeypatch) -> None:
    module = _load_module(
        "study_c_generate_probe",
        "scripts/studies/study_c/gold_plans/generate_nli_plans.py",
    )

    data_dir = tmp_path / "v5_study_c"
    data_dir.mkdir()
    (data_dir / "study_c_test.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "c_001",
                        "patient_summary": "Low mood and poor sleep",
                        "critical_entities": ["major depressive disorder", "sertraline 50 mg daily"],
                        "metadata": {"source_openr1_ids": [246], "source_split": "test"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    weak_plans_path = tmp_path / "study_c_target_plans.json"
    weak_plans_path.write_text(
        json.dumps(
            {
                "plans": {
                    "c_001": {
                        "plan": "Therapy: CBT. Follow-up: review homework. Case anchors: Problem: major depressive disorder. Constraints/Meds: sertraline 50 mg daily."
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "study_c_target_plans.out.json"

    monkeypatch.setattr(
        module,
        "run_probe_labeler",
        lambda **kwargs: SimpleNamespace(
            predictions=["major depressive disorder"],
            confidences=[0.88],
            agreement_flags=[True],
            meta={"agreement_count": 1, "agreement_rate": 1.0},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_nli_plans.py",
            "--data-dir",
            str(data_dir),
            "--out",
            str(out_path),
            "--backend",
            "probe",
            "--weak-plans",
            str(weak_plans_path),
            "--primary-model",
            "primary-model",
            "--secondary-model",
            "secondary-model",
        ],
    )

    assert module.main() == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["meta"]["backend"] == "probe"
    assert payload["meta"]["primary_model"] == "primary-model"
    assert payload["meta"]["secondary_model"] == "secondary-model"
    assert payload["meta"]["weak_plans_path"] == str(weak_plans_path)
    assert "c_001" in payload["plans"]
    assert "major depressive disorder" in payload["plans"]["c_001"]["plan"].lower()
