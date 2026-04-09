import json
from pathlib import Path

import pytest

from reliable_clinical_benchmark.pairwise.config import load_pairwise_run_spec
from reliable_clinical_benchmark.pairwise.rubric import validate_criteria


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _judge_manifest_v1(count: int = 4) -> dict:
    return {
        "manifest_version": "pairwise.judges.v1",
        "judges": [
            {
                "judge_id": f"judge_{index}",
                "display_name": f"Judge {index}",
                "hf_source": f"org/model-{index}",
                "local_model_id": f"local-model-{index}",
                "generation_params": {"temperature": 0.1, "max_tokens": 64, "top_p": 0.9},
            }
            for index in range(count)
        ],
    }


def _judge_manifest_v2() -> dict:
    return {
        "manifest_version": "pairwise.judges.v2",
        "judges": [
            {
                "judge_id": "primary_judge",
                "display_name": "Primary Judge",
                "hf_source": "org/primary",
                "local_model_id": "local-primary",
                "role": "primary",
                "generation_params": {"temperature": 0.1, "max_tokens": 64, "top_p": 0.9},
            },
            {
                "judge_id": "audit_judge",
                "display_name": "Audit Judge",
                "hf_source": "org/audit",
                "local_model_id": "local-audit",
                "role": "audit",
                "generation_params": {"temperature": 0.1, "max_tokens": 64, "top_p": 0.9},
            },
            {
                "judge_id": "escalation_one",
                "display_name": "Escalation One",
                "hf_source": "org/escalation-one",
                "local_model_id": "local-escalation-one",
                "role": "escalation",
                "escalation_rank": 1,
                "generation_params": {"temperature": 0.1, "max_tokens": 64, "top_p": 0.9},
            },
            {
                "judge_id": "escalation_two",
                "display_name": "Escalation Two",
                "hf_source": "org/escalation-two",
                "local_model_id": "local-escalation-two",
                "role": "escalation",
                "escalation_rank": 2,
                "generation_params": {"temperature": 0.1, "max_tokens": 64, "top_p": 0.9},
            },
        ],
    }


def _case_manifest() -> dict:
    return {
        "manifest_version": "pairwise.cases.v1",
        "slice_id": "study_a",
        "layer": "core",
        "status": "ready",
        "systems": ["model_a", "model_b"],
        "cases": [],
        "boundary_notes": [],
        "source_paths": [],
    }


def test_load_pairwise_run_spec_resolves_relative_paths(tmp_path: Path):
    manifests_dir = tmp_path / "manifests"
    configs_dir = manifests_dir / "configs"
    configs_dir.mkdir(parents=True)

    case_manifest = _write_json(manifests_dir / "study_a_case_manifest.json", _case_manifest())
    judge_manifest = _write_json(manifests_dir / "judge_panel.v2.json", _judge_manifest_v2())
    config_path = _write_json(
        configs_dir / "study_a.run_config.json",
        {
            "run_id": "pairwise_study_a_v2",
            "layer": "core",
            "slice_id": "study_a",
            "case_manifest_path": "../study_a_case_manifest.json",
            "judge_manifest_path": "../judge_panel.v2.json",
            "rubric_family": "core_communication",
            "run_mode": "stacked",
            "orders": ["AB", "BA"],
            "output_root": "../../",
        },
    )

    run_spec = load_pairwise_run_spec(config_path)

    assert Path(run_spec.config.case_manifest_path) == case_manifest.resolve()
    assert Path(run_spec.config.judge_manifest_path) == judge_manifest.resolve()
    assert Path(run_spec.config.output_root) == manifests_dir.parent.resolve()
    assert run_spec.config.run_mode == "stacked"
    assert run_spec.judge_manifest.primary_judge().judge_id == "primary_judge"


def test_load_pairwise_run_spec_rejects_fewer_than_four_judges(tmp_path: Path):
    case_manifest = _write_json(tmp_path / "study_a_case_manifest.json", _case_manifest())
    judge_manifest = _write_json(tmp_path / "judge_panel.v1.json", _judge_manifest_v1(count=3))
    config_path = _write_json(
        tmp_path / "study_a.run_config.json",
        {
            "run_id": "pairwise_study_a_v1",
            "layer": "core",
            "slice_id": "study_a",
            "case_manifest_path": str(case_manifest),
            "judge_manifest_path": str(judge_manifest),
            "rubric_family": "core_communication",
            "run_mode": "all_judges",
            "orders": ["AB", "BA"],
            "output_root": str(tmp_path / "out"),
        },
    )

    with pytest.raises(ValueError, match="exactly 4 judges"):
        load_pairwise_run_spec(config_path)


def test_load_pairwise_run_spec_rejects_missing_orders(tmp_path: Path):
    case_manifest = _write_json(tmp_path / "study_a_case_manifest.json", _case_manifest())
    judge_manifest = _write_json(tmp_path / "judge_panel.v2.json", _judge_manifest_v2())
    config_path = _write_json(
        tmp_path / "study_a.run_config.json",
        {
            "run_id": "pairwise_study_a_v2",
            "layer": "core",
            "slice_id": "study_a",
            "case_manifest_path": str(case_manifest),
            "judge_manifest_path": str(judge_manifest),
            "rubric_family": "core_communication",
            "run_mode": "stacked",
            "orders": ["AB"],
            "output_root": str(tmp_path / "out"),
        },
    )

    with pytest.raises(ValueError, match="orders must contain exactly AB and BA"):
        load_pairwise_run_spec(config_path)


def test_load_pairwise_run_spec_rejects_invalid_run_mode(tmp_path: Path):
    case_manifest = _write_json(tmp_path / "study_a_case_manifest.json", _case_manifest())
    judge_manifest = _write_json(tmp_path / "judge_panel.v2.json", _judge_manifest_v2())
    config_path = _write_json(
        tmp_path / "study_a.run_config.json",
        {
            "run_id": "pairwise_study_a_v2",
            "layer": "core",
            "slice_id": "study_a",
            "case_manifest_path": str(case_manifest),
            "judge_manifest_path": str(judge_manifest),
            "rubric_family": "core_communication",
            "run_mode": "not_real",
            "orders": ["AB", "BA"],
            "output_root": str(tmp_path / "out"),
        },
    )

    with pytest.raises(ValueError, match="unknown run_mode"):
        load_pairwise_run_spec(config_path)


def test_load_pairwise_run_spec_rejects_v1_manifest_in_stacked_mode(tmp_path: Path):
    case_manifest = _write_json(tmp_path / "study_a_case_manifest.json", _case_manifest())
    judge_manifest = _write_json(tmp_path / "judge_panel.v1.json", _judge_manifest_v1())
    config_path = _write_json(
        tmp_path / "study_a.run_config.json",
        {
            "run_id": "pairwise_study_a_v2",
            "layer": "core",
            "slice_id": "study_a",
            "case_manifest_path": str(case_manifest),
            "judge_manifest_path": str(judge_manifest),
            "rubric_family": "core_communication",
            "run_mode": "stacked",
            "orders": ["AB", "BA"],
            "output_root": str(tmp_path / "out"),
        },
    )

    with pytest.raises(ValueError, match="only valid with run_mode=all_judges"):
        load_pairwise_run_spec(config_path)


def test_load_pairwise_run_spec_rejects_bad_role_shape(tmp_path: Path):
    bad_manifest = _judge_manifest_v2()
    bad_manifest["judges"][0]["role"] = "audit"

    case_manifest = _write_json(tmp_path / "study_a_case_manifest.json", _case_manifest())
    judge_manifest = _write_json(tmp_path / "judge_panel.v2.json", bad_manifest)
    config_path = _write_json(
        tmp_path / "study_a.run_config.json",
        {
            "run_id": "pairwise_study_a_v2",
            "layer": "core",
            "slice_id": "study_a",
            "case_manifest_path": str(case_manifest),
            "judge_manifest_path": str(judge_manifest),
            "rubric_family": "core_communication",
            "run_mode": "stacked",
            "orders": ["AB", "BA"],
            "output_root": str(tmp_path / "out"),
        },
    )

    with pytest.raises(ValueError, match="exactly one primary judge"):
        load_pairwise_run_spec(config_path)


def test_load_pairwise_run_spec_rejects_unknown_escalation_reason(tmp_path: Path):
    case_manifest = _write_json(tmp_path / "study_a_case_manifest.json", _case_manifest())
    judge_manifest = _write_json(tmp_path / "judge_panel.v2.json", _judge_manifest_v2())
    config_path = _write_json(
        tmp_path / "study_a.run_config.json",
        {
            "run_id": "pairwise_study_a_v2",
            "layer": "core",
            "slice_id": "study_a",
            "case_manifest_path": str(case_manifest),
            "judge_manifest_path": str(judge_manifest),
            "rubric_family": "core_communication",
            "run_mode": "stacked",
            "orders": ["AB", "BA"],
            "escalate_on": ["disagreement", "not_real"],
            "output_root": str(tmp_path / "out"),
        },
    )

    with pytest.raises(ValueError, match="unknown escalate_on values"):
        load_pairwise_run_spec(config_path)


def test_validate_criteria_rejects_unknown_and_prohibited():
    with pytest.raises(ValueError, match="unknown criteria IDs"):
        validate_criteria(["not_real"])

    with pytest.raises(ValueError, match="prohibited primary-metric criteria"):
        validate_criteria(["clinical_correctness"])
