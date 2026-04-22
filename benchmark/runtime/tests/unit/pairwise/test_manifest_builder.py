import json
from pathlib import Path

from reliable_clinical_benchmark.pairwise.manifest_builder import build_case_manifest


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")


def test_manifest_builder_populates_controllability_and_invariance_slices(tmp_path: Path):
    runtime_root = tmp_path

    _write_jsonl(
        runtime_root / "results" / "demo-model" / "ctrl_study_a_generations.jsonl",
        [
            {
                "id": "ctrl_a_0001",
                "arm": "spontaneous",
                "control_prompt_id": "none",
                "prompt": "Base prompt",
                "output_text": "Spontaneous answer",
                "meta": {"response_tokens": 12},
            },
            {
                "id": "ctrl_a_0001",
                "arm": "generic_control",
                "control_prompt_id": "generic",
                "prompt": "Base prompt",
                "output_text": "Generic control answer",
                "meta": {"response_tokens": 11},
            },
            {
                "id": "ctrl_a_0001",
                "arm": "explicit_control",
                "control_prompt_id": "explicit",
                "prompt": "Base prompt",
                "output_text": "Explicit control answer",
                "meta": {"response_tokens": 10},
            },
        ],
    )

    _write_jsonl(
        runtime_root
        / "processed"
        / "_archived"
        / "duplicates"
        / "study_a_cleaned"
        / "demo-model"
        / "study_a_generations.jsonl",
        [
            {
                "id": "a_0001",
                "prompt": "Base prompt",
                "output_text": "Base benchmark answer",
                "meta": {"response_tokens": 9},
            }
        ],
    )

    _write_jsonl(
        runtime_root / "results_invariance" / "demo-model" / "study_a_invariance_generations.jsonl",
        [
            {
                "id": "a_0001",
                "prompt": "Perturbed prompt",
                "output_text": "Perturbed benchmark answer",
                "meta": {"response_tokens": 8},
            }
        ],
    )

    _write_jsonl(
        runtime_root
        / "results_ctrl_invariance"
        / "demo-model"
        / "study_a_invariance_generations.jsonl",
        [
            {
                "id": "ctrl_a_0001",
                "prompt": "Perturbed control prompt",
                "output_text": "Perturbed control answer",
                "meta": {"response_tokens": 7},
            }
        ],
    )

    controllability = build_case_manifest(
        slice_id="study_a_controllability",
        runtime_root=runtime_root,
    )
    invariance = build_case_manifest(slice_id="invariance", runtime_root=runtime_root)
    invariance_under_control = build_case_manifest(
        slice_id="invariance_under_control",
        runtime_root=runtime_root,
    )
    control_under_invariance = build_case_manifest(
        slice_id="control_under_invariance",
        runtime_root=runtime_root,
    )

    assert controllability["status"] == "ready"
    assert controllability["systems"] == [
        "explicit_control",
        "generic_control",
        "spontaneous",
    ]
    assert len(controllability["cases"]) == 1

    assert invariance["status"] == "ready"
    assert invariance["systems"] == ["benchmark_base", "invariance_variant"]
    assert len(invariance["cases"]) == 1

    assert invariance_under_control["status"] == "ready"
    assert invariance_under_control["systems"] == [
        "explicit_control_base",
        "ctrl_invariance_variant",
    ]
    assert len(invariance_under_control["cases"]) == 1

    assert control_under_invariance["status"] == "ready"
    assert control_under_invariance["systems"] == [
        "explicit_control_base",
        "control_under_invariance_variant",
    ]
    assert len(control_under_invariance["cases"]) == 1


def test_manifest_builder_filters_requested_candidate_systems(tmp_path: Path):
    runtime_root = tmp_path

    _write_jsonl(
        runtime_root
        / "processed"
        / "_archived"
        / "duplicates"
        / "study_a_cleaned"
        / "keep-model"
        / "study_a_generations.jsonl",
        [
            {
                "id": "a_0001",
                "prompt": "Prompt",
                "output_text": "Keep answer",
            }
        ],
    )
    _write_jsonl(
        runtime_root
        / "processed"
        / "_archived"
        / "duplicates"
        / "study_a_cleaned"
        / "drop-model"
        / "study_a_generations.jsonl",
        [
            {
                "id": "a_0001",
                "prompt": "Prompt",
                "output_text": "Drop answer",
            }
        ],
    )

    manifest = build_case_manifest(
        slice_id="study_a",
        runtime_root=runtime_root,
        include_systems=["keep-model"],
    )

    assert manifest["status"] == "missing_inputs"
    assert manifest["systems"] == ["keep-model"]
    assert manifest["candidate_system_filter"] == ["keep-model"]
