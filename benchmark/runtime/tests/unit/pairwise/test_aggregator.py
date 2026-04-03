from pathlib import Path

from reliable_clinical_benchmark.pairwise.aggregator import PairwiseAggregator
from reliable_clinical_benchmark.pairwise.config import (
    JudgeManifest,
    JudgeManifestEntry,
    PairwiseConfig,
    PairwiseRunSpec,
)


def _run_spec(tmp_path: Path) -> PairwiseRunSpec:
    return PairwiseRunSpec(
        config=PairwiseConfig(
            run_id="pairwise_study_a_v1",
            layer="core",
            slice_id="study_a",
            case_manifest_path=str(tmp_path / "cases.json"),
            judge_manifest_path=str(tmp_path / "judges.json"),
            rubric_family="core_communication",
            output_root=str(tmp_path / "metric-results" / "pairwise"),
        ),
        judge_manifest=JudgeManifest(
            judges=[
                JudgeManifestEntry(
                    judge_id=f"judge_{index}",
                    display_name=f"Judge {index}",
                    hf_source=f"org/model-{index}",
                    local_model_id=f"local-model-{index}",
                )
                for index in range(4)
            ]
        ),
        criteria=["clarity"],
    )


def test_aggregator_marks_pooled_incomplete_when_a_judge_is_missing(tmp_path: Path):
    run_spec = _run_spec(tmp_path)
    case_manifest = {
        "status": "ready",
        "cases": [{"case_id": "case_1"}],
        "systems": ["model_a", "model_b"],
        "boundary_notes": [],
    }
    records = [
        {
            "case_id": "case_1",
            "layer": "core",
            "slice_id": "study_a",
            "criterion_id": "clarity",
            "judge_id": "judge_0",
            "order": "AB",
            "system_a": "model_a",
            "system_b": "model_b",
            "winner": "model_a",
            "is_tie": False,
            "is_invalid": False,
            "response_length_a": 8,
            "response_length_b": 4,
            "canonical_pair_key": "model_a__vs__model_b",
        }
    ]

    aggregate = PairwiseAggregator(run_spec=run_spec, case_manifest=case_manifest).aggregate(records)

    assert not aggregate["pooled_complete"]
    assert aggregate["pooled"]["status"] == "incomplete"
