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
            run_id="pairwise_study_a_v2",
            layer="core",
            slice_id="study_a",
            case_manifest_path=str(tmp_path / "cases.json"),
            judge_manifest_path=str(tmp_path / "judges.json"),
            rubric_family="core_communication",
            run_mode="stacked",
            output_root=str(tmp_path / "metric-results" / "pairwise"),
        ),
        judge_manifest=JudgeManifest(
            judges=[
                JudgeManifestEntry(
                    judge_id="primary_judge",
                    display_name="Primary Judge",
                    hf_source="org/primary",
                    local_model_id="local-primary",
                    role="primary",
                ),
                JudgeManifestEntry(
                    judge_id="audit_judge",
                    display_name="Audit Judge",
                    hf_source="org/audit",
                    local_model_id="local-audit",
                    role="audit",
                ),
                JudgeManifestEntry(
                    judge_id="escalation_one",
                    display_name="Escalation One",
                    hf_source="org/escalation-one",
                    local_model_id="local-escalation-one",
                    role="escalation",
                    escalation_rank=1,
                ),
                JudgeManifestEntry(
                    judge_id="escalation_two",
                    display_name="Escalation Two",
                    hf_source="org/escalation-two",
                    local_model_id="local-escalation-two",
                    role="escalation",
                    escalation_rank=2,
                ),
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
            "judge_id": "primary_judge",
            "judge_role": "primary",
            "order": "AB",
            "system_a": "model_a",
            "system_b": "model_b",
            "winner": "model_a",
            "is_tie": False,
            "is_invalid": False,
            "response_length_a": 8,
            "response_length_b": 4,
            "canonical_pair_key": "model_a__vs__model_b",
            "comparison_key": "case_1::clarity::model_a__vs__model_b",
            "judge_stage": "routine",
            "comparison_outcome": "resolved_routine",
            "escalation_reason": [],
            "high_risk_forced": False,
        }
    ]

    aggregate = PairwiseAggregator(run_spec=run_spec, case_manifest=case_manifest).aggregate(records)

    assert not aggregate["pooled_complete"]
    assert aggregate["pooled"]["status"] == "incomplete"
    assert aggregate["execution_summary"]["mode"] == "stacked"
