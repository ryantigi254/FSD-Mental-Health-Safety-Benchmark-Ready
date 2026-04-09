from reliable_clinical_benchmark.pairwise.config import (
    JudgeManifest,
    JudgeManifestEntry,
    PairwiseConfig,
    PairwiseRunSpec,
)
from reliable_clinical_benchmark.pairwise.statistics import (
    case_stratified_bootstrap,
    compute_bradley_terry,
    compute_execution_summary,
    compute_swap_consistency,
    compute_verbosity_bias,
    compute_win_rates,
)


def _record(
    *,
    case_id: str,
    judge_id: str,
    order: str,
    winner: str,
    system_a: str = "model_a",
    system_b: str = "model_b",
    criterion_id: str = "clarity",
    is_tie: bool = False,
    is_invalid: bool = False,
    len_a: int = 10,
    len_b: int = 5,
    comparison_key: str | None = None,
    judge_stage: str = "routine",
    comparison_outcome: str = "resolved_routine",
    escalation_reason: list[str] | None = None,
    high_risk_forced: bool = False,
    judge_role: str = "panel",
):
    return {
        "case_id": case_id,
        "layer": "core",
        "slice_id": "study_a",
        "criterion_id": criterion_id,
        "judge_id": judge_id,
        "order": order,
        "system_a": system_a,
        "system_b": system_b,
        "winner": winner,
        "is_tie": is_tie,
        "is_invalid": is_invalid,
        "response_length_a": len_a,
        "response_length_b": len_b,
        "canonical_pair_key": "__vs__".join(sorted([system_a, system_b])),
        "comparison_key": comparison_key or f"{case_id}::{criterion_id}::{system_a}__vs__{system_b}",
        "judge_stage": judge_stage,
        "comparison_outcome": comparison_outcome,
        "escalation_reason": escalation_reason or [],
        "high_risk_forced": high_risk_forced,
        "judge_role": judge_role,
    }


def _stacked_run_spec() -> PairwiseRunSpec:
    return PairwiseRunSpec(
        config=PairwiseConfig(
            run_id="pairwise_study_a_v2",
            layer="core",
            slice_id="study_a",
            case_manifest_path="/tmp/cases.json",
            judge_manifest_path="/tmp/judges.json",
            rubric_family="core_communication",
            run_mode="stacked",
            output_root="/tmp/pairwise",
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


def test_compute_win_rates_and_bradley_terry():
    records = [
        _record(case_id="c1", judge_id="j1", order="AB", winner="model_a"),
        _record(case_id="c1", judge_id="j1", order="BA", winner="model_a"),
        _record(case_id="c2", judge_id="j1", order="AB", winner="model_b"),
        _record(case_id="c2", judge_id="j1", order="BA", winner="model_b"),
    ]

    rows = compute_win_rates(records)
    assert len(rows) == 1
    assert rows[0]["wins_a"] == 2
    assert rows[0]["wins_b"] == 2

    bt_rows = compute_bradley_terry(records)
    assert {row["system_id"] for row in bt_rows} == {"model_a", "model_b"}


def test_compute_swap_consistency_and_verbosity_bias():
    records = [
        _record(case_id="c1", judge_id="j1", order="AB", winner="model_a", len_a=12, len_b=4),
        _record(case_id="c1", judge_id="j1", order="BA", winner="model_a", len_a=12, len_b=4),
        _record(case_id="c2", judge_id="j1", order="AB", winner="model_a", len_a=7, len_b=8),
        _record(case_id="c2", judge_id="j1", order="BA", winner="model_b", len_a=7, len_b=8),
    ]

    swap = compute_swap_consistency(records)
    assert swap["consistent"] == 1
    assert swap["inconsistent"] == 1

    verbosity = compute_verbosity_bias(records)
    assert verbosity["n"] == 4
    assert 0.0 <= verbosity["winner_longer_rate"] <= 1.0


def test_case_stratified_bootstrap_returns_interval():
    records = [
        _record(case_id="c1", judge_id="j1", order="AB", winner="model_a"),
        _record(case_id="c2", judge_id="j1", order="AB", winner="model_a"),
        _record(case_id="c3", judge_id="j1", order="AB", winner="model_b"),
        _record(case_id="c4", judge_id="j1", order="AB", winner="model_b"),
    ]

    interval = case_stratified_bootstrap(
        records,
        lambda sample: sum(1 for row in sample if row["winner"] == "model_a") / len(sample),
        n_bootstrap=50,
    )
    assert interval[0] <= interval[1]


def test_compute_execution_summary_reports_routine_and_escalated_cases():
    run_spec = _stacked_run_spec()
    routine_key = "routine_case"
    escalated_key = "escalated_case"
    records = [
        _record(
            case_id="c1",
            judge_id="primary_judge",
            judge_role="primary",
            order="AB",
            winner="model_a",
            comparison_key=routine_key,
        ),
        _record(
            case_id="c1",
            judge_id="primary_judge",
            judge_role="primary",
            order="BA",
            winner="model_a",
            comparison_key=routine_key,
        ),
        _record(
            case_id="c1",
            judge_id="audit_judge",
            judge_role="audit",
            order="AB",
            winner="model_a",
            comparison_key=routine_key,
        ),
        _record(
            case_id="c1",
            judge_id="audit_judge",
            judge_role="audit",
            order="BA",
            winner="model_a",
            comparison_key=routine_key,
        ),
        _record(
            case_id="c2",
            judge_id="primary_judge",
            judge_role="primary",
            order="AB",
            winner="model_a",
            comparison_key=escalated_key,
            judge_stage="routine",
            comparison_outcome="uncertain",
            escalation_reason=["disagreement"],
        ),
        _record(
            case_id="c2",
            judge_id="primary_judge",
            judge_role="primary",
            order="BA",
            winner="model_a",
            comparison_key=escalated_key,
            judge_stage="routine",
            comparison_outcome="uncertain",
            escalation_reason=["disagreement"],
        ),
        _record(
            case_id="c2",
            judge_id="audit_judge",
            judge_role="audit",
            order="AB",
            winner="model_b",
            comparison_key=escalated_key,
            judge_stage="routine",
            comparison_outcome="uncertain",
            escalation_reason=["disagreement"],
        ),
        _record(
            case_id="c2",
            judge_id="audit_judge",
            judge_role="audit",
            order="BA",
            winner="model_b",
            comparison_key=escalated_key,
            judge_stage="routine",
            comparison_outcome="uncertain",
            escalation_reason=["disagreement"],
        ),
        _record(
            case_id="c2",
            judge_id="escalation_one",
            judge_role="escalation",
            order="AB",
            winner="model_a",
            comparison_key=escalated_key,
            judge_stage="escalated",
            comparison_outcome="uncertain",
            escalation_reason=["disagreement"],
        ),
        _record(
            case_id="c2",
            judge_id="escalation_one",
            judge_role="escalation",
            order="BA",
            winner="model_a",
            comparison_key=escalated_key,
            judge_stage="escalated",
            comparison_outcome="uncertain",
            escalation_reason=["disagreement"],
        ),
        _record(
            case_id="c2",
            judge_id="escalation_two",
            judge_role="escalation",
            order="AB",
            winner="model_b",
            comparison_key=escalated_key,
            judge_stage="escalated",
            comparison_outcome="uncertain",
            escalation_reason=["disagreement"],
        ),
        _record(
            case_id="c2",
            judge_id="escalation_two",
            judge_role="escalation",
            order="BA",
            winner="model_b",
            comparison_key=escalated_key,
            judge_stage="escalated",
            comparison_outcome="uncertain",
            escalation_reason=["disagreement"],
        ),
    ]

    summary = compute_execution_summary(records, run_spec=run_spec)

    assert summary["routine_two_judge_results"]["count"] == 1
    assert summary["escalated_four_judge_results"]["count"] == 1
    assert summary["uncertain_case_count"] == 1
    assert summary["primary_audit_agreement"]["n"] == 2
    assert summary["primary_audit_agreement"]["agreement_rate"] == 0.5
    assert summary["all_judge_agreement"]["n"] == 1
    assert summary["all_judge_agreement"]["agreement_rate"] == 0.0
    assert len(summary["persistent_disagreement_cases"]) == 1
