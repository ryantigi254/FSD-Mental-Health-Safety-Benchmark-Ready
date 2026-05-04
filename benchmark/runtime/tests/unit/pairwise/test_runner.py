from reliable_clinical_benchmark.pairwise.config import (
    JudgeManifest,
    JudgeManifestEntry,
    PairwiseConfig,
    PairwiseRunSpec,
)
from reliable_clinical_benchmark.pairwise.runner import (
    PairwiseRunner,
    determine_stacked_escalation_reasons,
    resolve_stacked_outcome,
    summarise_judge_orders,
)


def _record(order: str, *, winner: str, is_tie: bool = False, is_invalid: bool = False) -> dict:
    return {
        "order": order,
        "winner": winner,
        "is_tie": is_tie,
        "is_invalid": is_invalid,
    }


def _stacked_runner() -> PairwiseRunner:
    run_spec = PairwiseRunSpec(
        config=PairwiseConfig(
            run_id="pairwise_study_a_v3",
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
    case_manifest = {
        "status": "ready",
        "cases": [
            {
                "case_id": "routine_case",
                "context": "Routine case.",
                "tags": [],
                "responses": [
                    {"system_id": "model_a", "text": "A"},
                    {"system_id": "model_b", "text": "B"},
                ],
                "pairings": [{"system_a": "model_a", "system_b": "model_b"}],
            },
            {
                "case_id": "disagreement_case",
                "context": "Disagreement case.",
                "tags": [],
                "responses": [
                    {"system_id": "model_a", "text": "A"},
                    {"system_id": "model_b", "text": "B"},
                ],
                "pairings": [{"system_a": "model_a", "system_b": "model_b"}],
            },
        ],
    }
    return PairwiseRunner(run_spec=run_spec, case_manifest=case_manifest)


def test_summarise_judge_orders_detects_swap_failure():
    summary = summarise_judge_orders(
        [
            _record("AB", winner="model_a"),
            _record("BA", winner="model_b"),
        ]
    )
    assert summary["status"] == "swap_failure"
    assert summary["winner"] is None


def test_determine_stacked_escalation_reasons_captures_disagreement_and_high_risk():
    reasons = determine_stacked_escalation_reasons(
        primary_summary={"status": "decisive", "winner": "model_a"},
        audit_summary={"status": "decisive", "winner": "model_b"},
        high_risk_forced=True,
        escalate_on=["disagreement", "tie", "invalid", "swap_failure", "high_risk"],
    )
    assert reasons == ["disagreement", "high_risk"]


def test_resolve_stacked_outcome_returns_uncertain_on_split():
    outcome = resolve_stacked_outcome(
        all_summaries={
            "primary": {"status": "decisive", "winner": "model_a"},
            "audit": {"status": "decisive", "winner": "model_a"},
            "escalation_1": {"status": "decisive", "winner": "model_b"},
            "escalation_2": {"status": "decisive", "winner": "model_a"},
        },
        persistent_disagreement_policy="mark_uncertain",
    )
    assert outcome == "uncertain"


def test_pending_groups_for_escalation_only_include_disputed_cases():
    runner = _stacked_runner()
    primary = runner.run_spec.judge_manifest.primary_judge()
    audit = runner.run_spec.judge_manifest.audit_judge()
    escalation = runner.run_spec.judge_manifest.escalation_judges()[0]

    existing = [
        {
            "slice_id": "study_a",
            "comparison_key": "routine_case::clarity::model_a__vs__model_b",
            "judge_id": primary.judge_id,
            "order": "AB",
            "winner": "model_a",
            "is_tie": False,
            "is_invalid": False,
        },
        {
            "slice_id": "study_a",
            "comparison_key": "routine_case::clarity::model_a__vs__model_b",
            "judge_id": primary.judge_id,
            "order": "BA",
            "winner": "model_a",
            "is_tie": False,
            "is_invalid": False,
        },
        {
            "slice_id": "study_a",
            "comparison_key": "routine_case::clarity::model_a__vs__model_b",
            "judge_id": audit.judge_id,
            "order": "AB",
            "winner": "model_a",
            "is_tie": False,
            "is_invalid": False,
        },
        {
            "slice_id": "study_a",
            "comparison_key": "routine_case::clarity::model_a__vs__model_b",
            "judge_id": audit.judge_id,
            "order": "BA",
            "winner": "model_a",
            "is_tie": False,
            "is_invalid": False,
        },
        {
            "slice_id": "study_a",
            "comparison_key": "disagreement_case::clarity::model_a__vs__model_b",
            "judge_id": primary.judge_id,
            "order": "AB",
            "winner": "model_a",
            "is_tie": False,
            "is_invalid": False,
        },
        {
            "slice_id": "study_a",
            "comparison_key": "disagreement_case::clarity::model_a__vs__model_b",
            "judge_id": primary.judge_id,
            "order": "BA",
            "winner": "model_a",
            "is_tie": False,
            "is_invalid": False,
        },
        {
            "slice_id": "study_a",
            "comparison_key": "disagreement_case::clarity::model_a__vs__model_b",
            "judge_id": audit.judge_id,
            "order": "AB",
            "winner": "model_b",
            "is_tie": False,
            "is_invalid": False,
        },
        {
            "slice_id": "study_a",
            "comparison_key": "disagreement_case::clarity::model_a__vs__model_b",
            "judge_id": audit.judge_id,
            "order": "BA",
            "winner": "model_b",
            "is_tie": False,
            "is_invalid": False,
        },
    ]

    pending = runner.pending_groups_for_judge(
        judge=escalation,
        existing_parsed_records=existing,
    )

    assert len(pending) == 1
    assert pending[0]["comparison_key"] == "disagreement_case::clarity::model_a__vs__model_b"


def test_run_groups_for_judge_supports_workers(monkeypatch, tmp_path):
    runner = _stacked_runner()
    runner.output_root = tmp_path / "pairwise"
    runner.raw_dir = runner.output_root / "raw" / runner.run_spec.config.run_id
    runner.parsed_dir = runner.output_root / "parsed" / runner.run_spec.config.run_id
    runner.raw_dir.mkdir(parents=True)
    runner.parsed_dir.mkdir(parents=True)
    judge = runner.run_spec.judge_manifest.primary_judge()
    groups = runner.comparison_groups()

    def fake_run_orders(*, comparison, judge):
        raw = {
            "judge_id": judge.judge_id,
            "slice_id": runner.run_spec.config.slice_id,
            "comparison_key": comparison["comparison_key"],
        }
        parsed = {
            "judge_id": judge.judge_id,
            "slice_id": runner.run_spec.config.slice_id,
            "comparison_key": comparison["comparison_key"],
            "order": "AB",
        }
        return [raw], [parsed]

    monkeypatch.setattr(runner, "_run_orders_for_judge", fake_run_orders)

    raw_records, parsed_records = runner.run_groups_for_judge(
        comparison_groups=groups,
        judge=judge,
        workers=2,
    )

    assert [record["comparison_key"] for record in raw_records] == [
        group["comparison_key"] for group in groups
    ]
    assert len(parsed_records) == len(groups)
    assert (runner.raw_dir / f"{judge.judge_id}__study_a.jsonl").exists()
    assert (runner.parsed_dir / f"{judge.judge_id}__study_a.jsonl").exists()
