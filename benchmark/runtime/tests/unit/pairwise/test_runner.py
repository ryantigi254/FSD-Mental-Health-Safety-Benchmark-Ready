from reliable_clinical_benchmark.pairwise.runner import (
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
