from reliable_clinical_benchmark.pairwise.judge_audit import (
    JudgeAuditManifest,
    JudgeAuditItem,
    compute_judge_audit_report,
)


def _record(
    *,
    audit_item_id: str,
    judge_id: str,
    order: str,
    winner: str,
    is_tie: bool = False,
    is_invalid: bool = False,
) -> dict:
    return {
        "audit_item_id": audit_item_id,
        "judge_id": judge_id,
        "order": order,
        "winner": winner,
        "is_tie": is_tie,
        "is_invalid": is_invalid,
        "system_a": "system_a",
        "system_b": "system_b",
    }


def test_compute_judge_audit_report_scores_each_metric_family():
    manifest = JudgeAuditManifest(
        items=[
            JudgeAuditItem(
                audit_item_id="easy_base",
                case_id="case_easy",
                criterion_id="clarity",
                canonical_pair_key="model_a__vs__model_b",
                gold_label="system_a",
                difficulty="easy",
            ),
            JudgeAuditItem(
                audit_item_id="easy_prompt_variant",
                case_id="case_easy",
                criterion_id="clarity",
                canonical_pair_key="model_a__vs__model_b",
                gold_label="system_a",
                difficulty="easy",
                prompt_variant_group="prompt_case",
                prompt_variant_id="variant_b",
            ),
            JudgeAuditItem(
                audit_item_id="repeat_one",
                case_id="case_repeat",
                criterion_id="clarity",
                canonical_pair_key="model_a__vs__model_b",
                gold_label="system_b",
                difficulty="easy",
                repeat_group="repeat_case",
                repeat_id="run_1",
            ),
            JudgeAuditItem(
                audit_item_id="repeat_two",
                case_id="case_repeat",
                criterion_id="clarity",
                canonical_pair_key="model_a__vs__model_b",
                gold_label="system_b",
                difficulty="easy",
                repeat_group="repeat_case",
                repeat_id="run_2",
            ),
            JudgeAuditItem(
                audit_item_id="ambiguous_case",
                case_id="case_ambiguous",
                criterion_id="clarity",
                canonical_pair_key="model_a__vs__model_b",
                gold_label="TIE",
                difficulty="ambiguous",
            ),
        ]
    )

    records = [
        _record(audit_item_id="easy_base", judge_id="judge_one", order="AB", winner="system_a"),
        _record(audit_item_id="easy_base", judge_id="judge_one", order="BA", winner="system_a"),
        _record(audit_item_id="easy_prompt_variant", judge_id="judge_one", order="AB", winner="system_a"),
        _record(audit_item_id="easy_prompt_variant", judge_id="judge_one", order="BA", winner="system_a"),
        _record(audit_item_id="repeat_one", judge_id="judge_one", order="AB", winner="system_b"),
        _record(audit_item_id="repeat_one", judge_id="judge_one", order="BA", winner="system_b"),
        _record(audit_item_id="repeat_two", judge_id="judge_one", order="AB", winner="system_b"),
        _record(audit_item_id="repeat_two", judge_id="judge_one", order="BA", winner="system_b"),
        _record(audit_item_id="ambiguous_case", judge_id="judge_one", order="AB", winner="TIE", is_tie=True),
        _record(audit_item_id="ambiguous_case", judge_id="judge_one", order="BA", winner="TIE", is_tie=True),
    ]

    report = compute_judge_audit_report(records=records, manifest=manifest)
    metrics = report["per_judge"]["judge_one"]

    assert metrics["gold_agreement"]["n"] == 4
    assert metrics["gold_agreement"]["agreement_rate"] == 1.0
    assert metrics["swap_consistency"]["n"] == 5
    assert metrics["swap_consistency"]["agreement_rate"] == 1.0
    assert metrics["prompt_invariance"]["n"] == 1
    assert metrics["prompt_invariance"]["agreement_rate"] == 1.0
    assert metrics["sensitivity"]["n"] == 4
    assert metrics["sensitivity"]["agreement_rate"] == 1.0
    assert metrics["stability"]["n"] == 1
    assert metrics["stability"]["agreement_rate"] == 1.0
    assert metrics["calibration_proxy"]["ambiguous_uncertainty_rate"]["agreement_rate"] == 1.0
    assert metrics["calibration_proxy"]["easy_uncertainty_rate"]["agreement_rate"] == 0.0
