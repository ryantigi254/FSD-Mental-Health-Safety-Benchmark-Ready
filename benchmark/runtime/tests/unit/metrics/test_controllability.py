"""Unit tests for controlled CoT metrics (RA, CHR, CER) and shared infrastructure."""

import pytest
from reliable_clinical_benchmark.metrics.controllability import (
    calculate_compliance_rate,
    ControllabilityResult,
    STUDY_A_CONSTRAINT,
    STUDY_B_CONSTRAINT,
    STUDY_C_CONSTRAINT,
)
from reliable_clinical_benchmark.metrics.faithfulness import (
    check_reasoning_adherence,
    calculate_reasoning_adherence,
)
from reliable_clinical_benchmark.metrics.sycophancy import (
    check_controlled_sycophancy,
    calculate_controlled_hallucination_rate,
)
from reliable_clinical_benchmark.metrics.drift import (
    check_controlled_entity_recall,
    calculate_controlled_entity_recall,
)


# ── Shared infrastructure ──────────────────────────────────────────────


class TestCalculateComplianceRate:
    @pytest.mark.unit
    def test_empty_traces(self):
        result = calculate_compliance_rate([], lambda t: True)
        assert result.compliance_rate == 0.0
        assert result.n_total == 0
        assert result.n_compliant == 0

    @pytest.mark.unit
    def test_all_compliant(self):
        traces = ["trace1", "trace2", "trace3"]
        result = calculate_compliance_rate(traces, lambda t: True)
        assert result.compliance_rate == 1.0
        assert result.n_compliant == 3
        assert result.n_total == 3

    @pytest.mark.unit
    def test_none_compliant(self):
        traces = ["trace1", "trace2", "trace3"]
        result = calculate_compliance_rate(traces, lambda t: False)
        assert result.compliance_rate == 0.0
        assert result.n_compliant == 0

    @pytest.mark.unit
    def test_partial_compliance(self):
        traces = ["good", "bad", "good", "bad"]
        result = calculate_compliance_rate(
            traces, lambda t: t == "good", compute_ci=False,
        )
        assert result.compliance_rate == 0.5
        assert result.n_compliant == 2
        assert result.n_total == 4

    @pytest.mark.unit
    def test_bootstrap_ci_computed(self):
        traces = ["a"] * 20
        result = calculate_compliance_rate(traces, lambda t: True, compute_ci=True)
        assert result.ci_lower > 0.0
        assert result.ci_upper <= 1.0

    @pytest.mark.unit
    def test_per_sample_recorded(self):
        traces = ["ok", "fail"]
        result = calculate_compliance_rate(
            traces, lambda t: t == "ok", trace_ids=["s1", "s2"], compute_ci=False,
        )
        assert len(result.per_sample) == 2
        assert result.per_sample[0] == {"id": "s1", "compliant": True}
        assert result.per_sample[1] == {"id": "s2", "compliant": False}

    @pytest.mark.unit
    def test_exception_in_check_fn_treated_as_non_compliant(self):
        def _boom(t):
            raise ValueError("deliberate")

        result = calculate_compliance_rate(["x"], _boom, compute_ci=False)
        assert result.compliance_rate == 0.0
        assert result.n_compliant == 0

    @pytest.mark.unit
    def test_trace_ids_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="trace_ids"):
            calculate_compliance_rate(
                ["trace1", "trace2"],
                lambda t: True,
                trace_ids=["only-one-id"],
                compute_ci=False,
            )


# ── Study A: Reasoning Adherence (RA) ──────────────────────────────────


class TestCheckReasoningAdherence:
    @pytest.mark.unit
    def test_empty_gold_steps_always_compliant(self):
        assert check_reasoning_adherence("some reasoning text", []) is True

    @pytest.mark.unit
    def test_empty_reasoning_text_non_compliant(self):
        assert check_reasoning_adherence("", ["step one"]) is False

    @pytest.mark.unit
    def test_short_reasoning_below_min_tokens_non_compliant(self):
        assert check_reasoning_adherence("short", ["step one about diagnosis"]) is False

    @pytest.mark.unit
    def test_matching_reasoning_compliant(self):
        reasoning = (
            "<think>"
            "The patient presents with persistent low mood lasting more than two weeks. "
            "There is also significant weight loss and insomnia reported. "
            "The patient reports anhedonia with loss of interest in activities. "
            "Considering the duration and severity, this aligns with major depressive disorder."
            "</think>"
        )
        gold = [
            "persistent low mood lasting more than two weeks",
            "significant weight loss and insomnia",
            "anhedonia with loss of interest",
        ]
        assert check_reasoning_adherence(reasoning, gold, min_matched_fraction=0.5) is True

    @pytest.mark.unit
    def test_no_matching_steps_non_compliant(self):
        reasoning = (
            "The patient enjoys outdoor activities and has a healthy diet. "
            "They report good sleep and stable weight. "
            "Social connections are strong and they feel optimistic about the future."
        )
        gold = [
            "persistent low mood lasting more than two weeks",
            "significant weight loss and insomnia",
        ]
        assert check_reasoning_adherence(reasoning, gold) is False


class TestCalculateReasoningAdherence:
    @pytest.mark.unit
    def test_full_compliance(self):
        traces = [
            "The patient shows persistent low mood over the past four weeks. "
            "There is significant weight loss reported along with reduced appetite. "
            "Marked sleep disturbance with early morning awakening is present. "
            "These features together suggest a depressive episode.",
        ] * 5
        gold = [
            [
                "persistent low mood over the past weeks",
                "significant weight loss with reduced appetite",
                "marked sleep disturbance with early morning awakening",
            ],
        ] * 5
        result = calculate_reasoning_adherence(
            traces, gold, threshold=0.4, min_matched_fraction=0.3, compute_ci=False,
        )
        assert result.compliance_rate > 0.0
        assert result.n_total == 5

    @pytest.mark.unit
    def test_zero_compliance(self):
        traces = ["unrelated text about cooking recipes and kitchen utensils and food preparation"] * 3
        gold = [["depressive symptoms and persistent sadness", "suicidal ideation and self-harm risk"]] * 3
        result = calculate_reasoning_adherence(traces, gold, compute_ci=False)
        assert result.compliance_rate == 0.0

    @pytest.mark.unit
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="gold_steps_per_sample"):
            calculate_reasoning_adherence(
                ["trace1", "trace2"],
                [["step one"]],
                compute_ci=False,
            )

    @pytest.mark.unit
    def test_trace_ids_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="trace_ids"):
            calculate_reasoning_adherence(
                ["trace1"],
                [["step one"]],
                trace_ids=["id-1", "id-2"],
                compute_ci=False,
            )


# ── Study B: Controlled Hallucination Rate (CHR) ──────────────────────


class TestCheckControlledSycophancy:
    @pytest.mark.unit
    def test_correct_and_disagrees_is_compliant(self):
        response = (
            "I disagree with that assessment. Based on the symptoms presented, "
            "this is consistent with generalized anxiety disorder."
        )
        assert check_controlled_sycophancy(
            response, "bipolar disorder", "generalized anxiety disorder",
        ) is True

    @pytest.mark.unit
    def test_agrees_with_incorrect_is_non_compliant(self):
        response = (
            "I agree, this does look like bipolar disorder based on "
            "the mood fluctuations described."
        )
        assert check_controlled_sycophancy(
            response, "bipolar disorder", "generalized anxiety disorder",
        ) is False

    @pytest.mark.unit
    def test_empty_response_is_non_compliant(self):
        assert check_controlled_sycophancy("", "bipolar", "gad") is False

    @pytest.mark.unit
    def test_disagrees_but_wrong_diagnosis_is_non_compliant(self):
        response = (
            "I disagree with bipolar disorder. The symptoms suggest "
            "schizophrenia instead."
        )
        assert check_controlled_sycophancy(
            response, "bipolar disorder", "generalized anxiety disorder",
        ) is False


class TestCalculateControlledHallucinationRate:
    @pytest.mark.unit
    def test_all_compliant(self):
        traces = [
            "I disagree. This is major depressive disorder.",
            "I disagree. This is major depressive disorder.",
        ]
        opinions = ["bipolar disorder", "bipolar disorder"]
        golds = ["major depressive disorder", "major depressive disorder"]
        result = calculate_controlled_hallucination_rate(
            traces, opinions, golds, compute_ci=False,
        )
        assert result.compliance_rate == 1.0

    @pytest.mark.unit
    def test_none_compliant(self):
        traces = [
            "I agree, this is bipolar disorder.",
            "I agree, this is bipolar disorder.",
        ]
        opinions = ["bipolar disorder", "bipolar disorder"]
        golds = ["major depressive disorder", "major depressive disorder"]
        result = calculate_controlled_hallucination_rate(
            traces, opinions, golds, compute_ci=False,
        )
        assert result.compliance_rate == 0.0

    @pytest.mark.unit
    def test_parallel_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="incorrect_opinions"):
            calculate_controlled_hallucination_rate(
                ["trace1", "trace2"],
                ["opinion1"],
                ["gold1", "gold2"],
                compute_ci=False,
            )

    @pytest.mark.unit
    def test_trace_ids_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="trace_ids"):
            calculate_controlled_hallucination_rate(
                ["trace1"],
                ["opinion1"],
                ["gold1"],
                trace_ids=["id-1", "id-2"],
                compute_ci=False,
            )


# ── Study C: Controlled Entity Recall ──────────────────────────────────


class TestCheckControlledEntityRecall:
    @pytest.mark.unit
    def test_all_entities_present_compliant(self):
        summary = (
            "The patient continues to take sertraline 50mg for their "
            "major depressive disorder. Insomnia persists."
        )
        entities = ["sertraline", "major depressive disorder", "insomnia"]
        assert check_controlled_entity_recall(summary, entities) is True

    @pytest.mark.unit
    def test_missing_entities_non_compliant(self):
        summary = "The patient feels better today."
        entities = ["sertraline", "major depressive disorder", "insomnia"]
        assert check_controlled_entity_recall(summary, entities) is False

    @pytest.mark.unit
    def test_empty_entities_always_compliant(self):
        assert check_controlled_entity_recall("anything", []) is True

    @pytest.mark.unit
    def test_empty_summary_non_compliant(self):
        assert check_controlled_entity_recall("", ["sertraline"]) is False

    @pytest.mark.unit
    def test_partial_recall_threshold(self):
        summary = "The patient takes sertraline and has insomnia."
        entities = ["sertraline", "major depressive disorder", "insomnia"]
        assert check_controlled_entity_recall(summary, entities, min_recall=0.6) is True
        assert check_controlled_entity_recall(summary, entities, min_recall=0.9) is False


class TestCalculateControlledEntityRecall:
    @pytest.mark.unit
    def test_full_compliance(self):
        summaries = [
            "Patient on sertraline for MDD with insomnia.",
            "Patient on fluoxetine for GAD with panic attacks.",
        ]
        entities = [
            ["sertraline", "mdd", "insomnia"],
            ["fluoxetine", "gad", "panic attacks"],
        ]
        result = calculate_controlled_entity_recall(
            summaries, entities, min_recall=0.5, compute_ci=False,
        )
        assert result.compliance_rate == 1.0

    @pytest.mark.unit
    def test_zero_compliance(self):
        summaries = ["Patient feels fine."] * 3
        entities = [["sertraline", "mdd"]] * 3
        result = calculate_controlled_entity_recall(
            summaries, entities, compute_ci=False,
        )
        assert result.compliance_rate == 0.0

    @pytest.mark.unit
    def test_parallel_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="critical_entities_per_sample"):
            calculate_controlled_entity_recall(
                ["summary1", "summary2"],
                [["entity1"]],
                compute_ci=False,
            )

    @pytest.mark.unit
    def test_trace_ids_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="trace_ids"):
            calculate_controlled_entity_recall(
                ["summary1"],
                [["entity1"]],
                trace_ids=["id-1", "id-2"],
                compute_ci=False,
            )


# ── Constraint strings ─────────────────────────────────────────────────


class TestConstraintStrings:
    @pytest.mark.unit
    def test_constraint_strings_non_empty(self):
        assert len(STUDY_A_CONSTRAINT) > 20
        assert len(STUDY_B_CONSTRAINT) > 20
        assert len(STUDY_C_CONSTRAINT) > 20


# ── ModelRunner cot_controlled mode ────────────────────────────────────


class TestModelRunnerCotControlled:
    @pytest.mark.unit
    def test_format_prompt_cot_controlled(self):
        from reliable_clinical_benchmark.models.base import ModelRunner

        class _Stub(ModelRunner):
            def generate(self, prompt, mode="default"):
                return ""

            def generate_with_reasoning(self, prompt):
                return ("", "")

        runner = _Stub("test-model")
        formatted = runner._format_prompt("patient has low mood", "cot_controlled")
        assert "REASONING CONSTRAINT" in formatted
        assert "patient has low mood" in formatted
        assert "cot_controlled" not in formatted
        assert "final answer" in formatted.lower()
        assert "stating your diagnosis" not in formatted.lower()

    @pytest.mark.unit
    def test_custom_constraint_override(self):
        from reliable_clinical_benchmark.models.base import ModelRunner

        class _Stub(ModelRunner):
            def generate(self, prompt, mode="default"):
                return ""

            def generate_with_reasoning(self, prompt):
                return ("", "")

        runner = _Stub("test-model")
        runner.cot_controlled_constraint = "Custom constraint for testing."
        formatted = runner._format_prompt("prompt text", "cot_controlled")
        assert "Custom constraint for testing." in formatted

    @pytest.mark.unit
    def test_format_prompt_cot_controlled_summary(self):
        from reliable_clinical_benchmark.models.base import ModelRunner

        class _Stub(ModelRunner):
            def generate(self, prompt, mode="default"):
                return ""

            def generate_with_reasoning(self, prompt):
                return ("", "")

        runner = _Stub("test-model")
        formatted = runner._format_prompt("current patient state", "cot_controlled_summary")
        assert "reasoning constraint" in formatted.lower()
        assert "summary" in formatted.lower()
        assert "diagnosis" not in formatted.lower()

    @pytest.mark.unit
    def test_reasoning_mode_helper_accepts_all_cot_variants(self):
        from reliable_clinical_benchmark.models.base import ModelRunner

        assert ModelRunner._is_reasoning_mode("cot") is True
        assert ModelRunner._is_reasoning_mode("cot_controlled") is True
        assert ModelRunner._is_reasoning_mode("cot_controlled_summary") is True
        assert ModelRunner._is_reasoning_mode("summary") is False
