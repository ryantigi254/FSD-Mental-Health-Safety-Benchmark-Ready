"""Tests for pipeline.edit_plan — structured edit planning."""

import pytest
from unittest.mock import MagicMock

from reliable_clinical_benchmark.pipeline.anchor import AnchorSet
from reliable_clinical_benchmark.pipeline.config import PipelineConfig
from reliable_clinical_benchmark.pipeline.edit_plan import plan_edit, EditPlan, OPERATORS


@pytest.fixture
def config():
    return PipelineConfig()


@pytest.fixture
def sample_anchor():
    return AnchorSet(
        entities={"anxiety", "depression", "sertraline"},
        negated_entities=set(),
        risk_markers={"suicidal"},
        medications={"sertraline"},
        allergies={"penicillin"},
        demographics={"age": "32"},
        tone_features=["first_person_dominant"],
        core_facts=["I have been feeling very anxious"],
    )


class TestPlanEdit:
    def test_bias_operator(self, config, sample_anchor):
        plan = plan_edit(
            "I feel anxious. I take sertraline. I have suicidal thoughts.",
            sample_anchor, "bias", config,
            source_row_id="test_001",
            insertion_angle="identity",
            bias_feature="homeless",
            bias_label="substance misuse",
        )
        assert isinstance(plan, EditPlan)
        assert plan.target_operator == "bias"
        assert plan.bias_feature == "homeless"
        assert plan.insertion_angle == "identity"
        # Protected spans should be immutable
        assert len(plan.immutable_spans) > 0

    def test_pressure_operator(self, config, sample_anchor):
        plan = plan_edit(
            "I feel anxious. I take sertraline.",
            sample_anchor, "pressure", config,
            pressure_type="mild_doubt",
            escalation_level=1,
        )
        assert plan.target_operator == "pressure"
        assert plan.pressure_type == "mild_doubt"

    def test_invalid_operator_raises(self, config, sample_anchor):
        with pytest.raises(ValueError, match="Unknown operator"):
            plan_edit("text", sample_anchor, "invalid_op", config)

    def test_immutable_spans_contain_protected(self, config, sample_anchor):
        text = "I take sertraline daily. I feel anxious. I have suicidal thoughts."
        plan = plan_edit(text, sample_anchor, "bias", config)
        immutable_text = " ".join(plan.immutable_spans).lower()
        assert "sertraline" in immutable_text or "suicidal" in immutable_text

    def test_to_dict_roundtrip(self, config, sample_anchor):
        plan = plan_edit(
            "I feel anxious.", sample_anchor, "bias", config,
            source_row_id="test_001",
            bias_feature="female",
        )
        d = plan.to_dict()
        assert d["source_row_id"] == "test_001"
        assert d["target_operator"] == "bias"
        assert d["bias_feature"] == "female"
        assert isinstance(d["mutable_spans"], list)
        assert isinstance(d["immutable_spans"], list)

    def test_all_operators_accepted(self, config, sample_anchor):
        for op in OPERATORS:
            plan = plan_edit("text", sample_anchor, op, config)
            assert plan.target_operator == op
