"""Tests for pipeline.validation — gate logic."""

import pytest
from unittest.mock import MagicMock, patch

from reliable_clinical_benchmark.pipeline.anchor import AnchorSet
from reliable_clinical_benchmark.pipeline.config import PipelineConfig
from reliable_clinical_benchmark.pipeline.edit_plan import EditPlan
from reliable_clinical_benchmark.pipeline.validation import (
    validate_candidate,
    ValidationVerdict,
    _style_check_patient_voice,
)


@pytest.fixture
def config():
    return PipelineConfig()


@pytest.fixture
def mock_ner():
    ner = MagicMock()
    ner.extract_clinical_entities.return_value = {"anxiety", "depression", "sertraline"}
    return ner


@pytest.fixture
def mock_nli():
    nli = MagicMock()
    nli.predict.return_value = "entailment"
    return nli


@pytest.fixture
def sample_anchor():
    return AnchorSet(
        entities={"anxiety", "depression", "sertraline"},
        negated_entities=set(),
        risk_markers=set(),
        medications={"sertraline"},
        allergies=set(),
        demographics={"age": "32"},
        tone_features=["first_person_dominant"],
        core_facts=[],
    )


@pytest.fixture
def sample_plan():
    return EditPlan(
        source_row_id="test",
        target_operator="bias",
        insertion_angle="identity",
        bias_feature="homeless",
    )


class TestAnnotationeseRejection:
    def test_rejects_bias_probe(self, config, mock_ner, mock_nli, sample_anchor, sample_plan):
        candidate = "I feel anxious.\nBias probe profile: homeless."
        with patch("reliable_clinical_benchmark.pipeline.validation._get_embedder") as mock_emb:
            mock_emb.return_value = None
            verdict = validate_candidate(
                "I feel anxious.", candidate, sample_anchor, sample_plan,
                config, mock_ner, mock_nli,
            )
        assert not verdict.annotationese_clean
        assert not verdict.passed

    def test_rejects_assess_clinically(self, config, mock_ner, mock_nli, sample_anchor, sample_plan):
        candidate = "I feel anxious.\nAssess this case clinically."
        with patch("reliable_clinical_benchmark.pipeline.validation._get_embedder") as mock_emb:
            mock_emb.return_value = None
            verdict = validate_candidate(
                "I feel anxious.", candidate, sample_anchor, sample_plan,
                config, mock_ner, mock_nli,
            )
        assert not verdict.annotationese_clean

    def test_rejects_bracket_placeholder(self, config, mock_ner, mock_nli, sample_anchor, sample_plan):
        candidate = "I feel anxious. [INSERT ATTRIBUTE] here."
        with patch("reliable_clinical_benchmark.pipeline.validation._get_embedder") as mock_emb:
            mock_emb.return_value = None
            verdict = validate_candidate(
                "I feel anxious.", candidate, sample_anchor, sample_plan,
                config, mock_ner, mock_nli,
            )
        assert not verdict.annotationese_clean

    def test_accepts_clean_text(self, config, mock_ner, mock_nli, sample_anchor, sample_plan):
        candidate = "I feel anxious and I've been struggling with my depression."
        with patch("reliable_clinical_benchmark.pipeline.validation._get_embedder") as mock_emb:
            mock_emb.return_value = None
            verdict = validate_candidate(
                "I feel anxious.", candidate, sample_anchor, sample_plan,
                config, mock_ner, mock_nli,
            )
        assert verdict.annotationese_clean


class TestNLIGate:
    def test_contradiction_fails(self, config, mock_ner, sample_anchor, sample_plan):
        nli = MagicMock()
        nli.predict.return_value = "contradiction"
        with patch("reliable_clinical_benchmark.pipeline.validation._get_embedder") as mock_emb:
            mock_emb.return_value = None
            verdict = validate_candidate(
                "I feel anxious.", "I feel great.", sample_anchor, sample_plan,
                config, mock_ner, nli,
            )
        assert "NLI contradiction" in verdict.failure_reasons[0]
        assert not verdict.passed


class TestProtectedAnchorAutoFail:
    def test_lost_medication_fails(self, config, mock_ner, mock_nli, sample_plan):
        anchor = AnchorSet(
            entities={"sertraline"},
            medications={"sertraline"},
        )
        # Candidate drops sertraline
        with patch("reliable_clinical_benchmark.pipeline.validation._get_embedder") as mock_emb:
            mock_emb.return_value = None
            verdict = validate_candidate(
                "I take sertraline daily.",
                "I feel better now.",
                anchor, sample_plan, config, mock_ner, mock_nli,
            )
        assert any("Protected entity lost" in r for r in verdict.failure_reasons)
        assert not verdict.passed

    def test_lost_age_fails(self, config, mock_ner, mock_nli, sample_plan):
        anchor = AnchorSet(
            entities=set(),
            demographics={"age": "45"},
        )
        with patch("reliable_clinical_benchmark.pipeline.validation._get_embedder") as mock_emb:
            mock_emb.return_value = None
            verdict = validate_candidate(
                "I'm a 45-year-old man.",
                "I'm a man who feels anxious.",
                anchor, sample_plan, config, mock_ner, mock_nli,
            )
        assert any("Age anchor lost" in r for r in verdict.failure_reasons)


class TestStyleCheck:
    def test_patient_voice_accepted(self):
        assert _style_check_patient_voice(
            "I've been feeling really down lately. My sleep is terrible."
        )

    def test_clinical_header_rejected(self):
        assert not _style_check_patient_voice(
            "Assessment: Patient presents with major depressive disorder."
        )

    def test_empty_text_rejected(self):
        assert not _style_check_patient_voice("")


class TestVerdictSerialization:
    def test_to_dict(self):
        v = ValidationVerdict(
            passed=True,
            nli_verdict="entailment",
            cosine_similarity=0.92,
            fuzz_score=88.5,
            entity_retention_rate=1.0,
        )
        d = v.to_dict()
        assert d["passed"] is True
        assert d["nli_verdict"] == "entailment"
        assert isinstance(d["cosine_similarity"], float)
