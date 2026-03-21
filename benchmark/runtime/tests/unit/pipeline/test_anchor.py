"""Tests for pipeline.anchor — extraction of immutable clinical anchors."""

import pytest
from unittest.mock import MagicMock

from reliable_clinical_benchmark.pipeline.anchor import (
    AnchorSet,
    extract_anchors,
    _is_negated,
)
from reliable_clinical_benchmark.pipeline.config import PipelineConfig


@pytest.fixture
def mock_ner():
    ner = MagicMock()
    ner.extract_clinical_entities.return_value = {
        "anxiety", "depression", "sertraline", "insomnia",
    }
    return ner


@pytest.fixture
def config():
    return PipelineConfig()


class TestIsNegated:
    def test_negation_detected(self):
        assert _is_negated("depression", "no depression or anxiety")

    def test_negation_with_denies(self):
        assert _is_negated("suicidal ideation", "patient denies suicidal ideation")

    def test_no_negation(self):
        assert not _is_negated("anxiety", "I have anxiety and depression")

    def test_empty_inputs(self):
        assert not _is_negated("", "some text")
        assert not _is_negated("entity", "")


class TestExtractAnchors:
    def test_basic_extraction(self, mock_ner, config):
        text = "I'm a 32-year-old woman and I've been on sertraline 50 mg daily for 3 months."
        anchor = extract_anchors(text, mock_ner, config)

        assert isinstance(anchor, AnchorSet)
        assert "sertraline" in anchor.entities
        assert "32" in anchor.demographics.get("age", "")
        assert anchor.medications  # should detect sertraline via regex

    def test_risk_markers_detected(self, mock_ner, config):
        text = "I've been having thoughts of self-harm and I'm in crisis."
        anchor = extract_anchors(text, mock_ner, config)
        assert anchor.risk_markers
        assert any("self" in m or "crisis" in m for m in anchor.risk_markers)

    def test_allergy_detection(self, mock_ner, config):
        text = "I have a penicillin allergy and I'm allergic to latex."
        anchor = extract_anchors(text, mock_ner, config)
        assert anchor.allergies

    def test_family_mentions(self, mock_ner, config):
        text = "My mother has depression and my father was an alcoholic."
        anchor = extract_anchors(text, mock_ner, config)
        assert "family_mentions" in anchor.demographics
        assert "mother" in anchor.demographics["family_mentions"]
        assert "father" in anchor.demographics["family_mentions"]

    def test_tone_first_person(self, mock_ner, config):
        text = "I feel terrible. I can't sleep. I've lost my appetite. I'm scared."
        anchor = extract_anchors(text, mock_ner, config)
        assert "first_person_dominant" in anchor.tone_features

    def test_empty_text(self, mock_ner, config):
        anchor = extract_anchors("", mock_ner, config)
        assert anchor.entities == set()
        assert anchor.risk_markers == set()

    def test_protected_entities_union(self, mock_ner, config):
        text = "I take sertraline and I have a penicillin allergy. I feel suicidal."
        anchor = extract_anchors(text, mock_ner, config)
        protected = anchor.protected_entities
        assert isinstance(protected, set)
        # Should include risk + meds + allergies
        assert anchor.risk_markers <= protected
        assert anchor.medications <= protected
        assert anchor.allergies <= protected

    def test_negated_entities(self, mock_ner, config):
        mock_ner.extract_clinical_entities.return_value = {"depression", "anxiety"}
        text = "I have anxiety but no depression."
        anchor = extract_anchors(text, mock_ner, config)
        assert "depression" in anchor.negated_entities
        assert "anxiety" not in anchor.negated_entities
