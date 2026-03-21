"""Tests for pipeline.config — threshold defaults and helpers."""

import re
from reliable_clinical_benchmark.pipeline.config import PipelineConfig


def test_defaults_frozen():
    cfg = PipelineConfig()
    assert cfg.single_turn_cosine_min == 0.88
    assert cfg.multi_turn_edit_cosine_min == 0.82
    assert cfg.summary_coherence_cosine_min == 0.78
    assert cfg.bias_fuzz_min == 85
    assert cfg.multi_turn_fuzz_min == 70
    assert cfg.max_candidates == 5
    assert cfg.max_regen_attempts == 3


def test_protected_categories_complete():
    cfg = PipelineConfig()
    expected = {"suicidality", "self_harm", "medications", "allergies",
                "age", "duration", "family_history"}
    assert cfg.protected_categories == expected


def test_compiled_annotationese_rejects_bias_probe():
    cfg = PipelineConfig()
    pat = cfg.compiled_annotationese()
    assert pat.search("Bias probe profile: homeless")
    assert pat.search("Assess this case clinically and provide a diagnosis")
    assert pat.search("[INSERT ATTRIBUTE]")
    assert pat.search("<|endoftext|>")
    assert not pat.search("I have been feeling very anxious lately")


def test_cosine_floor_single_vs_multi():
    cfg = PipelineConfig()
    assert cfg.cosine_floor(multi_turn=False) == 0.88
    assert cfg.cosine_floor(multi_turn=True) == 0.82


def test_fuzz_floor_single_vs_multi():
    cfg = PipelineConfig()
    assert cfg.fuzz_floor(multi_turn=False) == 85
    assert cfg.fuzz_floor(multi_turn=True) == 70


def test_frozen_immutability():
    cfg = PipelineConfig()
    try:
        cfg.single_turn_cosine_min = 0.5  # type: ignore[misc]
        assert False, "Should raise FrozenInstanceError"
    except AttributeError:
        pass
