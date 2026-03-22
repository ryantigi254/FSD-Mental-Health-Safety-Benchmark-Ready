"""Unit tests for frozen evaluation data splits (Studies A, B, C and adversarial bias).

These tests assert simple structural invariants over the committed JSONs, so that:
  - the splits remain present and well-formed, and
  - accidental edits to the frozen benchmark data are caught early.
"""

from pathlib import Path
import json

import pytest

from reliable_clinical_benchmark.data.release_paths import CLINICIAN_READINESS_RELEASE_ID


BASE_DIR = Path(__file__).resolve().parents[3]
_RELEASE_DATA = f"data/releases/{CLINICIAN_READINESS_RELEASE_ID}"


def _load_json(relative_path: str):
    path = BASE_DIR / relative_path
    assert path.exists(), f"Expected data file not found: {path}"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.unit
def test_study_a_split_invariants():
    """Study A: basic schema and size checks for frozen split."""
    data = _load_json("data/openr1_psy_splits/study_a_test.json")

    assert "samples" in data
    samples = data["samples"]
    assert isinstance(samples, list)
    # Split size (minimum)
    assert len(samples) >= 2000

    for sample in samples:
        for key in ("id", "prompt", "gold_answer", "gold_reasoning", "metadata"):
            assert key in sample
        assert isinstance(sample["id"], str) and sample["id"]
        assert isinstance(sample["prompt"], str) and sample["prompt"].strip()
        assert isinstance(sample["gold_answer"], str) and sample["gold_answer"].strip()
        assert isinstance(sample["gold_reasoning"], list)
        assert isinstance(sample["metadata"], dict)
        # Reasoning steps should be non-empty strings if present
        for step in sample["gold_reasoning"]:
            assert isinstance(step, str)
            assert step.strip()


@pytest.mark.unit
def test_study_b_split_invariants():
    """Study B: persona-grounded sycophancy items and ToF cases."""
    samples_payload = _load_json("data/openr1_psy_splits/study_b_test.json")
    multi_turn_cases = _load_json("data/openr1_psy_splits/study_b_multi_turn_test.json")

    # Study B single-turn items are committed as a flat list.
    assert isinstance(samples_payload, list)
    samples = samples_payload

    assert isinstance(multi_turn_cases, list)

    # Split size (minimum)
    assert len(samples) >= 2000

    for item in samples:
        for key in ("id", "prompt", "gold_answer", "incorrect_opinion", "metadata"):
            assert key in item
        assert isinstance(item["id"], str) and item["id"]
        assert isinstance(item["prompt"], str) and item["prompt"].strip()
        assert isinstance(item["gold_answer"], str) and item["gold_answer"].strip()
        assert isinstance(item["incorrect_opinion"], str) and item["incorrect_opinion"].strip()

        metadata = item["metadata"]
        assert isinstance(metadata, dict)
        assert "persona_id" in metadata
        assert isinstance(metadata["persona_id"], str) and metadata["persona_id"]
        assert "age" in metadata
        assert isinstance(metadata["age"], int)

    # Multi-turn "truth-under-pressure" cases (minimum)
    assert len(multi_turn_cases) >= 40

    for case in multi_turn_cases:
        for key in ("id", "gold_answer", "turns", "metadata"):
            assert key in case
        assert isinstance(case["id"], str) and case["id"]
        assert isinstance(case["gold_answer"], str) and case["gold_answer"].strip()

        turns = case["turns"]
        assert isinstance(turns, list)
        # Each ToF conversation should have at least 5 turns of escalating pressure
        assert len(turns) >= 5
        for t in turns:
            assert isinstance(t.get("turn"), int)
            assert isinstance(t.get("message"), str) and t["message"].strip()

        metadata = case["metadata"]
        assert isinstance(metadata, dict)
        assert "persona_id" in metadata
        assert isinstance(metadata["persona_id"], str) and metadata["persona_id"]
        assert "age" in metadata
        assert isinstance(metadata["age"], int)


@pytest.mark.unit
def test_study_c_split_invariants():
    """Study C: longitudinal drift cases are persona-grounded and 10-turn each."""
    data = _load_json("data/openr1_psy_splits/study_c_test.json")

    assert "cases" in data
    cases = data["cases"]
    assert isinstance(cases, list)

    # Split size (minimum)
    assert len(cases) >= 100

    for case in cases:
        for key in ("id", "patient_summary", "critical_entities", "turns", "metadata", "num_turns"):
            assert key in case

        assert isinstance(case["id"], str) and case["id"]
        assert isinstance(case["patient_summary"], str) and case["patient_summary"].strip()

        critical_entities = case["critical_entities"]
        assert isinstance(critical_entities, list)
        assert critical_entities  # At least one critical entity
        for ent in critical_entities:
            assert isinstance(ent, str) and ent.strip()

        turns = case["turns"]
        assert isinstance(turns, list)
        # Longitudinal histories are fixed at 20 turns
        assert len(turns) == 20
        assert case.get("num_turns") == 20
        for t in turns:
            assert isinstance(t.get("turn"), int)
            assert isinstance(t.get("message"), str) and t["message"].strip()

        metadata = case["metadata"]
        assert isinstance(metadata, dict)
        assert "persona_id" in metadata
        assert isinstance(metadata["persona_id"], str) and metadata["persona_id"]
        # New: provenance back to OpenR1-Psy train trajectories
        assert "source_openr1_ids" in metadata
        assert isinstance(metadata["source_openr1_ids"], list)


@pytest.mark.unit
def test_adversarial_bias_invariants():
    """Adversarial bias set: basic schema and coverage checks."""
    data = _load_json(f"{_RELEASE_DATA}/adversarial_bias/biased_vignettes.json")
    catalog = _load_json(f"{_RELEASE_DATA}/adversarial_bias/dimension_catalog_v3_2.json")

    assert "cases" in data
    cases = data["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 2000

    dimensions = set()
    allowed_dimensions = {d["dimension"] for d in catalog.get("dimensions", [])}
    assert len(allowed_dimensions) == 44

    ids = set()
    canonical_keys = set()
    persona_counts = {}

    for case in cases:
        for key in ("id", "prompt", "bias_feature", "bias_label", "metadata"):
            assert key in case

        assert isinstance(case["id"], str) and case["id"]
        assert case["id"] not in ids, f"Duplicate adversarial bias id found: {case['id']}"
        ids.add(case["id"])
        assert isinstance(case["prompt"], str) and case["prompt"].strip()
        assert isinstance(case["bias_feature"], str) and case["bias_feature"].strip()
        assert isinstance(case["bias_label"], str) and case["bias_label"].strip()
        canonical_key = (
            case["prompt"].strip().lower(),
            case["bias_feature"].strip().lower(),
            case["bias_label"].strip().lower(),
        )
        assert canonical_key not in canonical_keys, (
            "Duplicate canonical adversarial triple found "
            f"(prompt,bias_feature,bias_label): {case['id']}"
        )
        canonical_keys.add(canonical_key)

        metadata = case["metadata"]
        assert isinstance(metadata, dict)
        assert "dimension" in metadata
        assert isinstance(metadata["dimension"], str) and metadata["dimension"].strip()
        dimensions.add(metadata["dimension"])
        assert metadata["dimension"] in allowed_dimensions

        for key in (
            "persona_id",
            "source_openr1_split",
            "source_openr1_id",
            "openr1_revision",
            "dimension_family",
        ):
            assert key in metadata
            if key == "source_openr1_id":
                assert isinstance(metadata[key], int)
            else:
                assert isinstance(metadata[key], str) and str(metadata[key]).strip()

        persona_id = metadata["persona_id"]
        persona_counts[persona_id] = persona_counts.get(persona_id, 0) + 1

        assert isinstance(case.get("pair_group_id"), str) and case["pair_group_id"].strip()
        assert isinstance(case.get("template_signature"), str) and case["template_signature"].strip()
        assert case.get("structure_version") == "v3.2"
        assert isinstance(case.get("source_variant_count"), int)
        assert case["source_variant_count"] >= 1

    assert dimensions == allowed_dimensions
    assert len(persona_counts) == 40
    assert all(count == 50 for count in persona_counts.values())


@pytest.mark.unit
def test_mapping_labels_match_canonical():
    """Mapping gold_label fields must agree with gold_diagnosis_labels.json."""
    mapping_data = _load_json(f"{_RELEASE_DATA}/study_a_gold/gold_labels_mapping.json")
    labels_data = _load_json(f"{_RELEASE_DATA}/study_a_gold/gold_diagnosis_labels.json")
    labels = labels_data["labels"]

    mismatches = []
    for sid, entry in mapping_data["mapping"].items():
        canonical = labels.get(sid, "")
        mapping_label = entry.get("gold_label", "")
        if mapping_label != canonical:
            mismatches.append(f"{sid}: mapping='{mapping_label}' canonical='{canonical}'")

    assert not mismatches, (
        f"{len(mismatches)} mapping/label mismatches:\n" + "\n".join(mismatches[:10])
    )


@pytest.mark.unit
def test_metadata_corrections_applied():
    """Metadata entries with new_label must match the canonical labels file."""
    metadata = _load_json(f"{_RELEASE_DATA}/study_a_gold/gold_diagnosis_metadata.json")
    labels_data = _load_json(f"{_RELEASE_DATA}/study_a_gold/gold_diagnosis_labels.json")
    labels = labels_data["labels"]

    mismatches = []
    for sid, entry in metadata.items():
        new_label = entry.get("new_label")
        if new_label is not None:
            actual = labels.get(sid, "")
            if actual != new_label:
                mismatches.append(
                    f"{sid}: labels='{actual}' metadata.new_label='{new_label}'"
                )

    assert not mismatches, (
        f"{len(mismatches)} metadata/label mismatches:\n" + "\n".join(mismatches)
    )
