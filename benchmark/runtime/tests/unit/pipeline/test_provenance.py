"""Tests for pipeline.provenance — record construction and serialisation."""

from reliable_clinical_benchmark.pipeline.provenance import (
    ProvenanceRecord,
    ProvenanceType,
    build_provenance,
    serialise_provenance,
)
from reliable_clinical_benchmark.pipeline.edit_plan import EditPlan
from reliable_clinical_benchmark.pipeline.validation import ValidationVerdict


class TestProvenanceType:
    def test_values(self):
        assert ProvenanceType.DIRECT_SOURCE.value == "direct_source"
        assert ProvenanceType.RETRIEVED_COMPOSED.value == "retrieved_composed"
        assert ProvenanceType.SOURCE_ANCHORED_GENERATED.value == "source_anchored_generated"


class TestBuildProvenance:
    def test_direct_source(self):
        prov = build_provenance(
            source_ids=[123],
            provenance_type=ProvenanceType.DIRECT_SOURCE,
        )
        assert prov.provenance_type == "direct_source"
        assert prov.source_openr1_ids == [123]
        assert prov.edit_operator is None
        assert prov.edit_plan is None

    def test_with_plan_and_verdict(self):
        plan = EditPlan(
            source_row_id="test_001",
            target_operator="bias",
            bias_feature="homeless",
        )
        verdict = ValidationVerdict(passed=True, nli_verdict="entailment")
        prov = build_provenance(
            source_ids=[456, 789],
            provenance_type=ProvenanceType.SOURCE_ANCHORED_GENERATED,
            plan=plan,
            verdict=verdict,
        )
        assert prov.provenance_type == "source_anchored_generated"
        assert prov.edit_operator == "bias"
        assert prov.edit_plan is not None
        assert prov.validation_verdict is not None

    def test_multi_turn_extensions(self):
        prov = build_provenance(
            source_ids=[100],
            provenance_type=ProvenanceType.RETRIEVED_COMPOSED,
            source_round_index=5,
            continuation_from_turn=4,
        )
        assert prov.source_round_index == 5
        assert prov.continuation_from_turn == 4


class TestSerialiseProvenance:
    def test_minimal_serialisation(self):
        prov = ProvenanceRecord(
            source_openr1_ids=[42],
            provenance_type="direct_source",
        )
        d = serialise_provenance(prov)
        assert d["source_openr1_ids"] == [42]
        assert d["provenance_type"] == "direct_source"
        # Optional fields should be absent
        assert "edit_operator" not in d
        assert "edit_plan" not in d
        assert "source_round_index" not in d

    def test_full_serialisation(self):
        plan = EditPlan(
            source_row_id="x",
            target_operator="pressure",
            pressure_type="mild_doubt",
        )
        verdict = ValidationVerdict(passed=True, cosine_similarity=0.9)
        prov = build_provenance(
            source_ids=[1, 2],
            provenance_type=ProvenanceType.RETRIEVED_COMPOSED,
            plan=plan,
            verdict=verdict,
            source_round_index=3,
            continuation_from_turn=2,
        )
        d = serialise_provenance(prov)
        assert d["provenance_type"] == "retrieved_composed"
        assert d["edit_operator"] == "pressure"
        assert "edit_plan" in d
        assert "validation" in d
        assert d["source_round_index"] == 3
        assert d["continuation_from_turn"] == 2

    def test_roundtrip_json_safe(self):
        import json
        prov = build_provenance(
            source_ids=[999],
            provenance_type=ProvenanceType.DIRECT_SOURCE,
        )
        d = serialise_provenance(prov)
        # Must be JSON-serialisable
        json_str = json.dumps(d)
        assert json_str
        loaded = json.loads(json_str)
        assert loaded["source_openr1_ids"] == [999]
