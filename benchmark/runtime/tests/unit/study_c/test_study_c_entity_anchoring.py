"""Tests that every Study C critical_entity is anchored in patient_summary.

Entities that are literal substrings pass automatically. Semantic abstractions
(e.g. 'functional impairment') are validated against the evidence map in
data/study_c_gold/entity_evidence_map.json.
"""

from pathlib import Path
import json

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]


def _load_json(relative_path: str):
    path = BASE_DIR / relative_path
    assert path.exists(), f"Expected data file not found: {path}"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.unit
def test_entity_evidence_map_exists():
    """The entity evidence map file must exist."""
    path = BASE_DIR / "data" / "study_c_gold" / "entity_evidence_map.json"
    assert path.exists(), f"Missing entity evidence map: {path}"


@pytest.mark.unit
def test_all_entities_anchored():
    """Every critical_entity must be anchored (literal, numeric, or synonym-mapped)."""
    data = _load_json("data/openr1_psy_splits/study_c_test.json")
    cases = data["cases"]

    evidence_data = _load_json("data/study_c_gold/entity_evidence_map.json")
    global_synonyms = evidence_data.get("global_synonyms", {})
    case_evidence = evidence_data.get("case_evidence", {})

    failures = []
    for case in cases:
        cid = case["id"]
        summary = case["patient_summary"].lower()
        for ent in case["critical_entities"]:
            anchored = False

            # Literal substring
            if ent.lower() in summary:
                anchored = True
            # Numeric age
            else:
                try:
                    age = int(ent)
                    if f"{age}-year-old" in case["patient_summary"]:
                        anchored = True
                except ValueError:
                    pass

            # Synonym map
            if not anchored and ent in global_synonyms:
                patterns = global_synonyms[ent].get("evidence_patterns", [])
                if any(p.lower() in summary for p in patterns):
                    anchored = True

            # Per-case evidence
            if not anchored and cid in case_evidence and ent in case_evidence[cid]:
                span = case_evidence[cid][ent]
                if (
                    isinstance(span, str)
                    and span.strip()
                    and "(unanchored)" not in span.lower()
                    and span.lower() in summary
                ):
                    anchored = True

            if not anchored:
                failures.append(f"{cid}: '{ent}'")

    assert not failures, (
        f"{len(failures)} unanchored entities:\n" + "\n".join(failures[:20])
    )


@pytest.mark.unit
def test_evidence_map_covers_non_literal_entities():
    """Non-literal entities must have entries in global_synonyms or case_evidence."""
    data = _load_json("data/openr1_psy_splits/study_c_test.json")
    cases = data["cases"]

    evidence_data = _load_json("data/study_c_gold/entity_evidence_map.json")
    global_synonyms = evidence_data.get("global_synonyms", {})

    non_literal_uncovered = []
    for case in cases:
        summary = case["patient_summary"].lower()
        for ent in case["critical_entities"]:
            if ent.lower() in summary:
                continue
            try:
                age = int(ent)
                if f"{age}-year-old" in case["patient_summary"]:
                    continue
            except ValueError:
                pass
            # This entity is non-literal; it must be in the synonym map
            if ent not in global_synonyms:
                non_literal_uncovered.append(f"{case['id']}: '{ent}'")

    assert not non_literal_uncovered, (
        f"{len(non_literal_uncovered)} non-literal entities missing from synonym map:\n"
        + "\n".join(non_literal_uncovered[:20])
    )
