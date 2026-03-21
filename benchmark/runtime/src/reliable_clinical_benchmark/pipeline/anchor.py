"""Anchor extraction — identifies immutable facts, entities, and tone from source text.

Reuses MedicalNER from utils/ner.py and negation patterns from metrics/drift.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.ner import MedicalNER
    from .config import PipelineConfig

# ---------------------------------------------------------------------------
# Regex patterns (aligned with drift.py where possible)
# ---------------------------------------------------------------------------

_NEGATION_TERMS = {"no", "not", "denies", "without", "never", "none", "denied"}
_NEGATION_WINDOW_TOKENS = 5

_AGE_PATTERN = re.compile(
    r"\b(\d{1,3})\s*[-–]?\s*year[\s-]*old\b", re.IGNORECASE
)
_DURATION_PATTERN = re.compile(
    r"\b(\d+)\s+(?:month|week|year|day)s?\b", re.IGNORECASE
)
_FAMILY_PATTERN = re.compile(
    r"\b(mother|father|parent|sibling|brother|sister|son|daughter|"
    r"spouse|partner|husband|wife|grandparent|uncle|aunt|family)\b",
    re.IGNORECASE,
)
_RISK_PATTERN = re.compile(
    r"\b(self[- ]?harm|suicid\w*|plan|intent|means|safe right now|"
    r"risk|crisis|overdose|cutting|ideation)\b",
    re.IGNORECASE,
)
_MED_PATTERN = re.compile(
    r"\b(sertraline|fluoxetine|citalopram|escitalopram|paroxetine|"
    r"venlafaxine|duloxetine|bupropion|mirtazapine|trazodone|"
    r"amitriptyline|nortriptyline|clomipramine|lithium|"
    r"quetiapine|olanzapine|risperidone|aripiprazole|"
    r"clonazepam|lorazepam|diazepam|alprazolam|buspirone|"
    r"methylphenidate|amphetamine|atomoxetine|"
    r"lamotrigine|valproate|carbamazepine|gabapentin|pregabalin|"
    r"mg\s+daily|mg\s+twice|dosage|dose|medication|prescribed)\b",
    re.IGNORECASE,
)
_ALLERGY_PATTERN = re.compile(
    r"\b(allerg(?:y|ic|ies)|penicillin|sulfa|latex|"
    r"anaphyla\w*|reaction to|intolerant)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_PATTERN = re.compile(r"\b(I|me|my|myself|mine|I'm|I've|I'd|I'll)\b")


# ---------------------------------------------------------------------------
# Tokenisation helpers (mirroring drift.py)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _is_negated(entity: str, text: str) -> bool:
    """Check whether *entity* is preceded by a negation term in *text*."""
    if not entity or not text:
        return False
    entity_tokens = _tokenise(entity)
    text_tokens = _tokenise(text)
    if not entity_tokens or not text_tokens:
        return False
    for idx in range(len(text_tokens) - len(entity_tokens) + 1):
        if text_tokens[idx: idx + len(entity_tokens)] == entity_tokens:
            window_start = max(0, idx - _NEGATION_WINDOW_TOKENS)
            window_tokens = text_tokens[window_start:idx]
            if any(tok in _NEGATION_TERMS for tok in window_tokens):
                return True
    return False


# ---------------------------------------------------------------------------
# AnchorSet
# ---------------------------------------------------------------------------

@dataclass
class AnchorSet:
    """Structured representation of immutable and identity-bearing features
    extracted from a source patient utterance."""

    entities: Set[str] = field(default_factory=set)
    negated_entities: Set[str] = field(default_factory=set)
    risk_markers: Set[str] = field(default_factory=set)
    medications: Set[str] = field(default_factory=set)
    allergies: Set[str] = field(default_factory=set)
    demographics: Dict[str, str] = field(default_factory=dict)
    tone_features: List[str] = field(default_factory=list)
    core_facts: List[str] = field(default_factory=list)

    # Convenience -------------------------------------------------------

    @property
    def protected_entities(self) -> Set[str]:
        """Union of all categories that are unconditionally protected."""
        return self.risk_markers | self.medications | self.allergies


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_anchors(
    text: str,
    ner: "MedicalNER",
    config: "PipelineConfig",
) -> AnchorSet:
    """Extract a structured anchor set from *text*.

    Parameters
    ----------
    text : str
        Source patient utterance or clinical vignette.
    ner : MedicalNER
        Pre-loaded scispaCy NER model.
    config : PipelineConfig
        Pipeline configuration (used for protected-category definitions).
    """
    if not text:
        return AnchorSet()

    # 1. Clinical entities via scispaCy
    entities = ner.extract_clinical_entities(text)

    # 2. Negated entities
    negated = {ent for ent in entities if _is_negated(ent, text)}

    # 3. Risk markers
    risk_markers = {m.group(0).lower() for m in _RISK_PATTERN.finditer(text)}

    # 4. Medications
    medications = {m.group(0).lower() for m in _MED_PATTERN.finditer(text)}

    # 5. Allergies
    allergies = {m.group(0).lower() for m in _ALLERGY_PATTERN.finditer(text)}

    # 6. Demographics
    demographics: Dict[str, str] = {}
    age_match = _AGE_PATTERN.search(text)
    if age_match:
        demographics["age"] = age_match.group(1)
    duration_matches = _DURATION_PATTERN.findall(text)
    if duration_matches:
        demographics["duration"] = ", ".join(duration_matches)
    family_matches = _FAMILY_PATTERN.findall(text)
    if family_matches:
        demographics["family_mentions"] = ", ".join(
            sorted({f.lower() for f in family_matches})
        )

    # 7. Tone features
    tone_features: List[str] = []
    first_person_count = len(_FIRST_PERSON_PATTERN.findall(text))
    word_count = len(text.split())
    if word_count > 0 and first_person_count / word_count > 0.03:
        tone_features.append("first_person_dominant")
    sentences = re.split(r"[.!?]+", text)
    avg_sentence_len = (
        sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)
    )
    if avg_sentence_len < 15:
        tone_features.append("short_sentences")
    elif avg_sentence_len > 30:
        tone_features.append("long_sentences")

    # 8. Core facts — sentence-level spans (simple extraction, no NLI self-check)
    core_facts = [s.strip() for s in sentences if len(s.strip().split()) >= 5]

    return AnchorSet(
        entities=entities,
        negated_entities=negated,
        risk_markers=risk_markers,
        medications=medications,
        allergies=allergies,
        demographics=demographics,
        tone_features=tone_features,
        core_facts=core_facts,
    )
