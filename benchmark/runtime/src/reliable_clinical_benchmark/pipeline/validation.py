"""Validation gates — all-must-pass acceptance for edit candidates.

Reuses NLIModel, MedicalNER, MiniLM embeddings, RapidFuzz, and
negation detection from the existing benchmark utilities.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..utils.ner import MedicalNER
    from ..utils.nli import NLIModel
    from .anchor import AnchorSet
    from .config import PipelineConfig
    from .edit_plan import EditPlan

logger = logging.getLogger(__name__)

# Singleton MiniLM embedder (shared with retrieval.py)
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            return None
    return _EMBEDDER


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Negation helpers (mirrored from anchor.py / drift.py)
# ---------------------------------------------------------------------------

_NEGATION_TERMS = {"no", "not", "denies", "without", "never", "none", "denied"}
_NEGATION_WINDOW_TOKENS = 5


def _tokenise(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _is_negated(entity: str, text: str) -> bool:
    if not entity or not text:
        return False
    entity_tokens = _tokenise(entity)
    text_tokens = _tokenise(text)
    if not entity_tokens or not text_tokens:
        return False
    for idx in range(len(text_tokens) - len(entity_tokens) + 1):
        if text_tokens[idx: idx + len(entity_tokens)] == entity_tokens:
            window_start = max(0, idx - _NEGATION_WINDOW_TOKENS)
            if any(tok in _NEGATION_TERMS for tok in text_tokens[window_start:idx]):
                return True
    return False


# ---------------------------------------------------------------------------
# Patient-voice heuristic
# ---------------------------------------------------------------------------

_FIRST_PERSON_RE = re.compile(r"\b(I|me|my|myself|mine|I'm|I've|I'd|I'll)\b")
_CLINICAL_HEADER_RE = re.compile(
    r"^\s*(diagnosis|assessment|plan|treatment|impression|hpi|ros|"
    r"chief complaint|history of present illness|subjective|objective)\s*[:.]",
    re.IGNORECASE | re.MULTILINE,
)


def _style_check_patient_voice(text: str) -> bool:
    """Return True if the text reads like patient-authored speech."""
    if not text:
        return False
    # Reject clinical headers
    if _CLINICAL_HEADER_RE.search(text):
        return False
    # Expect some first-person usage
    words = text.split()
    if len(words) < 5:
        return True  # too short to judge
    first_person_ratio = len(_FIRST_PERSON_RE.findall(text)) / len(words)
    return first_person_ratio > 0.01


# ---------------------------------------------------------------------------
# ValidationVerdict
# ---------------------------------------------------------------------------

@dataclass
class ValidationVerdict:
    """Result of running all gates on a candidate."""

    passed: bool = False
    nli_verdict: str = ""
    cosine_similarity: float = 0.0
    fuzz_score: float = 0.0
    entity_retention_rate: float = 0.0
    negation_retention: bool = True
    annotationese_clean: bool = True
    style_check: bool = True
    turn_coherence: Optional[bool] = None
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "passed": self.passed,
            "nli_verdict": self.nli_verdict,
            "cosine_similarity": round(self.cosine_similarity, 4),
            "fuzz_score": round(self.fuzz_score, 2),
            "entity_retention_rate": round(self.entity_retention_rate, 4),
            "negation_retention": self.negation_retention,
            "annotationese_clean": self.annotationese_clean,
            "style_check": self.style_check,
        }
        if self.turn_coherence is not None:
            out["turn_coherence"] = self.turn_coherence
        if self.failure_reasons:
            out["failure_reasons"] = self.failure_reasons
        return out


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_candidate(
    source_text: str,
    candidate_text: str,
    anchor: "AnchorSet",
    plan: "EditPlan",
    config: "PipelineConfig",
    ner: "MedicalNER",
    nli_model: "NLIModel",
    *,
    prev_turn_text: Optional[str] = None,
    multi_turn: bool = False,
) -> ValidationVerdict:
    """Run all validation gates on *candidate_text* against *source_text*.

    Every gate must pass for the verdict to be ``passed=True``.
    """
    failures: List[str] = []

    # --- 1. NLI contradiction check ---
    nli_verdict = nli_model.predict(source_text, candidate_text)
    if nli_verdict == "contradiction":
        failures.append("NLI contradiction against source")

    # --- 2. Entity retention ---
    source_entities = anchor.entities
    candidate_entities = ner.extract_clinical_entities(candidate_text)
    if source_entities:
        retained = sum(
            1 for ent in source_entities
            if ent in candidate_entities or ent.lower() in candidate_text.lower()
        )
        retention_rate = retained / len(source_entities)
    else:
        retention_rate = 1.0

    # Protected entities must ALL be present
    for pent in anchor.protected_entities:
        if pent.lower() not in candidate_text.lower():
            failures.append(f"Protected entity lost: {pent}")

    # --- 3. Cosine similarity ---
    embedder = _get_embedder()
    cosine_sim = 0.0
    if embedder is not None:
        src_emb = embedder.encode([source_text])[0]
        cand_emb = embedder.encode([candidate_text])[0]
        cosine_sim = _cosine_similarity(src_emb, cand_emb)

    cos_floor = config.cosine_floor(multi_turn=multi_turn)
    if cosine_sim < cos_floor:
        failures.append(
            f"Cosine similarity {cosine_sim:.4f} below floor {cos_floor}"
        )

    # --- 4. RapidFuzz lexical bound ---
    fuzz_score = 0.0
    try:
        from rapidfuzz import fuzz as rfuzz
        fuzz_score = rfuzz.token_set_ratio(source_text, candidate_text)
    except ImportError:
        # Fallback: simple token overlap ratio
        src_tokens = set(source_text.lower().split())
        cand_tokens = set(candidate_text.lower().split())
        if src_tokens:
            fuzz_score = len(src_tokens & cand_tokens) / len(src_tokens) * 100
    fuzz_floor = config.fuzz_floor(multi_turn=multi_turn)
    if fuzz_score < fuzz_floor:
        failures.append(
            f"Fuzz score {fuzz_score:.1f} below floor {fuzz_floor}"
        )

    # --- 5. Negation retention ---
    negation_ok = True
    for neg_ent in anchor.negated_entities:
        if not _is_negated(neg_ent, candidate_text):
            # Entity was negated in source but not in candidate — polarity flip
            if neg_ent.lower() in candidate_text.lower():
                negation_ok = False
                failures.append(f"Negation lost for: {neg_ent}")

    # --- 6. Annotationese rejection ---
    anno_re = config.compiled_annotationese()
    anno_clean = not bool(anno_re.search(candidate_text))
    if not anno_clean:
        failures.append("Annotationese detected in candidate")

    # --- 7. Style / patient-voice check ---
    style_ok = _style_check_patient_voice(candidate_text)
    if not style_ok:
        failures.append("Failed patient-voice style check")

    # --- 8. Turn-transition coherence (multi-turn only) ---
    turn_coherence: Optional[bool] = None
    if prev_turn_text and embedder is not None:
        prev_emb = embedder.encode([prev_turn_text])[0]
        cand_emb_t = embedder.encode([candidate_text])[0]
        turn_cos = _cosine_similarity(prev_emb, cand_emb_t)
        turn_coherence = turn_cos >= config.summary_coherence_cosine_min
        if not turn_coherence:
            failures.append(
                f"Turn coherence {turn_cos:.4f} below floor "
                f"{config.summary_coherence_cosine_min}"
            )

    # --- 9. Protected anchor auto-fail (demographics) ---
    for key, val in anchor.demographics.items():
        if key == "age" and val not in candidate_text:
            failures.append(f"Age anchor lost: {val}")
        elif key == "family_mentions":
            for fam in val.split(", "):
                if fam and fam.lower() not in candidate_text.lower():
                    failures.append(f"Family anchor lost: {fam}")

    passed = len(failures) == 0

    return ValidationVerdict(
        passed=passed,
        nli_verdict=nli_verdict,
        cosine_similarity=cosine_sim,
        fuzz_score=fuzz_score,
        entity_retention_rate=retention_rate,
        negation_retention=negation_ok,
        annotationese_clean=anno_clean,
        style_check=style_ok,
        turn_coherence=turn_coherence,
        failure_reasons=failures,
    )
