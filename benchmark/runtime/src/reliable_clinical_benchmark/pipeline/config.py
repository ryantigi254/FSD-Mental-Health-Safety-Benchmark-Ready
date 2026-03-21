"""Single source of truth for all pipeline gate thresholds and parameters."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the hardening pipeline.

    All gate thresholds, regeneration limits, protected categories, and
    annotationese rejection patterns live here so every pipeline stage
    shares a single set of defaults.
    """

    # --- Cosine similarity floors (MiniLM-L6-v2) ---
    single_turn_cosine_min: float = 0.88
    multi_turn_edit_cosine_min: float = 0.82
    summary_coherence_cosine_min: float = 0.78

    # --- RapidFuzz token_set_ratio floors ---
    bias_fuzz_min: int = 85
    multi_turn_fuzz_min: int = 70

    # --- Regeneration limits ---
    max_candidates: int = 5
    max_regen_attempts: int = 3

    # --- Protected anchor categories ---
    # Any loss or polarity flip of these is an automatic validation fail.
    protected_categories: FrozenSet[str] = frozenset({
        "suicidality",
        "self_harm",
        "medications",
        "allergies",
        "age",
        "duration",
        "family_history",
    })

    # --- Annotationese rejection patterns ---
    # If any pattern matches the candidate text the gate fails immediately.
    annotationese_patterns: Tuple[str, ...] = (
        r"Bias probe profile",
        r"Assess this case clinically",
        r"\[(?:INSERT|PLACEHOLDER|BIAS|FEATURE|ATTRIBUTE)[^\]]*\]",
        r"<\|[^|]*\|>",
    )

    # --- Convenience helpers (derived, not stored) ---

    def compiled_annotationese(self) -> re.Pattern:
        """Return a single compiled regex that matches any annotationese pattern."""
        return re.compile("|".join(self.annotationese_patterns), re.IGNORECASE)

    def cosine_floor(self, *, multi_turn: bool = False) -> float:
        """Return the appropriate cosine floor for single- or multi-turn edits."""
        return self.multi_turn_edit_cosine_min if multi_turn else self.single_turn_cosine_min

    def fuzz_floor(self, *, multi_turn: bool = False) -> int:
        """Return the appropriate fuzz floor for single- or multi-turn edits."""
        return self.multi_turn_fuzz_min if multi_turn else self.bias_fuzz_min
