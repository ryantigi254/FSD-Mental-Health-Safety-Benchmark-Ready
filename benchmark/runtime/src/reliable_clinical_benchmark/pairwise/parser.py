"""
Parse judge output into a canonical pairwise record.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

_VERDICT_PATTERNS = [
    (re.compile(r"\[\[\s*(?:Response\s+)?A\s*\]\]", re.IGNORECASE), "A"),
    (re.compile(r"\[\[\s*(?:Response\s+)?B\s*\]\]", re.IGNORECASE), "B"),
    (re.compile(r"\[\[\s*TIE\s*\]\]", re.IGNORECASE), "TIE"),
    (re.compile(r"\[\s*(?:Response\s+)?A\s*\]", re.IGNORECASE), "A"),
    (re.compile(r"\[\s*(?:Response\s+)?B\s*\]", re.IGNORECASE), "B"),
    (re.compile(r"\[\s*TIE\s*\]", re.IGNORECASE), "TIE"),
    (re.compile(r"(?:winner|verdict|choice)\s*:\s*(?:Response\s+)?A\b", re.IGNORECASE), "A"),
    (re.compile(r"(?:winner|verdict|choice)\s*:\s*(?:Response\s+)?B\b", re.IGNORECASE), "B"),
    (re.compile(r"(?:winner|verdict|choice)\s*:\s*TIE\b", re.IGNORECASE), "TIE"),
]

_EXCERPT_MAX = 600


@dataclass(frozen=True)
class ParsedJudgement:
    """Structured pairwise record."""

    case_id: str
    layer: str
    slice_id: str
    criterion_id: str
    judge_id: str
    order: str
    system_a: str
    system_b: str
    response_length_a: int
    response_length_b: int
    winner: str
    is_tie: bool
    is_invalid: bool
    verdict: str
    response_a_id: str
    response_b_id: str
    canonical_pair_key: str
    reasoning_excerpt: str
    raw_response: str


class PairwiseParser:
    """Extract structured verdicts from raw judge responses."""

    @staticmethod
    def extract_verdict(text: str) -> str:
        for pattern, label in _VERDICT_PATTERNS:
            if pattern.search(text):
                return label
        return "INVALID"

    @staticmethod
    def extract_reasoning(text: str) -> str:
        for pattern, _label in _VERDICT_PATTERNS:
            match = pattern.search(text)
            if match:
                excerpt = text[: match.start()].strip()
                return excerpt[-_EXCERPT_MAX:]
        return text[-_EXCERPT_MAX:].strip() if text else ""

    @staticmethod
    def _canonical_winner(
        *,
        verdict: str,
        order: str,
        system_a: str,
        system_b: str,
    ) -> str:
        if verdict == "TIE":
            return "TIE"
        if verdict == "INVALID":
            return "INVALID"

        if order == "AB":
            return system_a if verdict == "A" else system_b
        return system_b if verdict == "A" else system_a

    def parse(
        self,
        *,
        case_id: str,
        layer: str,
        slice_id: str,
        criterion_id: str,
        judge_id: str,
        order: str,
        system_a: str,
        system_b: str,
        response_length_a: int,
        response_length_b: int,
        raw_response: str,
    ) -> ParsedJudgement:
        verdict = self.extract_verdict(raw_response)
        reasoning = self.extract_reasoning(raw_response)
        winner = self._canonical_winner(
            verdict=verdict,
            order=order,
            system_a=system_a,
            system_b=system_b,
        )
        is_invalid = verdict == "INVALID"
        is_tie = verdict == "TIE"

        if is_invalid:
            logger.warning(
                "Invalid pairwise parse case=%s slice=%s criterion=%s judge=%s order=%s",
                case_id,
                slice_id,
                criterion_id,
                judge_id,
                order,
            )

        return ParsedJudgement(
            case_id=case_id,
            layer=layer,
            slice_id=slice_id,
            criterion_id=criterion_id,
            judge_id=judge_id,
            order=order,
            system_a=system_a,
            system_b=system_b,
            response_length_a=response_length_a,
            response_length_b=response_length_b,
            winner=winner,
            is_tie=is_tie,
            is_invalid=is_invalid,
            verdict=verdict,
            response_a_id=system_a,
            response_b_id=system_b,
            canonical_pair_key="__vs__".join(sorted([system_a, system_b])),
            reasoning_excerpt=reasoning,
            raw_response=raw_response,
        )

    @staticmethod
    def judgement_to_dict(judgement: ParsedJudgement) -> dict:
        return asdict(judgement)
