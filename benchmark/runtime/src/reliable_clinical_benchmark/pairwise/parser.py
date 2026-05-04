"""
Parse judge output into a canonical pairwise record.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

_VERDICT_LINE_PATTERNS = [
    (re.compile(r"^\s*(?:verdict\s*:\s*)?\[\[\s*(?:response\s+)?a\s*\]\]\s*$", re.IGNORECASE), "A"),
    (re.compile(r"^\s*(?:verdict\s*:\s*)?\[\[\s*(?:response\s+)?b\s*\]\]\s*$", re.IGNORECASE), "B"),
    (re.compile(r"^\s*(?:verdict\s*:\s*)?\[\[\s*tie\s*\]\]\s*$", re.IGNORECASE), "TIE"),
    (re.compile(r"^\s*(?:verdict\s*:\s*)?\[\s*(?:response\s+)?a\s*\]\s*$", re.IGNORECASE), "A"),
    (re.compile(r"^\s*(?:verdict\s*:\s*)?\[\s*(?:response\s+)?b\s*\]\s*$", re.IGNORECASE), "B"),
    (re.compile(r"^\s*(?:verdict\s*:\s*)?\[\s*tie\s*\]\s*$", re.IGNORECASE), "TIE"),
    (re.compile(r"^\s*(?:winner|verdict|choice)\s*:\s*(?:response\s+)?a\s*$", re.IGNORECASE), "A"),
    (re.compile(r"^\s*(?:winner|verdict|choice)\s*:\s*(?:response\s+)?b\s*$", re.IGNORECASE), "B"),
    (re.compile(r"^\s*(?:winner|verdict|choice)\s*:\s*tie\s*$", re.IGNORECASE), "TIE"),
    (re.compile(r"^\s*(?:verdict\s*:\s*)?a\s*$", re.IGNORECASE), "A"),
    (re.compile(r"^\s*(?:verdict\s*:\s*)?b\s*$", re.IGNORECASE), "B"),
    (re.compile(r"^\s*(?:verdict\s*:\s*)?tie\s*$", re.IGNORECASE), "TIE"),
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
    def _verdict_line_index(text: str) -> tuple[int, str] | None:
        lines = str(text or "").splitlines()
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx]
            if not line.strip():
                continue
            for pattern, label in _VERDICT_LINE_PATTERNS:
                if pattern.match(line):
                    return idx, label
            return None
        return None

    @classmethod
    def extract_verdict(cls, text: str) -> str:
        verdict = cls._verdict_line_index(text)
        return verdict[1] if verdict is not None else "INVALID"

    @classmethod
    def extract_reasoning(cls, text: str) -> str:
        verdict = cls._verdict_line_index(text)
        if verdict is not None:
            lines = str(text or "").splitlines()
            excerpt = "\n".join(lines[: verdict[0]]).strip()
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
