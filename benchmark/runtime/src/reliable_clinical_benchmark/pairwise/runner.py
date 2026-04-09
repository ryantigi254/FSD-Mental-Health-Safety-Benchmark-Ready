"""
Pairwise evaluation runner.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .config import PairwiseRunSpec
from .parser import PairwiseParser
from .prompt_templates import TEMPLATE_VERSION, TEMPLATES
from .rubric import criteria_for_case

logger = logging.getLogger(__name__)

_DEFAULT_API_BASE = "http://localhost:1234/v1"
_LMSTUDIO_CLIENT_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "lmstudio_client.py"
)
_LMSTUDIO_SPEC = importlib.util.spec_from_file_location(
    "pairwise_lmstudio_client", _LMSTUDIO_CLIENT_PATH
)
if _LMSTUDIO_SPEC is None or _LMSTUDIO_SPEC.loader is None:
    raise ImportError(f"Could not load LM Studio client from {_LMSTUDIO_CLIENT_PATH}")
_LMSTUDIO_MODULE = importlib.util.module_from_spec(_LMSTUDIO_SPEC)
_LMSTUDIO_SPEC.loader.exec_module(_LMSTUDIO_MODULE)
chat_completion = _LMSTUDIO_MODULE.chat_completion


class PairwiseRunner:
    """Execute manifest-driven pairwise judging against local LM Studio models."""

    def __init__(
        self,
        *,
        run_spec: PairwiseRunSpec,
        case_manifest: Dict[str, Any],
        api_base: str = _DEFAULT_API_BASE,
    ) -> None:
        self.run_spec = run_spec
        self.case_manifest = case_manifest
        self.api_base = api_base
        self.parser = PairwiseParser()
        self.output_root = Path(run_spec.config.output_root)
        self.raw_dir = self.output_root / "raw" / run_spec.config.run_id
        self.parsed_dir = self.output_root / "parsed" / run_spec.config.run_id
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

    def planned_calls(self) -> int:
        return self.planned_call_bounds()["max"]

    def planned_call_bounds(self) -> Dict[str, int]:
        group_count = sum(1 for _ in self.iter_comparison_groups())
        orders = len(self.run_spec.config.orders)
        if self.run_spec.config.run_mode == "all_judges":
            total = group_count * orders * len(self.run_spec.judge_manifest.judges)
            return {"min": total, "max": total}
        return {
            "min": group_count * orders * 2,
            "max": group_count * orders * 4,
        }

    def iter_comparison_groups(self) -> Iterable[Dict[str, Any]]:
        for case in self.case_manifest.get("cases", []):
            criteria = [
                criterion_id
                for criterion_id in criteria_for_case(
                    self.run_spec.config.rubric_family, case.get("tags", [])
                )
                if criterion_id in self.run_spec.criteria
            ]
            if not criteria:
                continue

            responses = {
                response["system_id"]: response
                for response in case.get("responses", [])
            }
            pairings = case.get("pairings") or [
                {"system_a": left, "system_b": right}
                for left, right in combinations(sorted(responses), 2)
            ]

            for pairing in pairings:
                system_a = pairing["system_a"]
                system_b = pairing["system_b"]
                if system_a not in responses or system_b not in responses:
                    continue
                response_a = responses[system_a]
                response_b = responses[system_b]
                canonical_pair_key = "__vs__".join(sorted([system_a, system_b]))
                for criterion_id in criteria:
                    yield {
                        "case": case,
                        "criterion_id": criterion_id,
                        "system_a": system_a,
                        "system_b": system_b,
                        "response_a": response_a,
                        "response_b": response_b,
                        "canonical_pair_key": canonical_pair_key,
                        "comparison_key": (
                            f"{case['case_id']}::{criterion_id}::{canonical_pair_key}"
                        ),
                    }

    def run_all(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        raw_records: List[Dict[str, Any]] = []
        parsed_records: List[Dict[str, Any]] = []

        if self.run_spec.config.run_mode == "all_judges":
            for comparison in self.iter_comparison_groups():
                group_raw, group_parsed = self._run_group_all_judges(comparison)
                raw_records.extend(group_raw)
                parsed_records.extend(group_parsed)
                self._persist_records(group_raw, group_parsed)
            return raw_records, parsed_records

        for comparison in self.iter_comparison_groups():
            group_raw, group_parsed = self._run_group_stacked(comparison)
            raw_records.extend(group_raw)
            parsed_records.extend(group_parsed)
            self._persist_records(group_raw, group_parsed)

        return raw_records, parsed_records

    def _run_group_all_judges(
        self, comparison: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        raw_records: List[Dict[str, Any]] = []
        parsed_records: List[Dict[str, Any]] = []
        for judge in self.run_spec.judge_manifest.judges:
            judge_raw, judge_parsed = self._run_orders_for_judge(
                comparison=comparison,
                judge=judge,
            )
            self._annotate_group_records(
                judge_raw,
                judge_parsed,
                judge_stage="all_judges",
                comparison_outcome="all_judges",
                escalation_reasons=[],
                high_risk_forced=False,
                judge_role=judge.role or "panel",
            )
            raw_records.extend(judge_raw)
            parsed_records.extend(judge_parsed)
        return raw_records, parsed_records

    def _run_group_stacked(
        self, comparison: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        primary = self.run_spec.judge_manifest.primary_judge()
        audit = self.run_spec.judge_manifest.audit_judge()
        escalation_judges = self.run_spec.judge_manifest.escalation_judges()
        if primary is None or audit is None or len(escalation_judges) != 2:
            raise ValueError("stacked mode requires one primary, one audit, and two escalation judges")

        raw_records: List[Dict[str, Any]] = []
        parsed_records: List[Dict[str, Any]] = []

        primary_raw, primary_parsed = self._run_orders_for_judge(comparison=comparison, judge=primary)
        audit_raw, audit_parsed = self._run_orders_for_judge(comparison=comparison, judge=audit)
        primary_summary = summarise_judge_orders(primary_parsed)
        audit_summary = summarise_judge_orders(audit_parsed)
        high_risk_forced = self._is_high_risk_case(comparison["case"])
        escalation_reasons = sorted(
            determine_stacked_escalation_reasons(
                primary_summary=primary_summary,
                audit_summary=audit_summary,
                high_risk_forced=high_risk_forced,
                escalate_on=self.run_spec.config.escalate_on,
            )
        )

        if not escalation_reasons:
            self._annotate_group_records(
                primary_raw + audit_raw,
                primary_parsed + audit_parsed,
                judge_stage="routine",
                comparison_outcome="resolved_routine",
                escalation_reasons=[],
                high_risk_forced=False,
            )
            raw_records.extend(primary_raw + audit_raw)
            parsed_records.extend(primary_parsed + audit_parsed)
            return raw_records, parsed_records

        routine_raw = primary_raw + audit_raw
        routine_parsed = primary_parsed + audit_parsed
        escalation_raw: List[Dict[str, Any]] = []
        escalation_parsed: List[Dict[str, Any]] = []
        all_summaries = {
            primary.judge_id: primary_summary,
            audit.judge_id: audit_summary,
        }
        for judge in escalation_judges:
            judge_raw, judge_parsed = self._run_orders_for_judge(
                comparison=comparison,
                judge=judge,
            )
            escalation_raw.extend(judge_raw)
            escalation_parsed.extend(judge_parsed)
            all_summaries[judge.judge_id] = summarise_judge_orders(judge_parsed)

        comparison_outcome = resolve_stacked_outcome(
            all_summaries=all_summaries,
            persistent_disagreement_policy=self.run_spec.config.persistent_disagreement_policy,
        )

        self._annotate_group_records(
            routine_raw,
            routine_parsed,
            judge_stage="routine",
            comparison_outcome=comparison_outcome,
            escalation_reasons=escalation_reasons,
            high_risk_forced=high_risk_forced,
        )
        self._annotate_group_records(
            escalation_raw,
            escalation_parsed,
            judge_stage="escalated",
            comparison_outcome=comparison_outcome,
            escalation_reasons=escalation_reasons,
            high_risk_forced=high_risk_forced,
        )

        raw_records.extend(routine_raw + escalation_raw)
        parsed_records.extend(routine_parsed + escalation_parsed)
        return raw_records, parsed_records

    def _run_orders_for_judge(
        self,
        *,
        comparison: Dict[str, Any],
        judge,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        raw_records: List[Dict[str, Any]] = []
        parsed_records: List[Dict[str, Any]] = []
        for order in self.run_spec.config.orders:
            raw_record, parsed_record = self._run_one_order_for_judge(
                comparison=comparison,
                judge=judge,
                order=order,
            )
            raw_records.append(raw_record)
            parsed_records.append(parsed_record)
        return raw_records, parsed_records

    def _run_one_order_for_judge(
        self,
        *,
        comparison: Dict[str, Any],
        judge,
        order: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        case = comparison["case"]
        response_a = comparison["response_a"]
        response_b = comparison["response_b"]
        display_a, display_b = (
            (response_a, response_b) if order == "AB" else (response_b, response_a)
        )

        template = TEMPLATES[order]
        criterion_id = comparison["criterion_id"]
        prompt = template.format(
            criterion_rubric=self._criterion_prompt(criterion_id),
            case_context=case.get("context", ""),
            response_a=display_a["text"],
            response_b=display_b["text"],
        )

        raw_response = ""
        parsed = None
        for attempt in range(1 + self.run_spec.config.max_retries_per_invalid_parse):
            raw_response = self._call_judge(
                model_string=judge.local_model_id,
                temperature=judge.generation_params.temperature,
                max_tokens=judge.generation_params.max_tokens,
                top_p=judge.generation_params.top_p,
                prompt=prompt,
            )
            parsed = self.parser.parse(
                case_id=case["case_id"],
                layer=self.run_spec.config.layer,
                slice_id=self.run_spec.config.slice_id,
                criterion_id=criterion_id,
                judge_id=judge.judge_id,
                order=order,
                system_a=comparison["system_a"],
                system_b=comparison["system_b"],
                response_length_a=_response_length(response_a),
                response_length_b=_response_length(response_b),
                raw_response=raw_response,
            )
            if not parsed.is_invalid:
                break
            logger.warning(
                "Invalid pairwise parse retry %s/%s for case=%s judge=%s criterion=%s order=%s",
                attempt + 1,
                1 + self.run_spec.config.max_retries_per_invalid_parse,
                case["case_id"],
                judge.judge_id,
                criterion_id,
                order,
            )

        raw_record = {
            "run_id": self.run_spec.config.run_id,
            "layer": self.run_spec.config.layer,
            "slice_id": self.run_spec.config.slice_id,
            "run_mode": self.run_spec.config.run_mode,
            "comparison_key": comparison["comparison_key"],
            "case_id": case["case_id"],
            "criterion_id": criterion_id,
            "judge_id": judge.judge_id,
            "judge_role": judge.role or "panel",
            "judge_hf_source": judge.hf_source,
            "local_model_id": judge.local_model_id,
            "order": order,
            "system_a": comparison["system_a"],
            "system_b": comparison["system_b"],
            "display_system_a": display_a["system_id"],
            "display_system_b": display_b["system_id"],
            "response_length_a": _response_length(response_a),
            "response_length_b": _response_length(response_b),
            "prompt_template_version": TEMPLATE_VERSION,
            "raw_response": raw_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        parsed_record = self.parser.judgement_to_dict(parsed)
        parsed_record.update(
            {
                "run_id": self.run_spec.config.run_id,
                "run_mode": self.run_spec.config.run_mode,
                "comparison_key": comparison["comparison_key"],
                "judge_role": judge.role or "panel",
            }
        )
        return raw_record, parsed_record

    def _annotate_group_records(
        self,
        raw_records: Sequence[Dict[str, Any]],
        parsed_records: Sequence[Dict[str, Any]],
        *,
        judge_stage: str,
        comparison_outcome: str,
        escalation_reasons: List[str],
        high_risk_forced: bool,
        judge_role: str | None = None,
    ) -> None:
        for raw_record, parsed_record in zip(raw_records, parsed_records):
            role = judge_role or parsed_record.get("judge_role") or raw_record.get("judge_role")
            for record in (raw_record, parsed_record):
                record["judge_stage"] = judge_stage
                record["comparison_outcome"] = comparison_outcome
                record["escalation_reason"] = list(escalation_reasons)
                record["high_risk_forced"] = high_risk_forced
                record["judge_role"] = role

    def _persist_records(
        self,
        raw_records: Sequence[Dict[str, Any]],
        parsed_records: Sequence[Dict[str, Any]],
    ) -> None:
        for raw_record in raw_records:
            self._append_jsonl(
                self.raw_dir
                / f"{raw_record['judge_id']}__{self.run_spec.config.slice_id}.jsonl",
                raw_record,
            )
        for parsed_record in parsed_records:
            self._append_jsonl(
                self.parsed_dir
                / f"{parsed_record['judge_id']}__{self.run_spec.config.slice_id}.jsonl",
                parsed_record,
            )

    def _criterion_prompt(self, criterion_id: str) -> str:
        from .rubric import ALL_CRITERIA

        return ALL_CRITERIA[criterion_id].rubric_prompt

    def _is_high_risk_case(self, case: Dict[str, Any]) -> bool:
        tags = set(case.get("tags", []))
        return any(tag in tags for tag in self.run_spec.config.high_risk_tags)

    def _call_judge(
        self,
        *,
        model_string: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        prompt: str,
    ) -> str:
        try:
            return chat_completion(
                api_base=self.api_base,
                model=model_string,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        except Exception:
            logger.exception("LM Studio judge call failed for %s", model_string)
            return ""

    @staticmethod
    def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarise_judge_orders(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_order = {record["order"]: record for record in records}
    if "AB" not in by_order or "BA" not in by_order:
        return {"status": "invalid", "winner": None}
    left = by_order["AB"]
    right = by_order["BA"]
    if left["is_invalid"] or right["is_invalid"]:
        return {"status": "invalid", "winner": None}
    if left["is_tie"] or right["is_tie"]:
        return {"status": "tie", "winner": None}
    if left["winner"] != right["winner"]:
        return {"status": "swap_failure", "winner": None}
    return {"status": "decisive", "winner": left["winner"]}


def determine_stacked_escalation_reasons(
    *,
    primary_summary: Dict[str, Any],
    audit_summary: Dict[str, Any],
    high_risk_forced: bool,
    escalate_on: Sequence[str],
) -> List[str]:
    reasons = set()
    if high_risk_forced and "high_risk" in escalate_on:
        reasons.add("high_risk")
    for summary in (primary_summary, audit_summary):
        if summary["status"] == "invalid" and "invalid" in escalate_on:
            reasons.add("invalid")
        elif summary["status"] == "tie" and "tie" in escalate_on:
            reasons.add("tie")
        elif summary["status"] == "swap_failure" and "swap_failure" in escalate_on:
            reasons.add("swap_failure")
    if (
        primary_summary["status"] == "decisive"
        and audit_summary["status"] == "decisive"
        and primary_summary["winner"] != audit_summary["winner"]
        and "disagreement" in escalate_on
    ):
        reasons.add("disagreement")
    return sorted(reasons)


def resolve_stacked_outcome(
    *,
    all_summaries: Dict[str, Dict[str, Any]],
    persistent_disagreement_policy: str,
) -> str:
    if persistent_disagreement_policy != "mark_uncertain":
        raise ValueError("unsupported persistent_disagreement_policy")
    decisive_winners = []
    for summary in all_summaries.values():
        if summary["status"] != "decisive":
            return "uncertain"
        decisive_winners.append(summary["winner"])
    if len(set(decisive_winners)) != 1:
        return "uncertain"
    return "resolved_escalated"


def _response_length(response: Dict[str, Any]) -> int:
    token_count = response.get("token_count")
    if isinstance(token_count, int) and token_count > 0:
        return token_count
    word_count = response.get("word_count")
    if isinstance(word_count, int) and word_count > 0:
        return word_count
    return len(str(response.get("text", "")).split())
