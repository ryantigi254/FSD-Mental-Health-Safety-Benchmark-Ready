"""
Pairwise evaluation runner.
"""

from __future__ import annotations

import json
import logging
import importlib.util
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .config import PairwiseRunSpec
from .parser import PairwiseParser
from .prompt_templates import TEMPLATES, TEMPLATE_VERSION
from .rubric import criteria_for_case

logger = logging.getLogger(__name__)

_DEFAULT_API_BASE = "http://localhost:1234/v1"
_LMSTUDIO_CLIENT_PATH = Path(__file__).resolve().parent.parent / "models" / "lmstudio_client.py"
_LMSTUDIO_SPEC = importlib.util.spec_from_file_location("pairwise_lmstudio_client", _LMSTUDIO_CLIENT_PATH)
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
        return sum(1 for _ in self.iter_comparisons())

    def iter_comparisons(self) -> Iterable[Dict[str, Any]]:
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
                for criterion_id in criteria:
                    for order in self.run_spec.config.orders:
                        yield {
                            "case": case,
                            "criterion_id": criterion_id,
                            "order": order,
                            "system_a": system_a,
                            "system_b": system_b,
                            "response_a": response_a,
                            "response_b": response_b,
                        }

    def run_all(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        raw_records: List[Dict[str, Any]] = []
        parsed_records: List[Dict[str, Any]] = []

        for comparison in self.iter_comparisons():
            for judge in self.run_spec.judge_manifest.judges:
                raw_record, parsed_record = self._run_comparison_for_judge(
                    comparison=comparison,
                    judge=judge,
                )
                raw_records.append(raw_record)
                parsed_records.append(parsed_record)

        return raw_records, parsed_records

    def _run_comparison_for_judge(
        self,
        *,
        comparison: Dict[str, Any],
        judge,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        case = comparison["case"]
        order = comparison["order"]
        response_a = comparison["response_a"]
        response_b = comparison["response_b"]
        display_a, display_b = (
            (response_a, response_b) if order == "AB" else (response_b, response_a)
        )

        template = TEMPLATES[order]
        criterion = next(
            criterion
            for criterion in self.run_spec.criteria
            if criterion == comparison["criterion_id"]
        )
        prompt = template.format(
            criterion_rubric=self._criterion_prompt(criterion),
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
                criterion_id=comparison["criterion_id"],
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
                comparison["criterion_id"],
                order,
            )

        raw_record = {
            "run_id": self.run_spec.config.run_id,
            "layer": self.run_spec.config.layer,
            "slice_id": self.run_spec.config.slice_id,
            "case_id": case["case_id"],
            "criterion_id": comparison["criterion_id"],
            "judge_id": judge.judge_id,
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

        self._append_jsonl(
            self.raw_dir / f"{judge.judge_id}__{self.run_spec.config.slice_id}.jsonl",
            raw_record,
        )
        self._append_jsonl(
            self.parsed_dir / f"{judge.judge_id}__{self.run_spec.config.slice_id}.jsonl",
            parsed_record,
        )
        return raw_record, parsed_record

    def _criterion_prompt(self, criterion_id: str) -> str:
        from .rubric import ALL_CRITERIA

        return ALL_CRITERIA[criterion_id].rubric_prompt

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


def _response_length(response: Dict[str, Any]) -> int:
    token_count = response.get("token_count")
    if isinstance(token_count, int) and token_count > 0:
        return token_count
    word_count = response.get("word_count")
    if isinstance(word_count, int) and word_count > 0:
        return word_count
    return len(str(response.get("text", "")).split())
