"""
Judge meta-evaluation helpers for the pairwise subsystem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, validator

from .runner import summarise_judge_orders


VALID_GOLD_LABELS = {"system_a", "system_b", "TIE"}
VALID_DIFFICULTIES = {"easy", "ambiguous"}


class JudgeAuditItem(BaseModel):
    """One hand-labelled audit item for judge evaluation."""

    audit_item_id: str
    case_id: str
    criterion_id: str
    canonical_pair_key: str
    gold_label: str
    difficulty: str = "easy"
    prompt_variant_group: Optional[str] = None
    prompt_variant_id: Optional[str] = None
    repeat_group: Optional[str] = None
    repeat_id: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        frozen = True

    @validator("gold_label")
    def _gold_label_must_be_known(cls, value: str) -> str:
        if value not in VALID_GOLD_LABELS:
            raise ValueError(f"unknown gold_label: {value}")
        return value

    @validator("difficulty")
    def _difficulty_must_be_known(cls, value: str) -> str:
        if value not in VALID_DIFFICULTIES:
            raise ValueError(f"unknown difficulty: {value}")
        return value


class JudgeAuditManifest(BaseModel):
    """Manifest for one judge-audit slice."""

    manifest_version: str = "pairwise.judge_audit.v1"
    items: List[JudgeAuditItem]

    class Config:
        frozen = True


def load_judge_audit_manifest(path: str | Path) -> JudgeAuditManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return JudgeAuditManifest.parse_obj(payload)


def compute_judge_audit_report(
    *,
    records: List[Dict[str, Any]],
    manifest: JudgeAuditManifest,
) -> Dict[str, Any]:
    manifest_items = {item.audit_item_id: item for item in manifest.items}
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for record in records:
        audit_item_id = record.get("audit_item_id")
        judge_id = record.get("judge_id")
        if audit_item_id not in manifest_items or not judge_id:
            continue
        grouped.setdefault(judge_id, {}).setdefault(audit_item_id, []).append(record)

    per_judge: Dict[str, Dict[str, Any]] = {}
    for judge_id, judge_items in sorted(grouped.items()):
        item_summaries = {
            audit_item_id: _summarise_audit_item(
                records=judge_records,
                item=manifest_items[audit_item_id],
            )
            for audit_item_id, judge_records in judge_items.items()
        }
        per_judge[judge_id] = _aggregate_judge_audit(
            judge_id=judge_id,
            item_summaries=item_summaries,
        )

    return {
        "manifest_version": manifest.manifest_version,
        "item_count": len(manifest.items),
        "per_judge": per_judge,
    }


def write_judge_audit_report(
    *,
    output_root: str | Path,
    report_name: str,
    report: Dict[str, Any],
) -> Path:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{report_name}.json"
    payload = {"report_name": report_name, **report}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _summarise_audit_item(
    *,
    records: List[Dict[str, Any]],
    item: JudgeAuditItem,
) -> Dict[str, Any]:
    summary = summarise_judge_orders(records)
    predicted_label = _prediction_label(summary=summary, records=records)
    return {
        "audit_item_id": item.audit_item_id,
        "difficulty": item.difficulty,
        "gold_label": item.gold_label,
        "prompt_variant_group": item.prompt_variant_group,
        "repeat_group": item.repeat_group,
        "status": summary["status"],
        "predicted_label": predicted_label,
        "matches_gold": predicted_label == item.gold_label,
        "is_uncertain": summary["status"] != "decisive",
    }


def _aggregate_judge_audit(
    *,
    judge_id: str,
    item_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    summaries = list(item_summaries.values())
    decisive_gold = [row for row in summaries if row["gold_label"] != "TIE"]
    easy = [row for row in summaries if row["difficulty"] == "easy"]
    ambiguous = [row for row in summaries if row["difficulty"] == "ambiguous"]

    prompt_groups: Dict[str, List[Dict[str, Any]]] = {}
    repeat_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in summaries:
        if row["prompt_variant_group"]:
            prompt_groups.setdefault(row["prompt_variant_group"], []).append(row)
        if row["repeat_group"]:
            repeat_groups.setdefault(row["repeat_group"], []).append(row)

    return {
        "judge_id": judge_id,
        "n_items": len(summaries),
        "gold_agreement": _agreement_rate(decisive_gold, lambda row: row["matches_gold"]),
        "swap_consistency": _agreement_rate(summaries, lambda row: row["status"] != "swap_failure"),
        "prompt_invariance": _group_consistency_rate(prompt_groups),
        "sensitivity": _agreement_rate(easy, lambda row: row["matches_gold"]),
        "stability": _group_consistency_rate(repeat_groups),
        "calibration_proxy": {
            "ambiguous_uncertainty_rate": _agreement_rate(ambiguous, lambda row: row["is_uncertain"]),
            "easy_uncertainty_rate": _agreement_rate(easy, lambda row: row["is_uncertain"]),
        },
        "item_summaries": summaries,
    }


def _prediction_label(
    *,
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> str:
    if summary["status"] == "decisive":
        first = records[0]
        if summary["winner"] == first["system_a"]:
            return "system_a"
        if summary["winner"] == first["system_b"]:
            return "system_b"
    return "TIE"


def _agreement_rate(
    rows: List[Dict[str, Any]],
    predicate,
) -> Dict[str, Any]:
    if not rows:
        return {"n": 0, "agreement_rate": 0.0}
    return {
        "n": len(rows),
        "agreement_rate": round(sum(1 for row in rows if predicate(row)) / len(rows), 6),
    }


def _group_consistency_rate(groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    if not groups:
        return {"n": 0, "agreement_rate": 0.0}
    consistent = 0
    for rows in groups.values():
        predicted = {row["predicted_label"] for row in rows}
        if len(predicted) == 1:
            consistent += 1
    return {
        "n": len(groups),
        "agreement_rate": round(consistent / len(groups), 6),
    }
