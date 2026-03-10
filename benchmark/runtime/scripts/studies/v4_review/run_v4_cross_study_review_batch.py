#!/usr/bin/env python3
"""Run v4 cross-study deterministic review in fixed-size batches (default 20 rows)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reliable_clinical_benchmark.review import (
    load_rules,
    now_iso,
    read_existing_ssv_state,
    score_study_a_case,
    score_study_b_multi_case,
    score_study_b_single_case,
    score_study_c_case,
    write_ssv_row,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data"
SPLITS_ROOT = DATA_ROOT / "openr1_psy_splits"
GOLD_ROOT = DATA_ROOT / "study_a_gold"
DEFAULT_OUT_DIR = DATA_ROOT / "verification" / "v4"
DEFAULT_RULES_PATH = DEFAULT_OUT_DIR / "rubric_rules_v1.json"

EXPECTED_COUNTS = {
    "study_a": 2000,
    "study_b_single": 2000,
    "study_b_multi": 120,
    "study_c": 100,
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_items(payload: Any, preferred_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unable to normalise payload with keys={preferred_keys}")


def _join_pipe(values: list[Any]) -> str:
    return "|".join(str(v).strip() for v in values if str(v).strip())


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader)


def _summarise_file(path: Path, expected: int) -> dict[str, int]:
    rows = _read_rows(path)
    counts = Counter((row.get("verdict", "") or "").strip() for row in rows)
    ids = [str(row.get("item_id", "")).strip() for row in rows]
    duplicate_ids = len(ids) - len(set(ids))
    return {
        "actual_rows": len(rows),
        "expected_rows": expected,
        "acceptable": int(counts.get("ACCEPTABLE", 0)),
        "needs_review": int(counts.get("NEEDS_REVIEW", 0)),
        "reject": int(counts.get("REJECT", 0)),
        "duplicate_item_ids": duplicate_ids,
    }


def _write_summary(out_dir: Path) -> None:
    summary_path = out_dir / "review_summary.ssv"
    targets = {
        "study_a": out_dir / "study_a_reference_verdicts.ssv",
        "study_b_single": out_dir / "study_b_single_verdicts.ssv",
        "study_b_multi": out_dir / "study_b_multi_verdicts.ssv",
        "study_c": out_dir / "study_c_verdicts.ssv",
    }

    fieldnames = [
        "study",
        "expected_rows",
        "actual_rows",
        "acceptable",
        "needs_review",
        "reject",
        "review_scope",
        "counts_match",
        "duplicate_item_ids",
        "notes",
    ]

    rows = []
    for study, path in targets.items():
        stats = _summarise_file(path, EXPECTED_COUNTS[study])
        scope = "study_a_full_rubric" if study == "study_a" else ("study_c_mapped" if study == "study_c" else "study_b_mapped")
        rows.append(
            {
                "study": study,
                "expected_rows": stats["expected_rows"],
                "actual_rows": stats["actual_rows"],
                "acceptable": stats["acceptable"],
                "needs_review": stats["needs_review"],
                "reject": stats["reject"],
                "review_scope": scope,
                "counts_match": int(stats["actual_rows"] == stats["expected_rows"]),
                "duplicate_item_ids": stats["duplicate_item_ids"],
                "notes": "",
            }
        )

    replacement_path = out_dir / "replacement_candidates_study_a.ssv"
    replacement_rows = _read_rows(replacement_path)
    accepted_candidates = sum(1 for r in replacement_rows if (r.get("accepted", "") or "").strip() == "1")
    rows.append(
        {
            "study": "study_a_replacements",
            "expected_rows": "",
            "actual_rows": len(replacement_rows),
            "acceptable": accepted_candidates,
            "needs_review": "",
            "reject": "",
            "review_scope": "study_a_full_rubric",
            "counts_match": "",
            "duplicate_item_ids": "",
            "notes": "batch_mode",
        }
    )

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _process_study_a(
    items: list[dict[str, Any]],
    labels: dict[str, str],
    rules: dict[str, Any],
    out_path: Path,
    batch_size: int,
    reset: bool,
) -> int:
    fieldnames = [
        "study",
        "row_number",
        "item_id",
        "verdict",
        "reason_codes",
        "review_scope",
        "review_timestamp_utc",
        "C1_complete_thinking",
        "C2_context_coherence",
        "C3_response_alignment",
        "C4_framework_adherence",
        "B1_diagnostic_plausibility",
        "B2_safety_recognition",
        "B3_therapeutic_safety",
        "risk_markers_present",
        "risk_markers_addressed",
        "gold_diagnosis_label",
        "source_openr1_ids",
    ]

    if reset and out_path.exists():
        out_path.unlink()

    existing_ids, _ = read_existing_ssv_state(out_path)

    sorted_items = sorted(items, key=lambda x: str(x.get("id", "")))
    id_to_rownum = {str(item.get("id", "")).strip(): i for i, item in enumerate(sorted_items, start=1)}
    pending = [item for item in sorted_items if str(item.get("id", "")).strip() not in existing_ids]
    batch = pending[:batch_size]

    for item in batch:
        item_id = str(item.get("id", "")).strip()
        review = score_study_a_case(item=item, gold_label=str(labels.get(item_id, "")), rules=rules)
        write_ssv_row(
            out_path,
            fieldnames,
            {
                "study": "study_a",
                "row_number": id_to_rownum[item_id],
                "item_id": item_id,
                "verdict": review["verdict"],
                "reason_codes": _join_pipe(review.get("reason_codes", [])),
                "review_scope": "study_a_full_rubric",
                "review_timestamp_utc": now_iso(),
                "C1_complete_thinking": review.get("C1_complete_thinking", ""),
                "C2_context_coherence": review.get("C2_context_coherence", ""),
                "C3_response_alignment": review.get("C3_response_alignment", ""),
                "C4_framework_adherence": review.get("C4_framework_adherence", ""),
                "B1_diagnostic_plausibility": review.get("B1_diagnostic_plausibility", ""),
                "B2_safety_recognition": review.get("B2_safety_recognition", ""),
                "B3_therapeutic_safety": review.get("B3_therapeutic_safety", ""),
                "risk_markers_present": _join_pipe(review.get("risk_markers_present", [])),
                "risk_markers_addressed": _join_pipe(review.get("risk_markers_addressed", [])),
                "gold_diagnosis_label": str(labels.get(item_id, "")),
                "source_openr1_ids": _join_pipe((item.get("metadata", {}) or {}).get("source_openr1_ids", [])),
            },
        )

    return len(batch)


def _process_study_b_single(items: list[dict[str, Any]], out_path: Path, batch_size: int, reset: bool) -> int:
    fieldnames = [
        "study",
        "row_number",
        "item_id",
        "verdict",
        "reason_codes",
        "review_scope",
        "review_timestamp_utc",
        "prompt_nonempty",
        "gold_answer_nonempty",
        "incorrect_opinion_nonempty",
        "persona_id_present",
        "id_unique",
        "source_openr1_ids_valid",
        "source_split_valid",
        "generated_only_when_source_empty",
        "mapped_contract_pass",
    ]

    if reset and out_path.exists():
        out_path.unlink()

    existing_ids, _ = read_existing_ssv_state(out_path)
    sorted_items = sorted(items, key=lambda x: str(x.get("id", "")))
    id_to_rownum = {str(item.get("id", "")).strip(): i for i, item in enumerate(sorted_items, start=1)}
    id_counts = Counter(str(item.get("id", "")).strip() for item in sorted_items)

    pending = [item for item in sorted_items if str(item.get("id", "")).strip() not in existing_ids]
    batch = pending[:batch_size]

    for item in batch:
        item_id = str(item.get("id", "")).strip()
        review = score_study_b_single_case(item=item, id_unique=id_counts[item_id] == 1)
        write_ssv_row(
            out_path,
            fieldnames,
            {
                "study": "study_b_single",
                "row_number": id_to_rownum[item_id],
                "item_id": item_id,
                "verdict": review["verdict"],
                "reason_codes": _join_pipe(review.get("reason_codes", [])),
                "review_scope": "study_b_mapped",
                "review_timestamp_utc": now_iso(),
                "prompt_nonempty": review.get("prompt_nonempty", ""),
                "gold_answer_nonempty": review.get("gold_answer_nonempty", ""),
                "incorrect_opinion_nonempty": review.get("incorrect_opinion_nonempty", ""),
                "persona_id_present": review.get("persona_id_present", ""),
                "id_unique": review.get("id_unique", ""),
                "source_openr1_ids_valid": review.get("source_openr1_ids_valid", ""),
                "source_split_valid": review.get("source_split_valid", ""),
                "generated_only_when_source_empty": review.get("generated_only_when_source_empty", ""),
                "mapped_contract_pass": review.get("mapped_contract_pass", ""),
            },
        )

    return len(batch)


def _process_study_b_multi(items: list[dict[str, Any]], out_path: Path, batch_size: int, reset: bool) -> int:
    fieldnames = [
        "study",
        "row_number",
        "item_id",
        "verdict",
        "reason_codes",
        "review_scope",
        "review_timestamp_utc",
        "gold_answer_nonempty",
        "incorrect_opinion_nonempty",
        "pressure_style_nonempty",
        "pressure_schedule_nonempty",
        "turns_nonempty",
        "turns_all_have_message",
        "turns_all_pressure_level_valid",
        "persona_id_present",
        "source_openr1_ids_present",
        "source_split_valid",
        "mapped_contract_pass",
    ]

    if reset and out_path.exists():
        out_path.unlink()

    existing_ids, _ = read_existing_ssv_state(out_path)
    sorted_items = sorted(items, key=lambda x: str(x.get("id", "")))
    id_to_rownum = {str(item.get("id", "")).strip(): i for i, item in enumerate(sorted_items, start=1)}

    pending = [item for item in sorted_items if str(item.get("id", "")).strip() not in existing_ids]
    batch = pending[:batch_size]

    for item in batch:
        item_id = str(item.get("id", "")).strip()
        review = score_study_b_multi_case(item)
        write_ssv_row(
            out_path,
            fieldnames,
            {
                "study": "study_b_multi",
                "row_number": id_to_rownum[item_id],
                "item_id": item_id,
                "verdict": review["verdict"],
                "reason_codes": _join_pipe(review.get("reason_codes", [])),
                "review_scope": "study_b_mapped",
                "review_timestamp_utc": now_iso(),
                "gold_answer_nonempty": review.get("gold_answer_nonempty", ""),
                "incorrect_opinion_nonempty": review.get("incorrect_opinion_nonempty", ""),
                "pressure_style_nonempty": review.get("pressure_style_nonempty", ""),
                "pressure_schedule_nonempty": review.get("pressure_schedule_nonempty", ""),
                "turns_nonempty": review.get("turns_nonempty", ""),
                "turns_all_have_message": review.get("turns_all_have_message", ""),
                "turns_all_pressure_level_valid": review.get("turns_all_pressure_level_valid", ""),
                "persona_id_present": review.get("persona_id_present", ""),
                "source_openr1_ids_present": review.get("source_openr1_ids_present", ""),
                "source_split_valid": review.get("source_split_valid", ""),
                "mapped_contract_pass": review.get("mapped_contract_pass", ""),
            },
        )

    return len(batch)


def _process_study_c(items: list[dict[str, Any]], out_path: Path, batch_size: int, reset: bool) -> int:
    fieldnames = [
        "study",
        "row_number",
        "item_id",
        "verdict",
        "reason_codes",
        "review_scope",
        "review_timestamp_utc",
        "patient_summary_nonempty",
        "critical_entities_nonempty",
        "turns_nonempty",
        "persona_id_present",
        "source_openr1_ids_present",
        "num_turns_matches_turns_length",
        "mapped_contract_pass",
    ]

    if reset and out_path.exists():
        out_path.unlink()

    existing_ids, _ = read_existing_ssv_state(out_path)
    sorted_items = sorted(items, key=lambda x: str(x.get("id", "")))
    id_to_rownum = {str(item.get("id", "")).strip(): i for i, item in enumerate(sorted_items, start=1)}

    pending = [item for item in sorted_items if str(item.get("id", "")).strip() not in existing_ids]
    batch = pending[:batch_size]

    for item in batch:
        item_id = str(item.get("id", "")).strip()
        review = score_study_c_case(item)
        write_ssv_row(
            out_path,
            fieldnames,
            {
                "study": "study_c",
                "row_number": id_to_rownum[item_id],
                "item_id": item_id,
                "verdict": review["verdict"],
                "reason_codes": _join_pipe(review.get("reason_codes", [])),
                "review_scope": "study_c_mapped",
                "review_timestamp_utc": now_iso(),
                "patient_summary_nonempty": review.get("patient_summary_nonempty", ""),
                "critical_entities_nonempty": review.get("critical_entities_nonempty", ""),
                "turns_nonempty": review.get("turns_nonempty", ""),
                "persona_id_present": review.get("persona_id_present", ""),
                "source_openr1_ids_present": review.get("source_openr1_ids_present", ""),
                "num_turns_matches_turns_length": review.get("num_turns_matches_turns_length", ""),
                "mapped_contract_pass": review.get("mapped_contract_pass", ""),
            },
        )

    return len(batch)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v4 review in fixed-size batches per study.")
    parser.add_argument(
        "--study",
        choices=["study_a", "study_b_single", "study_b_multi", "study_c", "all"],
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--reset", action="store_true", help="Reset selected study SSV(s) before running batch")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = load_rules(args.rules)
    labels = _load_json(GOLD_ROOT / "gold_diagnosis_labels.json").get("labels", {})

    study_a_items = _normalise_items(_load_json(SPLITS_ROOT / "study_a_test.json"), ("samples", "items", "cases"))
    study_b_single_items = _normalise_items(_load_json(SPLITS_ROOT / "study_b_test.json"), ("samples", "items"))
    study_b_multi_items = _normalise_items(_load_json(SPLITS_ROOT / "study_b_multi_turn_test.json"), ("multi_turn_cases", "cases", "items"))
    study_c_items = _normalise_items(_load_json(SPLITS_ROOT / "study_c_test.json"), ("cases", "samples", "items"))

    targets = [args.study] if args.study != "all" else ["study_a", "study_b_single", "study_b_multi", "study_c"]
    written = {}

    for study in targets:
        if study == "study_a":
            written[study] = _process_study_a(
                items=study_a_items,
                labels=labels,
                rules=rules,
                out_path=out_dir / "study_a_reference_verdicts.ssv",
                batch_size=args.batch_size,
                reset=args.reset,
            )
        elif study == "study_b_single":
            written[study] = _process_study_b_single(
                items=study_b_single_items,
                out_path=out_dir / "study_b_single_verdicts.ssv",
                batch_size=args.batch_size,
                reset=args.reset,
            )
        elif study == "study_b_multi":
            written[study] = _process_study_b_multi(
                items=study_b_multi_items,
                out_path=out_dir / "study_b_multi_verdicts.ssv",
                batch_size=args.batch_size,
                reset=args.reset,
            )
        elif study == "study_c":
            written[study] = _process_study_c(
                items=study_c_items,
                out_path=out_dir / "study_c_verdicts.ssv",
                batch_size=args.batch_size,
                reset=args.reset,
            )

    _write_summary(out_dir)

    metadata_path = out_dir / "run_metadata.json"
    previous_meta = _load_json(metadata_path) if metadata_path.exists() else {}
    previous_batches = int(previous_meta.get("batch_runs", 0) or 0)
    metadata = {
        "mode": "batch",
        "batch_runs": previous_batches + 1,
        "last_batch_size": args.batch_size,
        "last_targets": targets,
        "last_run_utc": now_iso(),
        "rule_file": str(args.rules),
        "rule_version": str(rules.get("rule_version", "unknown")),
        "last_rows_written": written,
        "summary": {
            study: _summarise_file(
                out_dir / (
                    "study_a_reference_verdicts.ssv" if study == "study_a"
                    else "study_b_single_verdicts.ssv" if study == "study_b_single"
                    else "study_b_multi_verdicts.ssv" if study == "study_b_multi"
                    else "study_c_verdicts.ssv"
                ),
                EXPECTED_COUNTS[study],
            )
            for study in ["study_a", "study_b_single", "study_b_multi", "study_c"]
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    for study in targets:
        print(f"{study}: wrote {written.get(study, 0)} row(s) this run")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
