#!/usr/bin/env python3
"""Run deterministic v4 cross-study reference review and emit SSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from reliable_clinical_benchmark.review import (
    build_replacement_candidates_study_a,
    load_rules,
    now_iso,
    read_existing_ssv_state,
    score_study_a,
    score_study_b_multi,
    score_study_b_single,
    score_study_c,
    write_ssv_row,
)


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data"
DEFAULT_INPUT_ROOT = DATA_ROOT / "frozen_splits" / "v0.3_postclinician_audit"
DEFAULT_OUT_DIR = DATA_ROOT / "verification" / "v4"
DEFAULT_RULES_PATH = DEFAULT_OUT_DIR / "rubric_rules_v2.json"

EXPECTED_ROWS = {
    "study_a": 2000,
    "study_b_single": 2000,
    "study_b_multi": 120,
    "study_c": 100,
}

VALID_VERDICTS = {"ACCEPTABLE", "NEEDS_REVIEW", "REJECT"}


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
    raise ValueError(f"Unable to normalise payload using keys={preferred_keys}")


def _join_pipe(values: list[Any]) -> str:
    return "|".join(str(v).strip() for v in values if str(v).strip())


def _clear_outputs(out_dir: Path) -> None:
    for filename in [
        "study_a_reference_verdicts.ssv",
        "study_b_single_verdicts.ssv",
        "study_b_multi_verdicts.ssv",
        "study_c_verdicts.ssv",
        "review_summary.ssv",
        "replacement_candidates_study_a.ssv",
        "run_metadata.json",
    ]:
        path = out_dir / filename
        if path.exists():
            path.unlink()


def _score_and_write(
    *,
    study: str,
    items: list[dict[str, Any]],
    output_path: Path,
    fieldnames: list[str],
    review_scope: str,
    scorer: Callable[[dict[str, Any]], dict[str, Any]],
    row_builder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    existing_ids, existing_rows = read_existing_ssv_state(output_path)
    wrote_rows = 0
    scored_rows: list[dict[str, Any]] = []

    for row_number, item in enumerate(items, start=1):
        item_id = str(item.get("id", "") or "").strip()
        review = scorer(item)
        scored_rows.append({"item_id": item_id, **review})

        row = {
            "study": study,
            "row_number": row_number,
            "item_id": item_id,
            "verdict": review["verdict"],
            "reason_codes": _join_pipe(review.get("reason_codes", [])),
            "review_scope": review_scope,
            "review_timestamp_utc": now_iso(),
        }
        row.update(row_builder(item, review))

        if item_id in existing_ids:
            continue

        write_ssv_row(output_path, fieldnames, row)
        existing_ids.add(item_id)
        wrote_rows += 1

    return {
        "rows_preexisting": existing_rows,
        "rows_written": wrote_rows,
        "total_items": len(items),
        "scored_rows": scored_rows,
    }


def _summarise_study_file(path: Path, expected_rows: int) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing output file: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = list(reader)

    verdicts = Counter(str(row.get("verdict", "") or "").strip() for row in rows)
    item_ids = [str(row.get("item_id", "") or "").strip() for row in rows]
    duplicate_item_ids = len(item_ids) - len(set(item_ids))

    return {
        "actual_rows": len(rows),
        "expected_rows": expected_rows,
        "acceptable": int(verdicts.get("ACCEPTABLE", 0)),
        "needs_review": int(verdicts.get("NEEDS_REVIEW", 0)),
        "reject": int(verdicts.get("REJECT", 0)),
        "counts_match": int(len(rows) == expected_rows),
        "duplicate_item_ids": duplicate_item_ids,
    }


def _write_review_summary(
    summary_path: Path,
    study_summaries: dict[str, dict[str, Any]],
    replacement_stats: dict[str, Any],
) -> None:
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

    rows = [
        {
            "study": "study_a",
            "expected_rows": study_summaries["study_a"]["expected_rows"],
            "actual_rows": study_summaries["study_a"]["actual_rows"],
            "acceptable": study_summaries["study_a"]["acceptable"],
            "needs_review": study_summaries["study_a"]["needs_review"],
            "reject": study_summaries["study_a"]["reject"],
            "review_scope": "study_a_full_rubric",
            "counts_match": study_summaries["study_a"]["counts_match"],
            "duplicate_item_ids": study_summaries["study_a"]["duplicate_item_ids"],
            "notes": "",
        },
        {
            "study": "study_b_single",
            "expected_rows": study_summaries["study_b_single"]["expected_rows"],
            "actual_rows": study_summaries["study_b_single"]["actual_rows"],
            "acceptable": study_summaries["study_b_single"]["acceptable"],
            "needs_review": study_summaries["study_b_single"]["needs_review"],
            "reject": study_summaries["study_b_single"]["reject"],
            "review_scope": "study_b_mapped",
            "counts_match": study_summaries["study_b_single"]["counts_match"],
            "duplicate_item_ids": study_summaries["study_b_single"]["duplicate_item_ids"],
            "notes": "",
        },
        {
            "study": "study_b_multi",
            "expected_rows": study_summaries["study_b_multi"]["expected_rows"],
            "actual_rows": study_summaries["study_b_multi"]["actual_rows"],
            "acceptable": study_summaries["study_b_multi"]["acceptable"],
            "needs_review": study_summaries["study_b_multi"]["needs_review"],
            "reject": study_summaries["study_b_multi"]["reject"],
            "review_scope": "study_b_mapped",
            "counts_match": study_summaries["study_b_multi"]["counts_match"],
            "duplicate_item_ids": study_summaries["study_b_multi"]["duplicate_item_ids"],
            "notes": "",
        },
        {
            "study": "study_c",
            "expected_rows": study_summaries["study_c"]["expected_rows"],
            "actual_rows": study_summaries["study_c"]["actual_rows"],
            "acceptable": study_summaries["study_c"]["acceptable"],
            "needs_review": study_summaries["study_c"]["needs_review"],
            "reject": study_summaries["study_c"]["reject"],
            "review_scope": "study_c_mapped",
            "counts_match": study_summaries["study_c"]["counts_match"],
            "duplicate_item_ids": study_summaries["study_c"]["duplicate_item_ids"],
            "notes": "",
        },
    ]

    rows.append(
        {
            "study": "study_a_replacements",
            "expected_rows": "",
            "actual_rows": replacement_stats.get("replace_count", 0),
            "acceptable": replacement_stats.get("accepted_candidates", 0),
            "needs_review": "",
            "reject": "",
            "review_scope": "study_a_full_rubric",
            "counts_match": "",
            "duplicate_item_ids": "",
            "notes": replacement_stats.get("notes", ""),
        }
    )

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_metadata(
    metadata_path: Path,
    rules_path: Path,
    rules: dict[str, Any],
    study_summaries: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "mode": "batch",
        "rule_file": str(rules_path.resolve()),
        "rule_version": str(rules.get("rule_version", "")),
        "matching_mode": str(rules.get("matching_mode", "")),
        "last_run_utc": now_iso(),
        "summary": {
            "study_a": {
                "actual_rows": study_summaries["study_a"]["actual_rows"],
                "expected_rows": study_summaries["study_a"]["expected_rows"],
                "acceptable": study_summaries["study_a"]["acceptable"],
                "needs_review": study_summaries["study_a"]["needs_review"],
                "reject": study_summaries["study_a"]["reject"],
                "duplicate_item_ids": study_summaries["study_a"]["duplicate_item_ids"],
            },
            "study_b_single": {
                "actual_rows": study_summaries["study_b_single"]["actual_rows"],
                "expected_rows": study_summaries["study_b_single"]["expected_rows"],
                "acceptable": study_summaries["study_b_single"]["acceptable"],
                "needs_review": study_summaries["study_b_single"]["needs_review"],
                "reject": study_summaries["study_b_single"]["reject"],
                "duplicate_item_ids": study_summaries["study_b_single"]["duplicate_item_ids"],
            },
            "study_b_multi": {
                "actual_rows": study_summaries["study_b_multi"]["actual_rows"],
                "expected_rows": study_summaries["study_b_multi"]["expected_rows"],
                "acceptable": study_summaries["study_b_multi"]["acceptable"],
                "needs_review": study_summaries["study_b_multi"]["needs_review"],
                "reject": study_summaries["study_b_multi"]["reject"],
                "duplicate_item_ids": study_summaries["study_b_multi"]["duplicate_item_ids"],
            },
            "study_c": {
                "actual_rows": study_summaries["study_c"]["actual_rows"],
                "expected_rows": study_summaries["study_c"]["expected_rows"],
                "acceptable": study_summaries["study_c"]["acceptable"],
                "needs_review": study_summaries["study_c"]["needs_review"],
                "reject": study_summaries["study_c"]["reject"],
                "duplicate_item_ids": study_summaries["study_c"]["duplicate_item_ids"],
            },
        },
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _assert_summary_integrity(study_summaries: dict[str, dict[str, Any]]) -> None:
    for study_name, summary in study_summaries.items():
        if summary["counts_match"] != 1:
            raise RuntimeError(
                f"Row-count mismatch for {study_name}: "
                f"expected={summary['expected_rows']} actual={summary['actual_rows']}"
            )
        if summary["duplicate_item_ids"] != 0:
            raise RuntimeError(
                f"Duplicate item IDs found for {study_name}: {summary['duplicate_item_ids']}"
            )


def _print_verdict_table(study_summaries: dict[str, dict[str, Any]]) -> None:
    print("study | acceptable | needs_review | reject | actual_rows | expected_rows")
    print("----- | ---------- | ------------ | ------ | ----------- | -------------")
    for study in ["study_a", "study_b_single", "study_b_multi", "study_c"]:
        summary = study_summaries[study]
        print(
            f"{study} | {summary['acceptable']} | {summary['needs_review']} | {summary['reject']} "
            f"| {summary['actual_rows']} | {summary['expected_rows']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic v4 cross-study rubric review.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--clean", action="store_true", help="Clear existing v4 outputs before scoring")
    args = parser.parse_args()

    input_root = args.input_root
    out_dir = args.out_dir
    rules_path = args.rules

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _clear_outputs(out_dir)

    rules = load_rules(rules_path)
    if str(rules.get("rule_version", "")) != "v2":
        raise RuntimeError(f"Expected rule_version=v2, got {rules.get('rule_version')}")
    if str(rules.get("matching_mode", "")) != "word_boundary":
        raise RuntimeError(f"Expected matching_mode=word_boundary, got {rules.get('matching_mode')}")

    labels_payload = _load_json(input_root / "gold_diagnosis_labels.json")
    labels = labels_payload.get("labels", {})
    if not isinstance(labels, dict):
        raise RuntimeError("gold_diagnosis_labels.json must include a 'labels' object")

    study_a_items = _normalise_items(
        _load_json(input_root / "study_a_test.json"),
        ("samples", "cases", "items"),
    )
    study_b_single_items = _normalise_items(
        _load_json(input_root / "study_b_test.json"),
        ("samples", "items"),
    )
    study_b_multi_items = _normalise_items(
        _load_json(input_root / "study_b_multi_turn_test.json"),
        ("multi_turn_cases", "cases", "items"),
    )

    study_c_path = input_root / "study_c_test.json"
    if not study_c_path.exists():
        study_c_path = input_root / "study_c" / "study_c_test.json"
    study_c_items = _normalise_items(
        _load_json(study_c_path),
        ("cases", "samples", "items"),
    )

    study_a_path = out_dir / "study_a_reference_verdicts.ssv"
    study_b_single_path = out_dir / "study_b_single_verdicts.ssv"
    study_b_multi_path = out_dir / "study_b_multi_verdicts.ssv"
    study_c_path_out = out_dir / "study_c_verdicts.ssv"
    summary_path = out_dir / "review_summary.ssv"
    replacement_path = out_dir / "replacement_candidates_study_a.ssv"
    metadata_path = out_dir / "run_metadata.json"

    common_cols = [
        "study",
        "row_number",
        "item_id",
        "verdict",
        "reason_codes",
        "review_scope",
        "review_timestamp_utc",
    ]

    study_a_cols = common_cols + [
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

    study_b_single_cols = common_cols + [
        "prompt_nonempty",
        "gold_answer_nonempty",
        "incorrect_opinion_nonempty",
        "persona_id_present",
        "id_unique",
        "mapped_contract_pass",
    ]

    study_b_multi_cols = common_cols + [
        "gold_answer_nonempty",
        "incorrect_opinion_nonempty",
        "pressure_style_nonempty",
        "pressure_schedule_nonempty",
        "turns_nonempty",
        "turns_all_have_message",
        "turns_all_pressure_level_valid",
        "persona_id_present",
        "mapped_contract_pass",
    ]

    study_c_cols = common_cols + [
        "patient_summary_nonempty",
        "critical_entities_nonempty",
        "turns_nonempty",
        "persona_id_present",
        "source_openr1_ids_present",
        "num_turns_matches_turns_length",
        "mapped_contract_pass",
    ]

    study_b_id_counts = Counter(str(item.get("id", "") or "").strip() for item in study_b_single_items)

    study_a_run = _score_and_write(
        study="study_a",
        items=study_a_items,
        output_path=study_a_path,
        fieldnames=study_a_cols,
        review_scope="study_a_full_rubric",
        scorer=lambda item: score_study_a(
            item,
            str(labels.get(str(item.get("id", "") or "").strip(), "") or ""),
            rules,
        ),
        row_builder=lambda item, review: {
            "C1_complete_thinking": review.get("C1_complete_thinking", ""),
            "C2_context_coherence": review.get("C2_context_coherence", ""),
            "C3_response_alignment": review.get("C3_response_alignment", ""),
            "C4_framework_adherence": review.get("C4_framework_adherence", ""),
            "B1_diagnostic_plausibility": review.get("B1_diagnostic_plausibility", ""),
            "B2_safety_recognition": review.get("B2_safety_recognition", ""),
            "B3_therapeutic_safety": review.get("B3_therapeutic_safety", ""),
            "risk_markers_present": _join_pipe(review.get("risk_markers_present", [])),
            "risk_markers_addressed": _join_pipe(review.get("risk_markers_addressed", [])),
            "gold_diagnosis_label": str(labels.get(str(item.get("id", "") or "").strip(), "") or ""),
            "source_openr1_ids": _join_pipe(
                (item.get("metadata", {}) or {}).get("source_openr1_ids", [])
            ),
        },
    )

    study_b_single_run = _score_and_write(
        study="study_b_single",
        items=study_b_single_items,
        output_path=study_b_single_path,
        fieldnames=study_b_single_cols,
        review_scope="study_b_mapped",
        scorer=lambda item: score_study_b_single(
            item,
            id_unique=study_b_id_counts[str(item.get("id", "") or "").strip()] == 1,
        ),
        row_builder=lambda _item, review: {
            "prompt_nonempty": review.get("prompt_nonempty", ""),
            "gold_answer_nonempty": review.get("gold_answer_nonempty", ""),
            "incorrect_opinion_nonempty": review.get("incorrect_opinion_nonempty", ""),
            "persona_id_present": review.get("persona_id_present", ""),
            "id_unique": review.get("id_unique", ""),
            "mapped_contract_pass": review.get("mapped_contract_pass", ""),
        },
    )

    study_b_multi_run = _score_and_write(
        study="study_b_multi",
        items=study_b_multi_items,
        output_path=study_b_multi_path,
        fieldnames=study_b_multi_cols,
        review_scope="study_b_mapped",
        scorer=score_study_b_multi,
        row_builder=lambda _item, review: {
            "gold_answer_nonempty": review.get("gold_answer_nonempty", ""),
            "incorrect_opinion_nonempty": review.get("incorrect_opinion_nonempty", ""),
            "pressure_style_nonempty": review.get("pressure_style_nonempty", ""),
            "pressure_schedule_nonempty": review.get("pressure_schedule_nonempty", ""),
            "turns_nonempty": review.get("turns_nonempty", ""),
            "turns_all_have_message": review.get("turns_all_have_message", ""),
            "turns_all_pressure_level_valid": review.get("turns_all_pressure_level_valid", ""),
            "persona_id_present": review.get("persona_id_present", ""),
            "mapped_contract_pass": review.get("mapped_contract_pass", ""),
        },
    )

    study_c_run = _score_and_write(
        study="study_c",
        items=study_c_items,
        output_path=study_c_path_out,
        fieldnames=study_c_cols,
        review_scope="study_c_mapped",
        scorer=score_study_c,
        row_builder=lambda _item, review: {
            "patient_summary_nonempty": review.get("patient_summary_nonempty", ""),
            "critical_entities_nonempty": review.get("critical_entities_nonempty", ""),
            "turns_nonempty": review.get("turns_nonempty", ""),
            "persona_id_present": review.get("persona_id_present", ""),
            "source_openr1_ids_present": review.get("source_openr1_ids_present", ""),
            "num_turns_matches_turns_length": review.get("num_turns_matches_turns_length", ""),
            "mapped_contract_pass": review.get("mapped_contract_pass", ""),
        },
    )

    replacement_stats = build_replacement_candidates_study_a(
        study_a_samples=study_a_items,
        study_a_rows=study_a_run["scored_rows"],
        labels=labels,
        rules=rules,
        output_path=replacement_path,
        review_timestamp=now_iso(),
    )

    study_summaries = {
        "study_a": _summarise_study_file(study_a_path, EXPECTED_ROWS["study_a"]),
        "study_b_single": _summarise_study_file(study_b_single_path, EXPECTED_ROWS["study_b_single"]),
        "study_b_multi": _summarise_study_file(study_b_multi_path, EXPECTED_ROWS["study_b_multi"]),
        "study_c": _summarise_study_file(study_c_path_out, EXPECTED_ROWS["study_c"]),
    }

    _assert_summary_integrity(study_summaries)

    _write_review_summary(summary_path, study_summaries, replacement_stats)
    _write_metadata(metadata_path, rules_path, rules, study_summaries)

    for study_name, path in {
        "study_a": study_a_path,
        "study_b_single": study_b_single_path,
        "study_b_multi": study_b_multi_path,
        "study_c": study_c_path_out,
    }.items():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            verdicts = {str(row.get("verdict", "") or "").strip() for row in reader}
        if not verdicts.issubset(VALID_VERDICTS):
            raise RuntimeError(f"Invalid verdicts detected in {study_name}: {sorted(verdicts - VALID_VERDICTS)}")

    _ = (study_b_single_run, study_b_multi_run, study_c_run)  # Retained for explicit run completeness.
    _print_verdict_table(study_summaries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
