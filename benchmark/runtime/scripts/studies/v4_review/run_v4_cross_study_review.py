#!/usr/bin/env python3
"""Run deterministic v4 cross-study reference review and emit per-case SSV files."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from reliable_clinical_benchmark.review import (
    build_replacement_candidates_study_a,
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
OUT_ROOT = DATA_ROOT / "verification" / "v4"
RULES_PATH = OUT_ROOT / "rubric_rules_v1.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return "unknown"


def _normalise_items(payload: Any, *, preferred_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unable to normalise payload with keys={preferred_keys}")


def _join_pipe(values: list[Any]) -> str:
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    return "|".join(cleaned)


def _write_study_rows(
    *,
    study: str,
    items: list[dict[str, Any]],
    output_path: Path,
    fieldnames: list[str],
    review_scope: str,
    scorer: Callable[[dict[str, Any]], dict[str, Any]],
    row_builder: Callable[[int, dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    existing_ids, existing_rows = read_existing_ssv_state(output_path)
    resumed = existing_rows > 0

    scored_rows: list[dict[str, Any]] = []
    written = 0

    sorted_items = sorted(items, key=lambda x: str(x.get("id", "")))
    for row_number, item in enumerate(sorted_items, start=1):
        item_id = str(item.get("id", "")).strip()
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
        row.update(row_builder(row_number, item, review))

        if item_id in existing_ids:
            continue

        write_ssv_row(output_path, fieldnames, row)
        existing_ids.add(item_id)
        written += 1

    return {
        "study": study,
        "total_items": len(sorted_items),
        "rows_written": written,
        "rows_preexisting": existing_rows,
        "resumed": resumed,
        "scored_rows": scored_rows,
    }


def _summarise_study_file(path: Path, expected_rows: int) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing SSV output: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = list(reader)

    counts = Counter(str(row.get("verdict", "")).strip() for row in rows)
    ids = [str(row.get("item_id", "")).strip() for row in rows]
    duplicate_count = len(ids) - len(set(ids))

    return {
        "actual_rows": len(rows),
        "expected_rows": expected_rows,
        "counts_match": len(rows) == expected_rows,
        "ACCEPTABLE": int(counts.get("ACCEPTABLE", 0)),
        "NEEDS_REVIEW": int(counts.get("NEEDS_REVIEW", 0)),
        "REJECT": int(counts.get("REJECT", 0)),
        "duplicate_item_ids": duplicate_count,
    }


def _write_summary(
    summary_path: Path,
    study_summaries: dict[str, dict[str, Any]],
    replacement_stats: dict[str, int],
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
            "acceptable": study_summaries["study_a"]["ACCEPTABLE"],
            "needs_review": study_summaries["study_a"]["NEEDS_REVIEW"],
            "reject": study_summaries["study_a"]["REJECT"],
            "review_scope": "study_a_full_rubric",
            "counts_match": int(study_summaries["study_a"]["counts_match"]),
            "duplicate_item_ids": study_summaries["study_a"]["duplicate_item_ids"],
            "notes": "",
        },
        {
            "study": "study_b_single",
            "expected_rows": study_summaries["study_b_single"]["expected_rows"],
            "actual_rows": study_summaries["study_b_single"]["actual_rows"],
            "acceptable": study_summaries["study_b_single"]["ACCEPTABLE"],
            "needs_review": study_summaries["study_b_single"]["NEEDS_REVIEW"],
            "reject": study_summaries["study_b_single"]["REJECT"],
            "review_scope": "study_b_mapped",
            "counts_match": int(study_summaries["study_b_single"]["counts_match"]),
            "duplicate_item_ids": study_summaries["study_b_single"]["duplicate_item_ids"],
            "notes": "",
        },
        {
            "study": "study_b_multi",
            "expected_rows": study_summaries["study_b_multi"]["expected_rows"],
            "actual_rows": study_summaries["study_b_multi"]["actual_rows"],
            "acceptable": study_summaries["study_b_multi"]["ACCEPTABLE"],
            "needs_review": study_summaries["study_b_multi"]["NEEDS_REVIEW"],
            "reject": study_summaries["study_b_multi"]["REJECT"],
            "review_scope": "study_b_mapped",
            "counts_match": int(study_summaries["study_b_multi"]["counts_match"]),
            "duplicate_item_ids": study_summaries["study_b_multi"]["duplicate_item_ids"],
            "notes": "",
        },
        {
            "study": "study_c",
            "expected_rows": study_summaries["study_c"]["expected_rows"],
            "actual_rows": study_summaries["study_c"]["actual_rows"],
            "acceptable": study_summaries["study_c"]["ACCEPTABLE"],
            "needs_review": study_summaries["study_c"]["NEEDS_REVIEW"],
            "reject": study_summaries["study_c"]["REJECT"],
            "review_scope": "study_c_mapped",
            "counts_match": int(study_summaries["study_c"]["counts_match"]),
            "duplicate_item_ids": study_summaries["study_c"]["duplicate_item_ids"],
            "notes": "",
        },
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
            "notes": (
                f"candidate_pool_size={replacement_stats.get('candidate_pool_size', 0)}"
            ),
        },
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v4 cross-study deterministic review.")
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_started = now_iso()

    rules = load_rules(args.rules)
    labels_payload = _load_json(GOLD_ROOT / "gold_diagnosis_labels.json")
    labels = labels_payload.get("labels", {})
    if not isinstance(labels, dict):
        raise RuntimeError("Invalid labels payload in gold_diagnosis_labels.json")

    study_a_payload = _load_json(SPLITS_ROOT / "study_a_test.json")
    study_b_single_payload = _load_json(SPLITS_ROOT / "study_b_test.json")
    study_b_multi_payload = _load_json(SPLITS_ROOT / "study_b_multi_turn_test.json")
    study_c_payload = _load_json(SPLITS_ROOT / "study_c_test.json")

    study_a_items = _normalise_items(study_a_payload, preferred_keys=("samples", "items", "cases"))
    study_b_single_items = _normalise_items(study_b_single_payload, preferred_keys=("samples", "items"))
    study_b_multi_items = _normalise_items(study_b_multi_payload, preferred_keys=("multi_turn_cases", "cases", "items"))
    study_c_items = _normalise_items(study_c_payload, preferred_keys=("cases", "samples", "items"))

    study_b_id_counts = Counter(str(item.get("id", "")).strip() for item in study_b_single_items)

    study_a_path = out_dir / "study_a_reference_verdicts.ssv"
    study_b_single_path = out_dir / "study_b_single_verdicts.ssv"
    study_b_multi_path = out_dir / "study_b_multi_verdicts.ssv"
    study_c_path = out_dir / "study_c_verdicts.ssv"
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

    def _study_a_scorer(item: dict[str, Any]) -> dict[str, Any]:
        item_id = str(item.get("id", "")).strip()
        return score_study_a_case(item=item, gold_label=str(labels.get(item_id, "")), rules=rules)

    study_a_run = _write_study_rows(
        study="study_a",
        items=study_a_items,
        output_path=study_a_path,
        fieldnames=study_a_cols,
        review_scope="study_a_full_rubric",
        scorer=_study_a_scorer,
        row_builder=lambda _n, item, review: {
            "C1_complete_thinking": review.get("C1_complete_thinking", ""),
            "C2_context_coherence": review.get("C2_context_coherence", ""),
            "C3_response_alignment": review.get("C3_response_alignment", ""),
            "C4_framework_adherence": review.get("C4_framework_adherence", ""),
            "B1_diagnostic_plausibility": review.get("B1_diagnostic_plausibility", ""),
            "B2_safety_recognition": review.get("B2_safety_recognition", ""),
            "B3_therapeutic_safety": review.get("B3_therapeutic_safety", ""),
            "risk_markers_present": _join_pipe(review.get("risk_markers_present", [])),
            "risk_markers_addressed": _join_pipe(review.get("risk_markers_addressed", [])),
            "gold_diagnosis_label": str(labels.get(str(item.get("id", "")), "")),
            "source_openr1_ids": _join_pipe((item.get("metadata", {}) or {}).get("source_openr1_ids", [])),
        },
    )

    study_b_single_run = _write_study_rows(
        study="study_b_single",
        items=study_b_single_items,
        output_path=study_b_single_path,
        fieldnames=study_b_single_cols,
        review_scope="study_b_mapped",
        scorer=lambda item: score_study_b_single_case(
            item=item,
            id_unique=study_b_id_counts.get(str(item.get("id", "")).strip(), 0) == 1,
        ),
        row_builder=lambda _n, _item, review: {
            "prompt_nonempty": review.get("prompt_nonempty", ""),
            "gold_answer_nonempty": review.get("gold_answer_nonempty", ""),
            "incorrect_opinion_nonempty": review.get("incorrect_opinion_nonempty", ""),
            "persona_id_present": review.get("persona_id_present", ""),
            "id_unique": review.get("id_unique", ""),
            "mapped_contract_pass": review.get("mapped_contract_pass", ""),
        },
    )

    study_b_multi_run = _write_study_rows(
        study="study_b_multi",
        items=study_b_multi_items,
        output_path=study_b_multi_path,
        fieldnames=study_b_multi_cols,
        review_scope="study_b_mapped",
        scorer=score_study_b_multi_case,
        row_builder=lambda _n, _item, review: {
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

    study_c_run = _write_study_rows(
        study="study_c",
        items=study_c_items,
        output_path=study_c_path,
        fieldnames=study_c_cols,
        review_scope="study_c_mapped",
        scorer=score_study_c_case,
        row_builder=lambda _n, _item, review: {
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

    if not replacement_path.exists():
        replacement_cols = [
            "study",
            "replacing_item_id",
            "original_diagnosis_label",
            "candidate_rank",
            "candidate_split",
            "candidate_openr1_id",
            "candidate_prompt_preview",
            "candidate_verdict",
            "candidate_reason_codes",
            "accepted",
            "review_timestamp_utc",
        ]
        with replacement_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=replacement_cols, delimiter=";")
            writer.writeheader()

    study_summaries = {
        "study_a": _summarise_study_file(study_a_path, expected_rows=2000),
        "study_b_single": _summarise_study_file(study_b_single_path, expected_rows=2000),
        "study_b_multi": _summarise_study_file(study_b_multi_path, expected_rows=120),
        "study_c": _summarise_study_file(study_c_path, expected_rows=100),
    }

    _write_summary(summary_path, study_summaries, replacement_stats)

    failing = [
        name
        for name, summary in study_summaries.items()
        if (not summary["counts_match"]) or summary["duplicate_item_ids"] > 0
    ]
    if failing:
        raise RuntimeError(f"SSV contract failed for studies: {', '.join(failing)}")

    metadata = {
        "run_started_utc": run_started,
        "run_finished_utc": now_iso(),
        "git_commit": _git_commit(),
        "rule_file": str(args.rules),
        "rule_version": str(rules.get("rule_version", "unknown")),
        "interrupted_resumed": any(
            run.get("resumed", False)
            for run in (study_a_run, study_b_single_run, study_b_multi_run, study_c_run)
        ),
        "runs": {
            "study_a": {
                "total_items": study_a_run["total_items"],
                "rows_written": study_a_run["rows_written"],
                "rows_preexisting": study_a_run["rows_preexisting"],
            },
            "study_b_single": {
                "total_items": study_b_single_run["total_items"],
                "rows_written": study_b_single_run["rows_written"],
                "rows_preexisting": study_b_single_run["rows_preexisting"],
            },
            "study_b_multi": {
                "total_items": study_b_multi_run["total_items"],
                "rows_written": study_b_multi_run["rows_written"],
                "rows_preexisting": study_b_multi_run["rows_preexisting"],
            },
            "study_c": {
                "total_items": study_c_run["total_items"],
                "rows_written": study_c_run["rows_written"],
                "rows_preexisting": study_c_run["rows_preexisting"],
            },
        },
        "replacement_plan": replacement_stats,
        "summary": study_summaries,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print("v4 review complete")
    for study_name, summary in study_summaries.items():
        print(
            f"{study_name}: rows={summary['actual_rows']} "
            f"acceptable={summary['ACCEPTABLE']} needs_review={summary['NEEDS_REVIEW']} reject={summary['REJECT']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
