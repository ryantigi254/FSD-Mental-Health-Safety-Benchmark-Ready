#!/usr/bin/env python3
"""Run deterministic controllability cross-study review for the large resolved suite."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"


def _supports_modules(python_bin: str, modules: list[str]) -> bool:
    probe = "import " + ", ".join(modules)
    proc = subprocess.run([python_bin, "-c", probe], capture_output=True, text=True)
    return proc.returncode == 0


def _find_compatible_python() -> str | None:
    candidates = [
        sys.executable,
        shutil.which("python"),
        shutil.which("python3"),
        "/opt/homebrew/Caskroom/miniforge/base/bin/python3",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists() and _supports_modules(candidate, ["datasets"]):
            return candidate
    return None


_COMPATIBLE_PYTHON = _find_compatible_python()
if _COMPATIBLE_PYTHON is None:
    raise RuntimeError("No compatible Python interpreter with 'datasets' available for review.")
if Path(sys.executable).resolve() != Path(_COMPATIBLE_PYTHON).resolve():
    os.execv(_COMPATIBLE_PYTHON, [_COMPATIBLE_PYTHON, __file__, *sys.argv[1:]])


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_review = _load_module(
    "ctrl_v4_reference_review",
    SRC_DIR / "reliable_clinical_benchmark" / "review" / "v4_reference_review.py",
)
_readiness = _load_module(
    "ctrl_controllability_clinical_readiness",
    SRC_DIR
    / "reliable_clinical_benchmark"
    / "review"
    / "controllability_clinical_readiness.py",
)

load_rules = _review.load_rules
now_iso = _review.now_iso
read_existing_ssv_state = _review.read_existing_ssv_state
score_study_a = _review.score_study_a
score_study_a_bias = _review.score_study_a_bias
score_study_b_multi = _review.score_study_b_multi
score_study_b_single = _review.score_study_b_single
score_study_c = _review.score_study_c
write_ssv_row = _review.write_ssv_row

DEFAULT_CTRL_DIR = _readiness.DEFAULT_CTRL_DIR
DEFAULT_RULES_PATH = _readiness.DEFAULT_RULES_PATH
DEFAULT_VERIFICATION_DIR = _readiness.DEFAULT_VERIFICATION_DIR
EXPECTED_CTRL_COUNTS = _readiness.EXPECTED_CTRL_COUNTS


DEFAULT_OUT_DIR = DEFAULT_VERIFICATION_DIR

VALID_VERDICTS = {"ACCEPTABLE", "NEEDS_REVIEW", "REJECT"}
DEFAULT_EXPECTED_ROWS = dict(EXPECTED_CTRL_COUNTS)


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
        "study_a_bias_verdicts.ssv",
        "study_b_single_verdicts.ssv",
        "study_b_multi_verdicts.ssv",
        "study_c_verdicts.ssv",
        "review_summary.ssv",
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

    for row_number, item in enumerate(items, start=1):
        item_id = str(item.get("id", "") or "").strip()
        review = scorer(item)
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

    invalid_verdicts = sorted(set(verdicts) - VALID_VERDICTS)
    if invalid_verdicts:
        raise RuntimeError(f"Invalid verdicts in {path.name}: {invalid_verdicts}")

    return {
        "actual_rows": len(rows),
        "expected_rows": expected_rows,
        "acceptable": int(verdicts.get("ACCEPTABLE", 0)),
        "needs_review": int(verdicts.get("NEEDS_REVIEW", 0)),
        "reject": int(verdicts.get("REJECT", 0)),
        "counts_match": int(len(rows) == expected_rows),
        "duplicate_item_ids": duplicate_item_ids,
    }


def _write_review_summary(summary_path: Path, study_summaries: dict[str, dict[str, Any]]) -> None:
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
            "study": "study_a_bias",
            "expected_rows": study_summaries["study_a_bias"]["expected_rows"],
            "actual_rows": study_summaries["study_a_bias"]["actual_rows"],
            "acceptable": study_summaries["study_a_bias"]["acceptable"],
            "needs_review": study_summaries["study_a_bias"]["needs_review"],
            "reject": study_summaries["study_a_bias"]["reject"],
            "review_scope": "study_a_bias_mapped",
            "counts_match": study_summaries["study_a_bias"]["counts_match"],
            "duplicate_item_ids": study_summaries["study_a_bias"]["duplicate_item_ids"],
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

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_metadata(
    metadata_path: Path,
    *,
    ctrl_dir: Path,
    rules_path: Path,
    rules: dict[str, Any],
    study_summaries: dict[str, dict[str, Any]],
) -> None:
    payload = {
        "mode": "batch",
        "suite": "controllability_v0.1_large_resolved",
        "ctrl_dir": str(ctrl_dir.resolve()),
        "rule_file": str(rules_path.resolve()),
        "rule_version": str(rules.get("rule_version", "")),
        "matching_mode": str(rules.get("matching_mode", "")),
        "last_run_utc": now_iso(),
        "summary": study_summaries,
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
    for study in ["study_a", "study_a_bias", "study_b_single", "study_b_multi", "study_c"]:
        summary = study_summaries[study]
        print(
            f"{study} | {summary['acceptable']} | {summary['needs_review']} | {summary['reject']} "
            f"| {summary['actual_rows']} | {summary['expected_rows']}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic controllability cross-study review.")
    parser.add_argument("--ctrl-dir", type=Path, default=DEFAULT_CTRL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH)
    parser.add_argument("--clean", action="store_true", help="Clear existing outputs before scoring")
    parser.add_argument("--expect-study-a", type=int, default=DEFAULT_EXPECTED_ROWS["study_a"])
    parser.add_argument("--expect-study-a-bias", type=int, default=DEFAULT_EXPECTED_ROWS["study_a_bias"])
    parser.add_argument("--expect-study-b-single", type=int, default=DEFAULT_EXPECTED_ROWS["study_b_single"])
    parser.add_argument("--expect-study-b-multi", type=int, default=DEFAULT_EXPECTED_ROWS["study_b_multi"])
    parser.add_argument("--expect-study-c", type=int, default=DEFAULT_EXPECTED_ROWS["study_c"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctrl_dir = args.ctrl_dir
    out_dir = args.out_dir
    rules_path = args.rules

    expected_rows = {
        "study_a": args.expect_study_a,
        "study_a_bias": args.expect_study_a_bias,
        "study_b_single": args.expect_study_b_single,
        "study_b_multi": args.expect_study_b_multi,
        "study_c": args.expect_study_c,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        _clear_outputs(out_dir)

    rules = load_rules(rules_path)
    labels_payload = _load_json(ctrl_dir / "ctrl_gold_diagnosis_labels.json")
    labels = labels_payload.get("labels", {})
    if not isinstance(labels, dict):
        raise RuntimeError("ctrl_gold_diagnosis_labels.json must include a 'labels' object")

    study_a_items = _normalise_items(
        _load_json(ctrl_dir / "study_a_controllability_test.json"),
        ("samples", "cases", "items"),
    )
    study_a_bias_items = _normalise_items(
        _load_json(ctrl_dir / "study_a_bias_controllability_test.json"),
        ("cases", "samples", "items"),
    )
    study_b_single_items = _normalise_items(
        _load_json(ctrl_dir / "study_b_controllability_test.json"),
        ("samples", "items"),
    )
    study_b_multi_items = _normalise_items(
        _load_json(ctrl_dir / "study_b_multi_turn_controllability_test.json"),
        ("cases", "multi_turn_cases", "items"),
    )
    study_c_items = _normalise_items(
        _load_json(ctrl_dir / "study_c_controllability_test.json"),
        ("cases", "samples", "items"),
    )

    study_a_path = out_dir / "study_a_reference_verdicts.ssv"
    study_a_bias_path = out_dir / "study_a_bias_verdicts.ssv"
    study_b_single_path = out_dir / "study_b_single_verdicts.ssv"
    study_b_multi_path = out_dir / "study_b_multi_verdicts.ssv"
    study_c_path = out_dir / "study_c_verdicts.ssv"
    summary_path = out_dir / "review_summary.ssv"
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
    study_a_bias_cols = common_cols + [
        "prompt_nonempty",
        "bias_feature_nonempty",
        "bias_label_nonempty",
        "dimension_present",
        "dimension_family_present",
        "source_openr1_id_present",
        "inferred_condition_present",
        "condition_resolution_source_present",
        "id_unique",
        "mapped_contract_pass",
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

    study_a_bias_id_counts = Counter(str(item.get("id", "") or "").strip() for item in study_a_bias_items)
    study_b_id_counts = Counter(str(item.get("id", "") or "").strip() for item in study_b_single_items)

    _score_and_write(
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
            "source_openr1_ids": _join_pipe((item.get("metadata", {}) or {}).get("source_openr1_ids", [])),
        },
    )

    _score_and_write(
        study="study_a_bias",
        items=study_a_bias_items,
        output_path=study_a_bias_path,
        fieldnames=study_a_bias_cols,
        review_scope="study_a_bias_mapped",
        scorer=lambda item: score_study_a_bias(
            item,
            id_unique=study_a_bias_id_counts[str(item.get("id", "") or "").strip()] == 1,
        ),
        row_builder=lambda _item, review: {
            "prompt_nonempty": review.get("prompt_nonempty", ""),
            "bias_feature_nonempty": review.get("bias_feature_nonempty", ""),
            "bias_label_nonempty": review.get("bias_label_nonempty", ""),
            "dimension_present": review.get("dimension_present", ""),
            "dimension_family_present": review.get("dimension_family_present", ""),
            "source_openr1_id_present": review.get("source_openr1_id_present", ""),
            "inferred_condition_present": review.get("inferred_condition_present", ""),
            "condition_resolution_source_present": review.get("condition_resolution_source_present", ""),
            "id_unique": review.get("id_unique", ""),
            "mapped_contract_pass": review.get("mapped_contract_pass", ""),
        },
    )

    _score_and_write(
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

    _score_and_write(
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

    _score_and_write(
        study="study_c",
        items=study_c_items,
        output_path=study_c_path,
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

    study_summaries = {
        "study_a": _summarise_study_file(study_a_path, expected_rows["study_a"]),
        "study_a_bias": _summarise_study_file(study_a_bias_path, expected_rows["study_a_bias"]),
        "study_b_single": _summarise_study_file(study_b_single_path, expected_rows["study_b_single"]),
        "study_b_multi": _summarise_study_file(study_b_multi_path, expected_rows["study_b_multi"]),
        "study_c": _summarise_study_file(study_c_path, expected_rows["study_c"]),
    }

    _assert_summary_integrity(study_summaries)
    _write_review_summary(summary_path, study_summaries)
    _write_metadata(metadata_path, ctrl_dir=ctrl_dir, rules_path=rules_path, rules=rules, study_summaries=study_summaries)
    _print_verdict_table(study_summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
