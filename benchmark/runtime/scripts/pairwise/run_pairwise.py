#!/usr/bin/env python3
"""
Run one pairwise secondary-evaluation slice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

_RUNTIME_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_RUNTIME_ROOT / "src"))

from reliable_clinical_benchmark.pairwise.aggregator import PairwiseAggregator  # noqa: E402
from reliable_clinical_benchmark.pairwise.config import load_pairwise_run_spec  # noqa: E402
from reliable_clinical_benchmark.pairwise.report import PairwiseReportBuilder  # noqa: E402
from reliable_clinical_benchmark.pairwise.runner import PairwiseRunner  # noqa: E402
from reliable_clinical_benchmark.utils.logging_config import setup_logging  # noqa: E402


logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_existing_parsed_records(parsed_dir: Path, *, slice_id: str) -> list[Dict[str, Any]]:
    deduped: dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for path in sorted(parsed_dir.glob(f"*__{slice_id}.jsonl")):
        for record in _iter_jsonl(path):
            key = (
                str(record.get("comparison_key", "")),
                str(record.get("judge_id", "")),
                str(record.get("order", "")),
            )
            deduped[key] = record
    return list(deduped.values())


def _dedupe_records(records: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    deduped: dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("comparison_key", "")),
            str(record.get("judge_id", "")),
            str(record.get("order", "")),
        )
        deduped[key] = record
    return list(deduped.values())


def _selected_judge_plan(
    runner: PairwiseRunner,
    *,
    selected_judges: list[str],
    existing_parsed_records: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    plan: list[Dict[str, Any]] = []
    simulated_records = list(existing_parsed_records)
    active_by_id = {judge.judge_id: judge for judge in runner.active_judges()}
    for judge_id in selected_judges:
        judge = active_by_id[judge_id]
        pending_groups = runner.pending_groups_for_judge(
            judge=judge,
            existing_parsed_records=simulated_records,
        )
        plan.append(
            {
                "judge_id": judge.judge_id,
                "role": judge.role or "panel",
                "local_model_id": judge.local_model_id,
                "pending_comparisons": len(pending_groups),
                "planned_calls": len(pending_groups) * len(runner.run_spec.config.orders),
            }
        )
        for comparison in pending_groups:
            for order in runner.run_spec.config.orders:
                simulated_records.append(
                    {
                        "slice_id": runner.run_spec.config.slice_id,
                        "comparison_key": comparison["comparison_key"],
                        "judge_id": judge.judge_id,
                        "order": order,
                    }
                )
    return plan


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging("INFO" if args.verbose else "WARNING")

    run_spec = load_pairwise_run_spec(args.config)
    case_manifest = json.loads(Path(run_spec.config.case_manifest_path).read_text(encoding="utf-8"))

    if case_manifest.get("status") != "ready":
        logger.warning(
            "Case manifest %s is not ready (status=%s). Proceeding with zero-case run.",
            run_spec.config.case_manifest_path,
            case_manifest.get("status"),
        )

    selected_judges = [judge_id for judge_id in args.judge_id if judge_id.strip()]
    known_judges = {judge.judge_id for judge in run_spec.judge_manifest.judges}
    unknown_judges = sorted(set(selected_judges) - known_judges)
    if unknown_judges:
        raise SystemExit(f"Unknown judge_id values: {', '.join(unknown_judges)}")

    runner = PairwiseRunner(
        run_spec=run_spec,
        case_manifest=case_manifest,
        api_base=args.api_base,
        selected_judge_ids=selected_judges or None,
    )
    existing_parsed_records = _load_existing_parsed_records(
        runner.parsed_dir,
        slice_id=run_spec.config.slice_id,
    )
    judge_plan = (
        _selected_judge_plan(
            runner,
            selected_judges=selected_judges,
            existing_parsed_records=existing_parsed_records,
        )
        if selected_judges and run_spec.config.run_mode == "stacked"
        else []
    )
    if judge_plan:
        planned_total = sum(item["planned_calls"] for item in judge_plan)
        planned_bounds = {"min": planned_total, "max": planned_total}
    else:
        planned_bounds = runner.planned_call_bounds()

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": run_spec.config.run_id,
                    "layer": run_spec.config.layer,
                    "slice_id": run_spec.config.slice_id,
                    "run_mode": run_spec.config.run_mode,
                    "planned_calls_min": planned_bounds["min"],
                    "planned_calls_max": planned_bounds["max"],
                    "case_count": len(case_manifest.get("cases", [])),
                    "selected_judges": selected_judges or "all",
                    "judge_plan": judge_plan,
                    "existing_parsed_records": len(existing_parsed_records),
                    "judges": [
                        {
                            "judge_id": judge.judge_id,
                            "role": judge.role or "panel",
                            "local_model_id": judge.local_model_id,
                        }
                        for judge in runner.active_judges()
                    ],
                    "criteria": run_spec.criteria,
                    "high_risk_tags": run_spec.config.high_risk_tags,
                    "output_root": run_spec.config.output_root,
                },
                indent=2,
            )
        )
        return 0

    if selected_judges and run_spec.config.run_mode == "stacked":
        raw_records = []
        parsed_records = []
        all_parsed_records = list(existing_parsed_records)
        active_by_id = {judge.judge_id: judge for judge in runner.active_judges()}
        for judge_id in selected_judges:
            judge = active_by_id[judge_id]
            pending_groups = runner.pending_groups_for_judge(
                judge=judge,
                existing_parsed_records=all_parsed_records,
            )
            judge_raw, judge_parsed = runner.run_groups_for_judge(
                comparison_groups=pending_groups,
                judge=judge,
            )
            raw_records.extend(judge_raw)
            parsed_records.extend(judge_parsed)
            all_parsed_records = _dedupe_records([*all_parsed_records, *judge_parsed])
    else:
        raw_records, parsed_records = runner.run_all()
        all_parsed_records = _load_existing_parsed_records(
            runner.parsed_dir,
            slice_id=run_spec.config.slice_id,
        )
    aggregate_blob = PairwiseAggregator(run_spec=run_spec, case_manifest=case_manifest).aggregate(all_parsed_records)
    written = PairwiseReportBuilder(run_spec=run_spec).build(aggregate_blob)

    print(
        json.dumps(
            {
                "run_mode": run_spec.config.run_mode,
                "selected_judges": selected_judges or "all",
                "planned_calls_min": planned_bounds["min"],
                "planned_calls_max": planned_bounds["max"],
                "executed_raw_records": len(raw_records),
                "executed_parsed_records": len(parsed_records),
                "loaded_parsed_records": len(all_parsed_records),
                "aggregate_path": str(written["aggregate_path"]),
                "report_path": str(written["report_path"]),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pairwise secondary evaluation for one slice.")
    parser.add_argument("--config", required=True, help="Path to the pairwise run config JSON.")
    parser.add_argument("--api-base", default="http://localhost:1234/v1", help="LM Studio API base URL.")
    parser.add_argument(
        "--judge-id",
        action="append",
        default=[],
        help="Optional judge_id to run. Repeat the flag to queue multiple judges while keeping one LM Studio model loaded at a time.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show the planned run without judge calls.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logging.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
