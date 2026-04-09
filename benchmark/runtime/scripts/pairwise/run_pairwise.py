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

_RUNTIME_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_RUNTIME_ROOT / "src"))

from reliable_clinical_benchmark.pairwise.aggregator import PairwiseAggregator  # noqa: E402
from reliable_clinical_benchmark.pairwise.config import load_pairwise_run_spec  # noqa: E402
from reliable_clinical_benchmark.pairwise.report import PairwiseReportBuilder  # noqa: E402
from reliable_clinical_benchmark.pairwise.runner import PairwiseRunner  # noqa: E402
from reliable_clinical_benchmark.utils.logging_config import setup_logging  # noqa: E402


logger = logging.getLogger(__name__)


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

    runner = PairwiseRunner(run_spec=run_spec, case_manifest=case_manifest, api_base=args.api_base)
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
                    "judges": [
                        {
                            "judge_id": judge.judge_id,
                            "role": judge.role or "panel",
                        }
                        for judge in run_spec.judge_manifest.judges
                    ],
                    "criteria": run_spec.criteria,
                    "high_risk_tags": run_spec.config.high_risk_tags,
                    "output_root": run_spec.config.output_root,
                },
                indent=2,
            )
        )
        return 0

    raw_records, parsed_records = runner.run_all()
    aggregate_blob = PairwiseAggregator(run_spec=run_spec, case_manifest=case_manifest).aggregate(parsed_records)
    written = PairwiseReportBuilder(run_spec=run_spec).build(aggregate_blob)

    print(
        json.dumps(
            {
                "run_mode": run_spec.config.run_mode,
                "planned_calls_min": planned_bounds["min"],
                "planned_calls_max": planned_bounds["max"],
                "raw_records": len(raw_records),
                "parsed_records": len(parsed_records),
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
    parser.add_argument("--dry-run", action="store_true", help="Show the planned run without judge calls.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logging.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
