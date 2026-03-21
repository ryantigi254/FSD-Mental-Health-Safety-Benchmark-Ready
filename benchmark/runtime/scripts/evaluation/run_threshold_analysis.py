#!/usr/bin/env python3
"""Run CI-aware threshold analysis over per-model metric rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

RUNTIME_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = RUNTIME_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reliable_clinical_benchmark.eval.threshold_analysis import (
    build_active_threshold_summary,
    build_threshold_analysis_table,
)


def _load_rows(input_path: Path) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            return payload["results"]
        if isinstance(payload.get("models"), list):
            return payload["models"]
    raise ValueError(f"Unsupported input JSON structure in {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        required=True,
        help="JSON file containing per-model metric rows (list or {'results': [...]})",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        default=[],
        help="Canonical metric name to analyse. Repeat for multiple metrics.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional output path. Defaults next to the input file as *_threshold_analysis.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    rows = _load_rows(input_path)
    metrics = args.metrics or [
        "faithfulness_gap",
        "step_f1",
        "silent_bias_rate",
        "sycophancy_probability",
        "flip_rate",
        "evidence_hallucination",
        "turn_of_flip",
        "entity_recall_at_t10",
        "knowledge_conflict_rate",
        "session_goal_alignment",
        "drift_slope",
    ]

    payload = {
        "input_json": str(input_path),
        "metrics": metrics,
        "threshold_analysis": build_threshold_analysis_table(rows, metrics),
        "active_threshold_summary": build_active_threshold_summary(rows, metrics),
    }

    output_path = (
        Path(args.output_json)
        if args.output_json
        else input_path.with_name(f"{input_path.stem}_threshold_analysis.json")
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote threshold analysis to {output_path}")


if __name__ == "__main__":
    main()
