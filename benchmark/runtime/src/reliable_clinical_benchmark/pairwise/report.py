"""
Report writers for pairwise evaluation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import PairwiseRunSpec


class PairwiseReportBuilder:
    """Persist aggregate outputs in machine-readable and CSV mirror form."""

    def __init__(self, *, run_spec: PairwiseRunSpec) -> None:
        self.run_spec = run_spec
        self.output_root = Path(run_spec.config.output_root)
        self.aggregate_dir = self.output_root / "aggregates" / run_spec.config.run_id
        self.report_dir = self.output_root / "reports" / run_spec.config.run_id
        self.aggregate_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def build(self, aggregate_blob: Dict[str, Any]) -> Dict[str, Path]:
        slice_id = self.run_spec.config.slice_id

        aggregate_path = self.aggregate_dir / f"{slice_id}__aggregates.json"
        report_path = self.report_dir / f"{slice_id}__pairwise_secondary_results.json"
        win_rates_path = self.report_dir / f"{slice_id}__win_rates.csv"
        bt_path = self.report_dir / f"{slice_id}__bradley_terry.csv"

        aggregate_path.write_text(json.dumps(aggregate_blob, indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(aggregate_blob, indent=2), encoding="utf-8")
        self._write_win_rates_csv(win_rates_path, aggregate_blob)
        self._write_bradley_terry_csv(bt_path, aggregate_blob)

        return {
            "aggregate_path": aggregate_path,
            "report_path": report_path,
            "win_rates_csv_path": win_rates_path,
            "bradley_terry_csv_path": bt_path,
        }

    def _write_win_rates_csv(self, path: Path, aggregate_blob: Dict[str, Any]) -> None:
        rows: List[Dict[str, Any]] = []
        for scope, payload in self._iter_scope_payloads(aggregate_blob):
            for row in payload.get("win_rates", []):
                rows.append({"scope": scope, **row})

        fieldnames = [
            "scope",
            "criterion_id",
            "system_a",
            "system_b",
            "wins_a",
            "wins_b",
            "ties",
            "invalid",
            "decisive_total",
            "total",
            "win_rate_a",
            "win_rate_b",
            "win_rate_a_ci_lower",
            "win_rate_a_ci_upper",
            "win_rate_b_ci_lower",
            "win_rate_b_ci_upper",
        ]
        self._write_csv(path, fieldnames, rows)

    def _write_bradley_terry_csv(self, path: Path, aggregate_blob: Dict[str, Any]) -> None:
        rows: List[Dict[str, Any]] = []
        for scope, payload in self._iter_scope_payloads(aggregate_blob):
            for row in payload.get("bradley_terry", []):
                rows.append({"scope": scope, **row})

        fieldnames = [
            "scope",
            "system_id",
            "score",
            "se",
            "ci_lower",
            "ci_upper",
            "decisive_total",
        ]
        self._write_csv(path, fieldnames, rows)

    @staticmethod
    def _iter_scope_payloads(aggregate_blob: Dict[str, Any]) -> Iterable[tuple[str, Dict[str, Any]]]:
        for judge_id, payload in aggregate_blob.get("per_judge", {}).items():
            yield judge_id, payload
        pooled = aggregate_blob.get("pooled")
        if isinstance(pooled, dict):
            yield "pooled", pooled

    @staticmethod
    def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
