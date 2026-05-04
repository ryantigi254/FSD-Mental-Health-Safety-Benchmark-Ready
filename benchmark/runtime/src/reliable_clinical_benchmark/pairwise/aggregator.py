"""
Aggregation layer for pairwise evaluation outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .config import PairwiseRunSpec
from .statistics import compute_slice_statistics


class PairwiseAggregator:
    """Aggregate parsed pairwise judgements into one canonical result blob."""

    def __init__(self, *, run_spec: PairwiseRunSpec, case_manifest: Dict[str, Any]) -> None:
        self.run_spec = run_spec
        self.case_manifest = case_manifest

    def aggregate(self, parsed_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return compute_slice_statistics(
            records=parsed_records,
            run_spec=self.run_spec,
            case_manifest=self.case_manifest,
        )
