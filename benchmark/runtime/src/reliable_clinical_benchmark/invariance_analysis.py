"""Analysis helpers for invariance and controllability outputs.

Ported from the metric-invariance branch and extended to support the
``metric-results/secondary_branch_metrics`` output structure produced by
``calculate_branch_secondary_metrics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class VariantSpec:
    """Variant cache plus metadata used in controllability analysis."""

    tag: str
    cache_path: Path
    variant_type: str = "control"
    intensity: Optional[float] = None


def flatten_invariance_results(
    payloads: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten one or more invariance comparison payloads into tabular rows.

    Supports both the legacy ``*_invariance*.json`` format and the new
    ``secondary_branch_metrics/{invariance,ctrl-invariance}/model/study.json``
    format.  Both formats share a ``metrics`` dict keyed by metric name, with
    sub-keys ``n_pairs``, ``base``, ``variant``, ``delta``, ``ci_low``,
    ``ci_high``.
    """

    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        study = str(payload.get("study", ""))
        base_cache = str(payload.get("base_cache", ""))
        variant_cache = str(payload.get("variant_cache", ""))
        model = str(payload.get("model", ""))
        lane = str(payload.get("lane", "invariance"))
        variant_tag = str(payload.get("variant_tag", ""))
        for metric_name, metric in (payload.get("metrics", {}) or {}).items():
            rows.append(
                {
                    "study": study,
                    "metric": metric_name,
                    "model": model,
                    "lane": lane,
                    "variant_tag": variant_tag,
                    "pairing_unit": payload.get("pairing_unit"),
                    "base_cache": base_cache,
                    "variant_cache": variant_cache,
                    "n_pairs": metric.get("n_pairs"),
                    "base": metric.get("base"),
                    "variant": metric.get("variant"),
                    "delta": metric.get("delta"),
                    "ci_low": metric.get("ci_low"),
                    "ci_high": metric.get("ci_high"),
                }
            )
    return rows


def _is_invariance_result(payload: Any) -> bool:
    """Return True if *payload* looks like a valid invariance/ctrl-inv result."""
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("metrics"), dict)
        and payload.get("status") == "ok"
    )


def scan_invariance_result_files(root: Path) -> List[Path]:
    """Find invariance comparison JSON files under *root*.

    Matches:
    - Legacy: any ``.json`` file whose **name** contains ``invariance``.
    - New: any ``.json`` file whose **parent directory name** is ``invariance``
      or ``ctrl-invariance`` (i.e. files in
      ``secondary_branch_metrics/{invariance,ctrl-invariance}/model/study.json``).
    """

    matches = []
    invariance_lane_dirs = {"invariance", "ctrl-invariance"}

    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue

        in_invariance_dir = (
            path.parent.parent.name in invariance_lane_dirs
            or path.parent.name in invariance_lane_dirs
        )
        name_has_invariance = "invariance" in path.name

        if not (in_invariance_dir or name_has_invariance):
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if _is_invariance_result(payload):
            matches.append(path)

    return matches


def summarize_invariance_result_files(root: Path) -> List[Dict[str, Any]]:
    """Load and flatten invariance comparison JSON files found under *root*."""

    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in scan_invariance_result_files(root)
    ]
    return flatten_invariance_results(payloads)
