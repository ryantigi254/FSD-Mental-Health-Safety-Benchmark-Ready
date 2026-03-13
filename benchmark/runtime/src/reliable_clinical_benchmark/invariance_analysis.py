"""Analysis helpers for invariance and controllability outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from reliable_clinical_benchmark.invariance import (
    compute_case_metrics,
    get_study_spec,
    load_manifest_records,
    study_metric_names,
)


@dataclass(frozen=True)
class VariantSpec:
    """Variant cache plus metadata used in controllability analysis."""

    tag: str
    cache_path: Path
    variant_type: str = "control"
    intensity: Optional[float] = None


def _aggregator(name: str) -> Callable[[Sequence[float]], float]:
    if name == "median":
        return lambda values: float(median(values)) if values else 0.0
    if name == "mean":
        return lambda values: float(mean(values)) if values else 0.0
    raise ValueError(f"Unsupported aggregation '{name}'")


def _bootstrap_ci(
    values: Sequence[float],
    *,
    aggregator: str,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    agg = _aggregator(aggregator)
    if not values:
        return 0.0, 0.0, 0.0

    values_np = np.array(list(values), dtype=float)
    point = agg(values_np.tolist())
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(n_resamples):
        idx = rng.randint(0, len(values_np), len(values_np))
        samples.append(agg(values_np[idx].tolist()))
    return float(point), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def flatten_invariance_results(payloads: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten one or more invariance comparison payloads into tabular rows."""

    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        study = str(payload.get("study", ""))
        base_cache = str(payload.get("base_cache", ""))
        variant_cache = str(payload.get("variant_cache", ""))
        for metric_name, metric in (payload.get("metrics", {}) or {}).items():
            rows.append(
                {
                    "study": study,
                    "metric": metric_name,
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


def scan_invariance_result_files(root: Path) -> List[Path]:
    """Find invariance comparison JSON files under a result root."""

    matches = []
    for path in sorted(root.rglob("*.json")):
        if "invariance" not in path.name or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict):
            matches.append(path)
    return matches


def summarize_invariance_result_files(root: Path) -> List[Dict[str, Any]]:
    """Load and flatten invariance comparison JSON files."""

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in scan_invariance_result_files(root)]
    return flatten_invariance_results(payloads)


def build_invariance_case_delta_rows(
    *,
    study: str,
    base_cache: Path,
    variant_cache: Path,
    data_root: Path,
    metrics: Optional[Sequence[str]] = None,
    use_nli: bool = False,
    nli_stride: int = 2,
) -> List[Dict[str, Any]]:
    """Build per-case base/variant/delta rows for invariance analysis notebooks."""

    spec = get_study_spec(study)
    metric_names = list(metrics or study_metric_names(spec.canonical_name))
    strata_records = {record.id: record for record in load_manifest_records(spec.canonical_name, data_root)}
    base_case_metrics = compute_case_metrics(
        study=spec.canonical_name,
        cache_path=base_cache,
        data_root=data_root,
        use_nli=use_nli,
        nli_stride=nli_stride,
    )
    variant_case_metrics = compute_case_metrics(
        study=spec.canonical_name,
        cache_path=variant_cache,
        data_root=data_root,
        use_nli=use_nli,
        nli_stride=nli_stride,
    )

    rows: List[Dict[str, Any]] = []
    for metric_name in metric_names:
        shared_ids = sorted(
            row_id
            for row_id in base_case_metrics
            if metric_name in base_case_metrics[row_id]
            and row_id in variant_case_metrics
            and metric_name in variant_case_metrics[row_id]
        )
        for row_id in shared_ids:
            record = strata_records.get(row_id)
            rows.append(
                {
                    "study": spec.canonical_name,
                    "pairing_unit": spec.pairing_unit,
                    "id": row_id,
                    "metric": metric_name,
                    "base": float(base_case_metrics[row_id][metric_name]),
                    "variant": float(variant_case_metrics[row_id][metric_name]),
                    "delta": float(variant_case_metrics[row_id][metric_name] - base_case_metrics[row_id][metric_name]),
                    "strata": record.strata if record else {},
                    "metadata": record.metadata if record else {},
                }
            )
    return rows


def run_controllability_comparison(
    *,
    study: str,
    base_cache: Path,
    variants: Sequence[VariantSpec],
    data_root: Path,
    metrics: Optional[Sequence[str]] = None,
    aggregation: str = "median",
    n_resamples: int = 1000,
    seed: int = 42,
    use_nli: bool = False,
    nli_stride: int = 2,
) -> Dict[str, Any]:
    """Compute controllability deltas from per-case study metrics."""

    spec = get_study_spec(study)
    metric_names = list(metrics or study_metric_names(spec.canonical_name))
    base_case_metrics = compute_case_metrics(
        study=spec.canonical_name,
        cache_path=base_cache,
        data_root=data_root,
        use_nli=use_nli,
        nli_stride=nli_stride,
    )

    variant_payloads: List[Dict[str, Any]] = []
    metric_variant_points: Dict[str, List[Dict[str, Any]]] = {metric_name: [] for metric_name in metric_names}

    for variant in variants:
        variant_case_metrics = compute_case_metrics(
            study=spec.canonical_name,
            cache_path=variant.cache_path,
            data_root=data_root,
            use_nli=use_nli,
            nli_stride=nli_stride,
        )
        variant_metrics: Dict[str, Any] = {}

        for metric_name in metric_names:
            shared_ids = sorted(
                row_id
                for row_id in base_case_metrics
                if metric_name in base_case_metrics[row_id]
                and row_id in variant_case_metrics
                and metric_name in variant_case_metrics[row_id]
            )
            per_case = []
            deltas = []
            base_values = []
            variant_values = []
            for row_id in shared_ids:
                base_value = float(base_case_metrics[row_id][metric_name])
                variant_value = float(variant_case_metrics[row_id][metric_name])
                delta = variant_value - base_value
                base_values.append(base_value)
                variant_values.append(variant_value)
                deltas.append(delta)
                per_case.append(
                    {
                        "id": row_id,
                        "base": base_value,
                        "variant": variant_value,
                        "delta_c": delta,
                    }
                )

            delta_point, delta_low, delta_high = _bootstrap_ci(
                deltas,
                aggregator=aggregation,
                n_resamples=n_resamples,
                seed=seed,
            )
            base_point, _, _ = _bootstrap_ci(
                base_values,
                aggregator=aggregation,
                n_resamples=n_resamples,
                seed=seed,
            )
            variant_point, _, _ = _bootstrap_ci(
                variant_values,
                aggregator=aggregation,
                n_resamples=n_resamples,
                seed=seed,
            )

            variant_metrics[metric_name] = {
                "aggregation": aggregation,
                "n_pairs": len(shared_ids),
                "base": base_point,
                "variant": variant_point,
                "delta_c": delta_point,
                "ci_low": delta_low,
                "ci_high": delta_high,
                "per_case": per_case,
            }

            if variant.intensity is not None and shared_ids:
                metric_variant_points[metric_name].append(
                    {
                        "tag": variant.tag,
                        "intensity": float(variant.intensity),
                        "delta_c": delta_point,
                    }
                )

        variant_payloads.append(
            {
                "tag": variant.tag,
                "variant_type": variant.variant_type,
                "intensity": variant.intensity,
                "cache_path": str(variant.cache_path),
                "metrics": variant_metrics,
            }
        )

    sensitivity: Dict[str, List[Dict[str, Any]]] = {}
    for metric_name, points in metric_variant_points.items():
        ordered = sorted(points, key=lambda point: (point["intensity"], point["tag"]))
        curves: List[Dict[str, Any]] = []
        for previous, current in zip(ordered, ordered[1:]):
            delta_intensity = current["intensity"] - previous["intensity"]
            if delta_intensity == 0:
                continue
            slope = (current["delta_c"] - previous["delta_c"]) / delta_intensity
            curves.append(
                {
                    "from_tag": previous["tag"],
                    "to_tag": current["tag"],
                    "from_intensity": previous["intensity"],
                    "to_intensity": current["intensity"],
                    "slope": slope,
                }
            )
        if curves:
            sensitivity[metric_name] = curves

    return {
        "study": spec.canonical_name,
        "pairing_unit": spec.pairing_unit,
        "base_cache": str(base_cache),
        "data_root": str(data_root),
        "aggregation": aggregation,
        "bootstrap_resamples": n_resamples,
        "seed": seed,
        "metrics": metric_names,
        "variants": variant_payloads,
        "sensitivity_curves": sensitivity,
    }
