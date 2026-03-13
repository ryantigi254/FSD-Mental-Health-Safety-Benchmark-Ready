#!/usr/bin/env python3
"""Compute controllability meta-metrics from base and control-variant caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = RUNTIME_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_clinical_benchmark.invariance import DEFAULT_V5_ROOT, study_cli_choices, study_metric_names
from reliable_clinical_benchmark.invariance_analysis import VariantSpec, run_controllability_comparison


def _parse_variant_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("Variant arguments must use tag=path format.")
    tag, path = raw.split("=", 1)
    tag = tag.strip()
    path = path.strip()
    if not tag or not path:
        raise ValueError("Variant arguments must use tag=path format.")
    return tag, Path(path).resolve()


def _parse_float_arg(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise ValueError("Intensity arguments must use tag=value format.")
    tag, value = raw.split("=", 1)
    return tag.strip(), float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controllability meta-metric comparisons.")
    parser.add_argument("--study", required=True, choices=list(study_cli_choices()))
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_V5_ROOT)
    parser.add_argument(
        "--variant-cache",
        action="append",
        required=True,
        help="Variant cache in tag=path format. Repeat for multiple variants.",
    )
    parser.add_argument(
        "--variant-intensity",
        action="append",
        default=[],
        help="Optional intensity metadata in tag=value format.",
    )
    parser.add_argument(
        "--variant-type",
        action="append",
        default=[],
        help="Optional variant type metadata in tag=value format.",
    )
    parser.add_argument("--metric", action="append", default=None, help="Optional metric name filter.")
    parser.add_argument("--aggregation", choices=["median", "mean"], default="median")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-nli", action="store_true")
    parser.add_argument("--nli-stride", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    intensity_by_tag = dict(_parse_float_arg(raw) for raw in args.variant_intensity)
    variant_type_by_tag = {}
    for raw in args.variant_type:
        if "=" not in raw:
            raise ValueError("Variant type arguments must use tag=value format.")
        tag, value = raw.split("=", 1)
        variant_type_by_tag[tag.strip()] = value.strip()

    variants = []
    for raw in args.variant_cache:
        tag, cache_path = _parse_variant_arg(raw)
        variants.append(
            VariantSpec(
                tag=tag,
                cache_path=cache_path,
                variant_type=variant_type_by_tag.get(tag, "control"),
                intensity=intensity_by_tag.get(tag),
            )
        )

    metrics = args.metric or list(study_metric_names(args.study))
    try:
        payload = run_controllability_comparison(
            study=args.study,
            base_cache=args.base_cache.resolve(),
            variants=variants,
            data_root=args.data_root.resolve(),
            metrics=metrics,
            aggregation=args.aggregation,
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
            use_nli=args.use_nli,
            nli_stride=args.nli_stride,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Variants: {len(payload['variants'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
