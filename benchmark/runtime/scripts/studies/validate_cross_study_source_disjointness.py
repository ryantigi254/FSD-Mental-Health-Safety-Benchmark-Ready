#!/usr/bin/env python3
"""Validate pairwise cross-study source disjointness for working or frozen snapshot roots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cases(root: Path, rel: str, kind: str) -> list[dict[str, Any]]:
    payload = _read_json(root / rel)
    if kind == "study_a":
        return payload.get("samples", [])
    if kind == "bias":
        return payload.get("cases", [])
    return payload if isinstance(payload, list) else payload.get("cases", []) or payload.get("samples", [])


def _refs_from_study_a(samples: list[dict[str, Any]]) -> tuple[set[tuple[str, int]], dict[tuple[str, int], list[str]]]:
    refs: set[tuple[str, int]] = set()
    mapping: dict[tuple[str, int], list[str]] = {}
    for sample in samples:
        metadata = sample.get("metadata", {}) or {}
        split = str(metadata.get("source_split", "") or "").strip().lower()
        for source_id in metadata.get("source_openr1_ids", []) or []:
            pair = (split, int(source_id))
            refs.add(pair)
            mapping.setdefault(pair, []).append(str(sample.get("id", "") or ""))
    return refs, mapping


def _refs_from_cases(cases: list[dict[str, Any]]) -> tuple[set[tuple[str, int]], dict[tuple[str, int], list[str]]]:
    refs: set[tuple[str, int]] = set()
    mapping: dict[tuple[str, int], list[str]] = {}
    for case in cases:
        metadata = case.get("metadata", {}) or {}
        split = str(metadata.get("source_split", "") or "").strip().lower()
        for source_id in metadata.get("source_openr1_ids", []) or []:
            pair = (split, int(source_id))
            refs.add(pair)
            mapping.setdefault(pair, []).append(str(case.get("id", "") or ""))

        split = str(metadata.get("source_openr1_split", "") or "").strip().lower()
        source_id = metadata.get("source_openr1_id")
        if split in {"test", "train"} and source_id is not None:
            pair = (split, int(source_id))
            refs.add(pair)
            mapping.setdefault(pair, []).append(str(case.get("pair_group_id", "") or str(case.get("id", "") or "")))

        # Defence-in-depth: extract turn-level donor IDs
        for turn in case.get("turns", []):
            t_split = str(turn.get("source_openr1_split", "") or "").strip().lower()
            t_id = turn.get("source_openr1_id")
            if t_split in {"test", "train"} and t_id is not None:
                pair = (t_split, int(t_id))
                refs.add(pair)
                mapping.setdefault(pair, []).append(str(case.get("id", "") or ""))
    return refs, mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/verification/cross_study_source_disjointness.json"),
    )
    args = parser.parse_args()

    root = args.data_root.resolve()
    study_a, study_a_ids = _refs_from_study_a(_load_cases(root, "study_a_test.json", "study_a") if (root / "study_a_test.json").exists() else _load_cases(root / "openr1_psy_splits" if (root / "openr1_psy_splits").exists() else root, "study_a_test.json", "study_a"))

    def _pick(rel: str, kind: str):
        direct = root / rel
        nested = root / "openr1_psy_splits" / rel
        bias_nested = root / "adversarial_bias" / rel
        if direct.exists():
            return _refs_from_cases(_load_cases(root, rel, kind))
        if nested.exists():
            return _refs_from_cases(_load_cases(root / "openr1_psy_splits", rel, kind))
        if bias_nested.exists():
            return _refs_from_cases(_load_cases(root / "adversarial_bias", rel, kind))
        raise FileNotFoundError(rel)

    study_b_single, study_b_single_ids = _pick("study_b_test.json", "case")
    study_b_multi, study_b_multi_ids = _pick("study_b_multi_turn_test.json", "case")
    study_c, study_c_ids = _pick("study_c_test.json", "case")
    study_a_bias, study_a_bias_ids = _pick("biased_vignettes.json", "bias")

    report = {
        "study_a_vs_study_b_single_overlap": sorted(f"{s}:{i}" for s, i in (study_a & study_b_single)),
        "study_a_vs_study_b_multi_overlap": sorted(f"{s}:{i}" for s, i in (study_a & study_b_multi)),
        "study_a_vs_study_c_overlap": sorted(f"{s}:{i}" for s, i in (study_a & study_c)),
        "study_a_bias_vs_study_a_overlap": sorted(f"{s}:{i}" for s, i in (study_a_bias & study_a)),
        "study_a_bias_vs_study_b_single_overlap": sorted(f"{s}:{i}" for s, i in (study_a_bias & study_b_single)),
        "study_b_multi_vs_study_b_single_overlap": sorted(f"{s}:{i}" for s, i in (study_b_multi & study_b_single)),
        "study_b_multi_vs_study_c_overlap": sorted(f"{s}:{i}" for s, i in (study_b_multi & study_c)),
        "study_c_vs_study_b_single_overlap": sorted(f"{s}:{i}" for s, i in (study_c & study_b_single)),
        "study_c_vs_study_a_bias_overlap": sorted(f"{s}:{i}" for s, i in (study_c & study_a_bias)),
        "detail": {
            "study_a": {f"{s}:{i}": ids for (s, i), ids in study_a_ids.items()},
            "study_b_single": {f"{s}:{i}": ids for (s, i), ids in study_b_single_ids.items()},
            "study_b_multi": {f"{s}:{i}": ids for (s, i), ids in study_b_multi_ids.items()},
            "study_c": {f"{s}:{i}": ids for (s, i), ids in study_c_ids.items()},
            "study_a_bias": {f"{s}:{i}": ids for (s, i), ids in study_a_bias_ids.items()},
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    has_overlap = any(
        report[key]
        for key in report
        if key != "detail"
    )
    print(json.dumps({k: len(v) if isinstance(v, list) else "detail" for k, v in report.items()}, indent=2))
    return 1 if has_overlap else 0


if __name__ == "__main__":
    raise SystemExit(main())
