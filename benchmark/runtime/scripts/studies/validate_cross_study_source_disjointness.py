#!/usr/bin/env python3
"""Validate cross-study source disjointness and Study B multi-turn uniqueness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    raise FileNotFoundError(f"Unable to resolve any of: {candidates} under {root}")


def _extract_main_pairs(payload: dict[str, Any]) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for sample in payload.get("samples", []):
        metadata = sample.get("metadata", {}) or {}
        split_name = str(metadata.get("source_split", "") or "").strip().lower()
        if split_name not in {"test", "train"}:
            continue
        for source_id in metadata.get("source_openr1_ids", []) or []:
            pairs.add((split_name, int(source_id)))
    return pairs


def _extract_bias_pairs(payload: dict[str, Any]) -> tuple[set[tuple[str, int]], dict[tuple[str, int], list[str]]]:
    pairs: set[tuple[str, int]] = set()
    pair_to_groups: dict[tuple[str, int], list[str]] = {}
    for case in payload.get("cases", []):
        metadata = case.get("metadata", {}) or {}
        split_name = str(metadata.get("source_openr1_split", "") or "").strip().lower()
        source_id = metadata.get("source_openr1_id")
        if split_name not in {"test", "train"} or source_id is None:
            continue
        pair = (split_name, int(source_id))
        pairs.add(pair)
        pair_to_groups.setdefault(pair, []).append(str(case.get("pair_group_id", "") or ""))
    return pairs, pair_to_groups


def _extract_case_pairs(cases: list[dict[str, Any]]) -> tuple[set[tuple[str, int]], dict[tuple[str, int], list[str]]]:
    pairs: set[tuple[str, int]] = set()
    pair_to_ids: dict[tuple[str, int], list[str]] = {}
    for case in cases:
        metadata = case.get("metadata", {}) or {}
        split_name = str(metadata.get("source_split", "") or "").strip().lower()
        if split_name not in {"test", "train"}:
            continue
        for source_id in metadata.get("source_openr1_ids", []) or []:
            pair = (split_name, int(source_id))
            pairs.add(pair)
            pair_to_ids.setdefault(pair, []).append(str(case.get("id", "") or ""))
    return pairs, pair_to_ids


def _multi_turn_signature(case: dict[str, Any]) -> str:
    turns = case.get("turns", [])
    joined = "||".join(" ".join(str(turn.get("message", "") or "").split()).strip().lower() for turn in turns)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Working data root or frozen snapshot root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/verification/v4_1/cross_study_source_disjointness.json"),
        help="Where to write the validation summary JSON.",
    )
    args = parser.parse_args()

    root = args.data_root.resolve()
    study_a_path = _resolve_path(root, ["openr1_psy_splits/study_a_test.json", "study_a_test.json"])
    study_b_single_path = _resolve_path(root, ["openr1_psy_splits/study_b_test.json", "study_b_test.json"])
    study_b_multi_path = _resolve_path(root, ["openr1_psy_splits/study_b_multi_turn_test.json", "study_b_multi_turn_test.json"])
    study_c_path = _resolve_path(root, ["openr1_psy_splits/study_c_test.json", "study_c_test.json"])
    bias_path = _resolve_path(root, ["adversarial_bias/biased_vignettes.json"])

    study_a_pairs = _extract_main_pairs(_load_json(study_a_path))
    study_b_single_payload = _load_json(study_b_single_path)
    study_b_single_cases = study_b_single_payload if isinstance(study_b_single_payload, list) else study_b_single_payload.get("samples", [])
    study_b_multi_payload = _load_json(study_b_multi_path)
    study_b_multi_cases = study_b_multi_payload if isinstance(study_b_multi_payload, list) else study_b_multi_payload.get("cases", [])
    study_c_payload = _load_json(study_c_path)
    study_c_cases = study_c_payload.get("cases", []) if isinstance(study_c_payload, dict) else study_c_payload

    bias_pairs, bias_pair_to_groups = _extract_bias_pairs(_load_json(bias_path))
    study_b_single_pairs, study_b_single_pair_to_ids = _extract_case_pairs(study_b_single_cases)
    study_b_multi_pairs, study_b_multi_pair_to_ids = _extract_case_pairs(study_b_multi_cases)
    study_c_pairs, study_c_pair_to_ids = _extract_case_pairs(study_c_cases)

    signatures: dict[str, str] = {}
    duplicate_signatures: list[dict[str, str]] = []
    for case in study_b_multi_cases:
        case_id = str(case.get("id", "") or "")
        sig = _multi_turn_signature(case)
        duplicate_of = signatures.get(sig)
        if duplicate_of:
            duplicate_signatures.append(
                {
                    "id": case_id,
                    "duplicate_of": duplicate_of,
                    "signature_sha256": sig,
                }
            )
        else:
            signatures[sig] = case_id

    internal_multi_source_duplicates = {
        f"{split}:{source_id}": ids
        for (split, source_id), ids in study_b_multi_pair_to_ids.items()
        if len(ids) > 1
    }
    bias_vs_main = study_a_pairs & bias_pairs
    multi_vs_main = study_b_multi_pairs & study_a_pairs
    multi_vs_bias = study_b_multi_pairs & bias_pairs
    multi_vs_single = study_b_multi_pairs & study_b_single_pairs
    multi_vs_c = study_b_multi_pairs & study_c_pairs
    study_c_vs_main = study_c_pairs & study_a_pairs
    study_c_vs_bias = study_c_pairs & bias_pairs
    study_c_vs_single = study_c_pairs & study_b_single_pairs
    study_c_vs_multi = study_c_pairs & study_b_multi_pairs

    summary = {
        "study_a_bias_vs_main_overlap": sorted(f"{split}:{source_id}" for split, source_id in bias_vs_main),
        "study_b_multi_vs_study_a_overlap": sorted(f"{split}:{source_id}" for split, source_id in multi_vs_main),
        "study_b_multi_vs_study_a_bias_overlap": sorted(f"{split}:{source_id}" for split, source_id in multi_vs_bias),
        "study_b_multi_vs_study_b_single_overlap": sorted(f"{split}:{source_id}" for split, source_id in multi_vs_single),
        "study_b_multi_vs_study_c_overlap": sorted(f"{split}:{source_id}" for split, source_id in multi_vs_c),
        "study_c_vs_study_a_overlap": sorted(f"{split}:{source_id}" for split, source_id in study_c_vs_main),
        "study_c_vs_study_a_bias_overlap": sorted(f"{split}:{source_id}" for split, source_id in study_c_vs_bias),
        "study_c_vs_study_b_single_overlap": sorted(f"{split}:{source_id}" for split, source_id in study_c_vs_single),
        "study_c_vs_study_b_multi_overlap": sorted(f"{split}:{source_id}" for split, source_id in study_c_vs_multi),
        "study_b_multi_internal_source_duplicates": internal_multi_source_duplicates,
        "study_b_multi_duplicate_signatures": duplicate_signatures,
        "bias_overlap_groups": {
            f"{split}:{source_id}": sorted(set(bias_pair_to_groups[(split, source_id)]))
            for split, source_id in bias_vs_main
        },
        "study_b_multi_overlap_case_ids": {
            "vs_study_b_single": {
                f"{split}:{source_id}": study_b_multi_pair_to_ids[(split, source_id)]
                for split, source_id in multi_vs_single
            },
            "vs_study_a_bias": {
                f"{split}:{source_id}": study_b_multi_pair_to_ids[(split, source_id)]
                for split, source_id in multi_vs_bias
            },
            "vs_study_c": {
                f"{split}:{source_id}": study_b_multi_pair_to_ids[(split, source_id)]
                for split, source_id in multi_vs_c
            },
        },
        "study_c_overlap_case_ids": {
            "vs_study_a": {
                f"{split}:{source_id}": study_c_pair_to_ids[(split, source_id)]
                for split, source_id in study_c_vs_main
            },
            "vs_study_a_bias": {
                f"{split}:{source_id}": study_c_pair_to_ids[(split, source_id)]
                for split, source_id in study_c_vs_bias
            },
            "vs_study_b_single": {
                f"{split}:{source_id}": study_c_pair_to_ids[(split, source_id)]
                for split, source_id in study_c_vs_single
            },
            "vs_study_b_multi": {
                f"{split}:{source_id}": study_c_pair_to_ids[(split, source_id)]
                for split, source_id in study_c_vs_multi
            },
        },
        "study_b_single_overlap_case_ids": {
            f"{split}:{source_id}": study_b_single_pair_to_ids[(split, source_id)]
            for split, source_id in multi_vs_single
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    has_failures = any(
        (
            summary["study_a_bias_vs_main_overlap"],
            summary["study_b_multi_vs_study_a_overlap"],
            summary["study_b_multi_vs_study_a_bias_overlap"],
            summary["study_b_multi_vs_study_b_single_overlap"],
            summary["study_b_multi_vs_study_c_overlap"],
            summary["study_c_vs_study_a_overlap"],
            summary["study_c_vs_study_a_bias_overlap"],
            summary["study_c_vs_study_b_single_overlap"],
            summary["study_c_vs_study_b_multi_overlap"],
            summary["study_b_multi_internal_source_duplicates"],
            summary["study_b_multi_duplicate_signatures"],
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
