#!/usr/bin/env python3
"""Patch v4.1 Study A to remove OpenR1-Psy source ID conflicts with adversarial bias dataset.

Identifies Study A items whose source_openr1_ids overlap with adversarial bias
source_openr1_id values, and replaces only those items with new ACCEPTABLE
candidates that don't conflict with either dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from reliable_clinical_benchmark.review import (
    extract_diagnosis_from_text,
    iter_openr1_candidates,
    load_rules,
    now_iso,
    score_study_a,
)


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = RUNTIME_ROOT / "data"

V41_FROZEN = DATA_ROOT / "frozen_splits" / "v4_1_resampled"
V4_RULES = DATA_ROOT / "verification" / "v4" / "rubric_rules_v2.json"

BIAS_VIGNETTES = V41_FROZEN / "adversarial_bias" / "biased_vignettes.json"
STUDY_A_PATH = V41_FROZEN / "study_a_test.json"
GOLD_LABELS_PATH = V41_FROZEN / "gold_diagnosis_labels.json"
STUDY_A_SUB_PATH = V41_FROZEN / "study_a" / "study_a_test.json"
GOLD_LABELS_SUB_PATH = V41_FROZEN / "study_a" / "gold_diagnosis_labels.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_bias_source_ids(bias_path: Path) -> set[int]:
    """Extract all OpenR1-Psy source IDs used by the adversarial bias dataset."""
    payload = _load_json(bias_path)
    cases = payload.get("cases", [])
    ids: set[int] = set()
    for case in cases:
        meta = case.get("metadata", {})
        oid = meta.get("source_openr1_id")
        if isinstance(oid, int):
            ids.add(oid)
    return ids


def _extract_study_a_source_ids(samples: list[dict[str, Any]]) -> set[int]:
    """Extract all OpenR1-Psy source IDs from Study A samples."""
    ids: set[int] = set()
    for sample in samples:
        meta = sample.get("metadata", {})
        if not isinstance(meta, dict):
            continue
        for sid in meta.get("source_openr1_ids", []):
            if isinstance(sid, int):
                ids.add(sid)
    return ids


def _find_conflicting_items(
    samples: list[dict[str, Any]], bias_ids: set[int]
) -> list[str]:
    """Return item IDs whose source OpenR1 IDs overlap with bias dataset."""
    conflicts: list[str] = []
    for sample in samples:
        meta = sample.get("metadata", {})
        if not isinstance(meta, dict):
            continue
        source_ids = meta.get("source_openr1_ids", [])
        if any(isinstance(sid, int) and sid in bias_ids for sid in source_ids):
            conflicts.append(str(sample.get("id", "")))
    return conflicts


def main() -> int:
    print("=== V4.1 Bias Conflict Patch ===\n")

    # Load data
    rules = load_rules(V4_RULES)
    sa_payload = _load_json(STUDY_A_PATH)
    samples = sa_payload.get("samples", [])
    labels_payload = _load_json(GOLD_LABELS_PATH)
    labels = labels_payload.get("labels", {})
    bias_ids = _extract_bias_source_ids(BIAS_VIGNETTES)

    print(f"Study A samples: {len(samples)}")
    print(f"Gold labels: {len(labels)}")
    print(f"Adversarial bias unique source IDs: {len(bias_ids)}")

    # Find conflicts
    conflict_ids = _find_conflicting_items(samples, bias_ids)
    print(f"Conflicting items: {len(conflict_ids)}")

    if not conflict_ids:
        print("\nNo conflicts found. Nothing to patch.")
        return 0

    # Build full exclusion set: all Study A IDs + all bias IDs
    all_sa_source_ids = _extract_study_a_source_ids(samples)
    exclusion_ids = all_sa_source_ids | bias_ids
    print(f"Total exclusion set: {len(exclusion_ids)} IDs")

    # Build replacement map
    samples_by_id = {str(s.get("id", "")): s for s in samples}
    replacements: dict[str, dict[str, Any]] = {}
    replacement_labels: dict[str, str] = {}
    pool_scanned = 0

    candidates = iter_openr1_candidates(used_source_ids=exclusion_ids)

    for item_id in conflict_ids:
        while True:
            try:
                candidate = next(candidates)
            except StopIteration:
                print(
                    f"\nERROR: Pool exhausted after {pool_scanned} candidates. "
                    f"Filled {len(replacements)}/{len(conflict_ids)}."
                )
                return 1

            pool_scanned += 1
            diagnosis_label = str(candidate.get("diagnosis_label", "") or "").strip()
            if not diagnosis_label:
                continue

            candidate_item = {
                "id": item_id,
                "prompt": candidate["prompt"],
                "gold_answer": candidate["gold_answer"],
                "gold_reasoning": candidate["gold_reasoning"],
                "metadata": {
                    "source_openr1_ids": [candidate["openr1_id"]],
                    "source_split": candidate["split"],
                    "resampled_from": "OpenR1-Psy",
                    "resampled_v4_1": True,
                    "bias_conflict_patch": True,
                },
            }

            review = score_study_a(candidate_item, diagnosis_label, rules)
            if review["verdict"] != "ACCEPTABLE":
                continue

            replacements[item_id] = candidate_item
            replacement_labels[item_id] = diagnosis_label
            exclusion_ids.add(int(candidate["openr1_id"]))

            print(
                f"  Replaced {item_id}: openr1={candidate['openr1_id']} "
                f"split={candidate['split']} diag={diagnosis_label} "
                f"(scanned {pool_scanned})"
            )
            break

    print(f"\nReplaced {len(replacements)}/{len(conflict_ids)} conflicting items")
    print(f"Pool candidates scanned: {pool_scanned}")

    # Rebuild samples list preserving order
    rebuilt_samples = []
    for sample in samples:
        item_id = str(sample.get("id", ""))
        if item_id in replacements:
            rebuilt_samples.append(replacements[item_id])
        else:
            rebuilt_samples.append(sample)

    # Update labels
    rebuilt_labels = dict(labels)
    rebuilt_labels.update(replacement_labels)

    # Validate
    ids = [str(s.get("id", "")) for s in rebuilt_samples]
    assert len(ids) == len(set(ids)), "Duplicate IDs after patch"
    assert len(rebuilt_samples) == 2000, f"Expected 2000 samples, got {len(rebuilt_samples)}"
    assert set(rebuilt_labels.keys()) == set(ids), "Label key mismatch"

    # Score all to verify
    print("\nVerifying all 2000 items pass rubric v2...")
    verdicts = Counter()
    for item in rebuilt_samples:
        item_id = str(item.get("id", ""))
        gold_label = rebuilt_labels.get(item_id, "")
        result = score_study_a(item, gold_label, rules)
        verdicts[result["verdict"]] += 1

    print(f"  ACCEPTABLE: {verdicts['ACCEPTABLE']}")
    print(f"  NEEDS_REVIEW: {verdicts['NEEDS_REVIEW']}")
    print(f"  REJECT: {verdicts['REJECT']}")

    if verdicts["ACCEPTABLE"] != 2000:
        print("\nERROR: Not all items ACCEPTABLE after patch!")
        return 1

    # Verify no bias conflicts remain
    new_sa_ids = _extract_study_a_source_ids(rebuilt_samples)
    remaining_conflicts = new_sa_ids & bias_ids
    print(f"\nRemaining bias conflicts: {len(remaining_conflicts)}")

    if remaining_conflicts:
        print("ERROR: Bias conflicts still present!")
        return 1

    # Write updated files
    study_a_meta = sa_payload.get("meta", {})
    if not isinstance(study_a_meta, dict):
        study_a_meta = {}

    study_a_out = {
        "samples": rebuilt_samples,
        "meta": {
            **study_a_meta,
            "bias_conflict_patch_utc": now_iso(),
            "bias_conflicts_patched": len(replacements),
        },
    }

    labels_meta = labels_payload.get("meta", {})
    if not isinstance(labels_meta, dict):
        labels_meta = {}

    labels_out = {
        "labels": rebuilt_labels,
        "meta": {
            **labels_meta,
            "bias_conflict_patch_utc": now_iso(),
            "bias_conflicts_patched": len(replacements),
        },
    }

    _write_json(STUDY_A_PATH, study_a_out)
    print(f"\nWrote {STUDY_A_PATH}")

    _write_json(GOLD_LABELS_PATH, labels_out)
    print(f"Wrote {GOLD_LABELS_PATH}")

    # Update study_a subdirectory copies
    if STUDY_A_SUB_PATH.parent.exists():
        _write_json(STUDY_A_SUB_PATH, study_a_out)
        print(f"Wrote {STUDY_A_SUB_PATH}")

        _write_json(GOLD_LABELS_SUB_PATH, labels_out)
        print(f"Wrote {GOLD_LABELS_SUB_PATH}")

    # Update manifest
    manifest_path = V41_FROZEN / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if isinstance(manifest, dict) and isinstance(manifest.get("files"), list):
            for entry in manifest["files"]:
                fname = entry.get("file", "")
                if fname in ("study_a_test.json", "study_a/study_a_test.json"):
                    entry["sha256"] = _sha256_file(
                        V41_FROZEN / fname
                    )
                elif fname in (
                    "gold_diagnosis_labels.json",
                    "study_a/gold_diagnosis_labels.json",
                ):
                    entry["sha256"] = _sha256_file(
                        V41_FROZEN / fname
                    )
            _write_json(manifest_path, manifest)
            print(f"Updated {manifest_path}")

    print("\n=== Patch complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
