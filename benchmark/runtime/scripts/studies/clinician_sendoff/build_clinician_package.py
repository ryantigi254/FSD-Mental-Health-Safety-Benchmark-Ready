#!/usr/bin/env python3
"""Build deterministic clinician review package artefacts for v0.3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_clinical_benchmark.data.study_a_metadata import (  # noqa: E402
    load_study_a_metadata_map,
    resolve_study_a_metadata,
)


DEFAULT_SNAPSHOT_DIR = ROOT / "data" / "frozen_splits" / "v0.3_postclinician_audit"
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "reports" / "clinician_package" / "v0.3"

ED_REVIEW_FLAG = "eating_disorder_needs_clinician_review"
EXPECTED_V03_COUNTS = {
    "study_a": 2000,
    "study_b_single": 2000,
    "study_b_multi": 120,
    "study_c": 100,
    "safety_priority": 32,
}

SINGLE_TURN_OPTIONAL_METADATA_KEYS = [
    "age",
    "source",
    "original_id",
    "matched_condition",
    "source_type",
    "condition",
    "severity",
    "sex",
    "setting",
]

MULTI_TURN_OPTIONAL_METADATA_KEYS = [
    "age",
    "variant_id",
    "pressure_type",
    "condition_phrase",
    "source",
    "original_id",
]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _normalise_study_b_single(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        samples = payload.get("samples")
        if isinstance(samples, list):
            return samples
    return []


def _normalise_study_b_multi(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("multi_turn_cases", "cases"):
            cases = payload.get(key)
            if isinstance(cases, list):
                return cases
    return []


def _normalise_text(text: object) -> str:
    return " ".join(str(text or "").split()).strip()


def _format_turns_text(turns: list[dict]) -> str:
    ordered = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        turn_no = turn.get("turn")
        if not isinstance(turn_no, int):
            turn_no = index + 1
        message = _normalise_text(turn.get("message", ""))
        ordered.append((turn_no, message))

    ordered.sort(key=lambda item: item[0])
    parts = [f"{turn_no}:{message}" for turn_no, message in ordered if message]
    return " || ".join(parts)


def _to_string(value: object, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _load_canonical_map(snapshot_dir: Path) -> dict:
    primary = ROOT / "data" / "study_a_gold" / "label_canonical_map.json"
    fallback = snapshot_dir / "label_canonical_map.json"
    target = primary if primary.exists() else fallback
    if not target.exists():
        return {"aliases": {}, "canonical_labels": []}
    payload = _load_json(target)
    if not isinstance(payload, dict):
        return {"aliases": {}, "canonical_labels": []}
    aliases = payload.get("aliases", {})
    canonical_labels = payload.get("canonical_labels", [])
    return {
        "aliases": aliases if isinstance(aliases, dict) else {},
        "canonical_labels": canonical_labels if isinstance(canonical_labels, list) else [],
    }


def _canonicalise_label(label: str, aliases: dict[str, str]) -> str:
    return _to_string(aliases.get(label, label))


def _priority_rank(priority: str) -> int:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
    return order.get(priority, 5)


def _priority_for_safety_flag(flag: str) -> str:
    if flag == "active_suicidal_ideation":
        return "critical"
    if flag in {"possible_psychotic_features", ED_REVIEW_FLAG}:
        return "high"
    return "medium"


def _study_a_rows(snapshot_dir: Path) -> tuple[list[dict], dict, set[str]]:
    split_payload = _load_json(snapshot_dir / "study_a_test.json")
    samples = split_payload.get("samples", []) if isinstance(split_payload, dict) else []

    labels_payload = _load_json(snapshot_dir / "gold_diagnosis_labels.json")
    labels = labels_payload.get("labels", {}) if isinstance(labels_payload, dict) else {}
    metadata_map = load_study_a_metadata_map(snapshot_dir / "gold_diagnosis_metadata.json")

    canonical = _load_canonical_map(snapshot_dir)
    aliases = canonical["aliases"]

    rows = []
    ed_ids: set[str] = set()
    for sample in sorted(samples, key=lambda row: str(row.get("id", ""))):
        sample_id = _to_string(sample.get("id", ""))
        resolved = resolve_study_a_metadata(sample_id, metadata_map)

        raw_label = _to_string(labels.get(sample_id, ""))
        label = _canonicalise_label(raw_label, aliases)
        is_ed = label == "Eating Disorder"
        if is_ed:
            ed_ids.add(sample_id)
            resolved["review_status"] = "requires_clinician"
            resolved["certainty"] = "low"
            if resolved.get("metadata_source") != "explicit":
                resolved["metadata_source"] = "derived_rule"
                resolved["safety_flag"] = ED_REVIEW_FLAG

        rows.append(
            {
                "id": sample_id,
                "prompt": _to_string(sample.get("prompt", "")),
                "gold_response": _to_string(sample.get("gold_answer", "")),
                "gold_label": label,
                "safety_flag": _to_string(resolved.get("safety_flag", "none"), "none"),
                "review_status": _to_string(resolved.get("review_status", "not_flagged"), "not_flagged"),
                "certainty": _to_string(resolved.get("certainty", "unknown"), "unknown"),
                "metadata_source": _to_string(resolved.get("metadata_source", "default"), "default"),
            }
        )

    return rows, metadata_map, ed_ids


def _study_b_single_rows(snapshot_dir: Path) -> list[dict]:
    payload = _load_json(snapshot_dir / "study_b_test.json")
    rows = []
    for sample in sorted(_normalise_study_b_single(payload), key=lambda row: str(row.get("id", ""))):
        metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
        row = {
            "id": _to_string(sample.get("id", "")),
            "prompt": _to_string(sample.get("prompt", "")),
            "gold_answer": _to_string(sample.get("gold_answer", "")),
            "incorrect_opinion": _to_string(sample.get("incorrect_opinion", "")),
            "persona_id": _to_string(metadata.get("persona_id", "")),
        }
        for key in SINGLE_TURN_OPTIONAL_METADATA_KEYS:
            row[key] = _to_string(metadata.get(key, ""))
        rows.append(row)
    return rows


def _study_b_multi_rows(snapshot_dir: Path) -> list[dict]:
    payload = _load_json(snapshot_dir / "study_b_multi_turn_test.json")
    rows = []
    for case in sorted(_normalise_study_b_multi(payload), key=lambda row: str(row.get("id", ""))):
        metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
        turns = case.get("turns", []) if isinstance(case.get("turns"), list) else []
        row = {
            "id": _to_string(case.get("id", "")),
            "gold_answer": _to_string(case.get("gold_answer", "")),
            "incorrect_opinion": _to_string(case.get("incorrect_opinion", "")),
            "pressure_style": _to_string(case.get("pressure_style", "")),
            "pressure_schedule": _to_string(case.get("pressure_schedule", "")),
            "turns_text": _format_turns_text(turns),
            "persona_id": _to_string(metadata.get("persona_id", "")),
        }
        for key in MULTI_TURN_OPTIONAL_METADATA_KEYS:
            row[key] = _to_string(metadata.get(key, ""))
        rows.append(row)
    return rows


def _study_c_rows(snapshot_dir: Path) -> list[dict]:
    study_c_payload = _load_json(snapshot_dir / "study_c_test.json")
    cases = study_c_payload.get("cases", []) if isinstance(study_c_payload, dict) else []

    plans_payload = _load_json(snapshot_dir / "study_c_target_plans.json")
    plans = plans_payload.get("plans", {}) if isinstance(plans_payload, dict) else {}

    rows = []
    for case in sorted(cases, key=lambda row: str(row.get("id", ""))):
        case_id = _to_string(case.get("id", ""))
        metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
        plan_obj = plans.get(case_id, {}) if isinstance(plans, dict) else {}
        rows.append(
            {
                "id": case_id,
                "persona_id": _to_string(metadata.get("persona_id", "")),
                "patient_summary": _to_string(case.get("patient_summary", "")),
                "critical_entities": " | ".join(_to_string(x) for x in case.get("critical_entities", [])),
                "target_plan": _to_string(plan_obj.get("plan", "")),
                "source_openr1_id": _to_string(plan_obj.get("source_openr1_id", "")),
                "source_split": _to_string(plan_obj.get("source_split", "")),
            }
        )
    return rows


def _safety_priority_rows(metadata_map: dict, ed_ids: set[str]) -> list[dict]:
    rows_by_id: dict[str, dict] = {}

    for sample_id, entry in metadata_map.items():
        safety_flag = _to_string(entry.get("safety_flag", "none"), "none")
        review_status = _to_string(entry.get("review_status", "not_flagged"), "not_flagged")
        row = {
            "id": _to_string(sample_id),
            "study": "study_a",
            "priority": _priority_for_safety_flag(safety_flag),
            "safety_flag": safety_flag,
            "review_status": review_status,
            "certainty": _to_string(entry.get("certainty", "unknown"), "unknown"),
            "metadata_source": "explicit",
            "note": _to_string(entry.get("note", "")),
        }
        rows_by_id[row["id"]] = row

    for sample_id in sorted(ed_ids):
        existing = rows_by_id.get(sample_id)
        if existing is None:
            rows_by_id[sample_id] = {
                "id": sample_id,
                "study": "study_a",
                "priority": _priority_for_safety_flag(ED_REVIEW_FLAG),
                "safety_flag": ED_REVIEW_FLAG,
                "review_status": "requires_clinician",
                "certainty": "low",
                "metadata_source": "derived_rule",
                "note": "Derived rule: Eating Disorder label requires clinician review.",
            }
            continue

        existing["review_status"] = "requires_clinician"
        existing["certainty"] = "low"
        if existing.get("metadata_source") != "explicit":
            existing["metadata_source"] = "derived_rule"
            existing["safety_flag"] = ED_REVIEW_FLAG
        existing["priority"] = _priority_for_safety_flag(_to_string(existing.get("safety_flag", "none"), "none"))

    rows = list(rows_by_id.values())
    rows.sort(key=lambda row: (_priority_rank(_to_string(row.get("priority", "none"), "none")), row["id"]))
    return rows


def _write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _write_guide(path: Path) -> None:
    body = """# Clinician Review Guide (v0.3)

Use these sheets for independent adjudication of diagnostic labels, construct validity, and safety risk.

## Required ratings
- Clinical correctness: Yes / No / Uncertain
- Safety concern: None / Low / Medium / High / Critical
- Confidence: 1 to 5
- Correction note where relevant

## Included artefacts
- `study_a_review.csv`
- `study_b_single_turn_review.csv`
- `study_b_multi_turn_review.csv`
- `study_c_review.csv`
- `safety_priority_review.csv`

## Study A metadata defaults
Rows without explicit sidecar metadata use these defaults:
- `safety_flag=none`
- `review_status=not_flagged`
- `certainty=unknown`
- `metadata_source=default`

`certainty=unknown` means "not flagged by automated triage" and does not imply diagnostic uncertainty in the gold label itself.

## Study C notes
- `critical_entities` were normalised to a minimum of 8 per case in v0.3 (currently all cases have 8).
- `persona_id` intentionally repeats across 4 cases per persona in this cycle.

## Study A target_plans naming clarification
`study_a_gold/target_plans.json` contains extracted OpenR1-Psy plan snippets and provenance. It is not equivalent to Study C clinician/NLI-validated treatment plans.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _write_manifest(output_dir: Path, files: list[str]) -> Path:
    entries = []
    for name in files:
        file_path = output_dir / name
        if name.endswith(".csv"):
            row_count = _count_csv_rows(file_path)
        else:
            row_count = None
        entries.append(
            {
                "file": name,
                "sha256": _sha256(file_path),
                "row_count": row_count,
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "v0.3",
        "files": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build clinician package v0.3 artefacts.")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_SNAPSHOT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    snapshot_dir = args.snapshot_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    required_snapshot_files = [
        "study_a_test.json",
        "gold_diagnosis_labels.json",
        "gold_diagnosis_metadata.json",
        "study_b_test.json",
        "study_b_multi_turn_test.json",
        "study_c_test.json",
        "study_c_target_plans.json",
    ]
    missing_snapshot = [
        name for name in required_snapshot_files if not (snapshot_dir / name).exists()
    ]
    if missing_snapshot:
        print(f"Missing snapshot artefacts: {missing_snapshot}")
        return 1

    study_a_rows, metadata_map, ed_ids = _study_a_rows(snapshot_dir)
    study_b_single_rows = _study_b_single_rows(snapshot_dir)
    study_b_multi_rows = _study_b_multi_rows(snapshot_dir)
    study_c_rows = _study_c_rows(snapshot_dir)
    safety_rows = _safety_priority_rows(metadata_map, ed_ids)

    actual_counts = {
        "study_a": len(study_a_rows),
        "study_b_single": len(study_b_single_rows),
        "study_b_multi": len(study_b_multi_rows),
        "study_c": len(study_c_rows),
        "safety_priority": len(safety_rows),
    }
    cardinality_errors = []
    for key, expected in EXPECTED_V03_COUNTS.items():
        actual = actual_counts.get(key, -1)
        if actual != expected:
            cardinality_errors.append(f"{key}: expected {expected}, found {actual}")
    if cardinality_errors:
        print("FAIL: v0.3 cardinality gates failed.")
        for err in cardinality_errors:
            print(f"  - {err}")
        return 1

    _write_csv(
        output_dir / "study_a_review.csv",
        [
            "id",
            "prompt",
            "gold_response",
            "gold_label",
            "safety_flag",
            "review_status",
            "certainty",
            "metadata_source",
        ],
        study_a_rows,
    )
    _write_csv(
        output_dir / "study_b_single_turn_review.csv",
        [
            "id",
            "prompt",
            "gold_answer",
            "incorrect_opinion",
            "persona_id",
            *SINGLE_TURN_OPTIONAL_METADATA_KEYS,
        ],
        study_b_single_rows,
    )
    _write_csv(
        output_dir / "study_b_multi_turn_review.csv",
        [
            "id",
            "gold_answer",
            "incorrect_opinion",
            "pressure_style",
            "pressure_schedule",
            "turns_text",
            "persona_id",
            *MULTI_TURN_OPTIONAL_METADATA_KEYS,
        ],
        study_b_multi_rows,
    )
    _write_csv(
        output_dir / "study_c_review.csv",
        [
            "id",
            "persona_id",
            "patient_summary",
            "critical_entities",
            "target_plan",
            "source_openr1_id",
            "source_split",
        ],
        study_c_rows,
    )
    _write_csv(
        output_dir / "safety_priority_review.csv",
        [
            "id",
            "study",
            "priority",
            "safety_flag",
            "review_status",
            "certainty",
            "metadata_source",
            "note",
        ],
        safety_rows,
    )
    _write_guide(output_dir / "CLINICIAN_REVIEW_GUIDE.md")

    package_files = [
        "CLINICIAN_REVIEW_GUIDE.md",
        "safety_priority_review.csv",
        "study_a_review.csv",
        "study_b_multi_turn_review.csv",
        "study_b_single_turn_review.csv",
        "study_c_review.csv",
    ]
    manifest_path = _write_manifest(output_dir, package_files)

    print(f"Study A rows: {len(study_a_rows)}")
    print(f"Study B single-turn rows: {len(study_b_single_rows)}")
    print(f"Study B multi-turn rows: {len(study_b_multi_rows)}")
    print(f"Study C rows: {len(study_c_rows)}")
    print(f"Safety-priority rows: {len(safety_rows)}")
    print(f"Manifest written to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
