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


def _study_a_rows(snapshot_dir: Path) -> list[dict]:
    split_payload = _load_json(snapshot_dir / "study_a_test.json")
    samples = split_payload.get("samples", []) if isinstance(split_payload, dict) else []

    labels_payload = _load_json(snapshot_dir / "gold_diagnosis_labels.json")
    labels = labels_payload.get("labels", {}) if isinstance(labels_payload, dict) else {}
    metadata_map = load_study_a_metadata_map(snapshot_dir / "gold_diagnosis_metadata.json")

    rows = []
    for sample in sorted(samples, key=lambda row: str(row.get("id", ""))):
        sample_id = str(sample.get("id", ""))
        resolved = resolve_study_a_metadata(sample_id, metadata_map)
        rows.append(
            {
                "id": sample_id,
                "prompt": str(sample.get("prompt", "")),
                "gold_response": str(sample.get("gold_answer", "")),
                "gold_label": str(labels.get(sample_id, "")),
                "safety_flag": str(resolved.get("safety_flag", "none")),
                "review_status": str(resolved.get("review_status", "not_flagged")),
                "certainty": str(resolved.get("certainty", "unknown")),
                "metadata_source": str(resolved.get("metadata_source", "default")),
            }
        )
    return rows


def _study_b_single_rows(snapshot_dir: Path) -> list[dict]:
    payload = _load_json(snapshot_dir / "study_b_test.json")
    rows = []
    for sample in sorted(_normalise_study_b_single(payload), key=lambda row: str(row.get("id", ""))):
        metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
        rows.append(
            {
                "id": str(sample.get("id", "")),
                "prompt": str(sample.get("prompt", "")),
                "gold_answer": str(sample.get("gold_answer", "")),
                "incorrect_opinion": str(sample.get("incorrect_opinion", "")),
                "persona_id": str(metadata.get("persona_id", "")),
                "age": "" if metadata.get("age") is None else str(metadata.get("age")),
            }
        )
    return rows


def _study_b_multi_rows(snapshot_dir: Path) -> list[dict]:
    payload = _load_json(snapshot_dir / "study_b_multi_turn_test.json")
    rows = []
    for case in sorted(_normalise_study_b_multi(payload), key=lambda row: str(row.get("id", ""))):
        metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), dict) else {}
        turns = case.get("turns", []) if isinstance(case.get("turns"), list) else []
        rows.append(
            {
                "id": str(case.get("id", "")),
                "gold_answer": str(case.get("gold_answer", "")),
                "incorrect_opinion": str(case.get("incorrect_opinion", "")),
                "pressure_style": str(case.get("pressure_style", "")),
                "pressure_schedule": str(case.get("pressure_schedule", "")),
                "turn_count": str(len(turns)),
                "persona_id": str(metadata.get("persona_id", "")),
                "age": "" if metadata.get("age") is None else str(metadata.get("age")),
            }
        )
    return rows


def _study_c_rows(snapshot_dir: Path) -> list[dict]:
    study_c_payload = _load_json(snapshot_dir / "study_c_test.json")
    cases = study_c_payload.get("cases", []) if isinstance(study_c_payload, dict) else []

    plans_payload = _load_json(snapshot_dir / "study_c_target_plans.json")
    plans = plans_payload.get("plans", {}) if isinstance(plans_payload, dict) else {}

    rows = []
    for case in sorted(cases, key=lambda row: str(row.get("id", ""))):
        case_id = str(case.get("id", ""))
        plan_obj = plans.get(case_id, {}) if isinstance(plans, dict) else {}
        rows.append(
            {
                "id": case_id,
                "patient_summary": str(case.get("patient_summary", "")),
                "critical_entities": " | ".join(str(x) for x in case.get("critical_entities", [])),
                "target_plan": str(plan_obj.get("plan", "")),
                "source_openr1_id": str(plan_obj.get("source_openr1_id", "")),
                "source_split": str(plan_obj.get("source_split", "")),
            }
        )
    return rows


def _priority_rank(priority: str) -> int:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}
    return order.get(priority, 5)


def _safety_priority_rows(snapshot_dir: Path) -> list[dict]:
    metadata_map = load_study_a_metadata_map(snapshot_dir / "gold_diagnosis_metadata.json")

    rows = []
    for sample_id, entry in metadata_map.items():
        safety_flag = str(entry.get("safety_flag", "none"))
        review_status = str(entry.get("review_status", "not_flagged"))
        if safety_flag == "none" and review_status != "requires_clinician":
            continue

        if safety_flag == "active_suicidal_ideation":
            priority = "critical"
        elif safety_flag == "possible_psychotic_features":
            priority = "high"
        else:
            priority = "medium"

        rows.append(
            {
                "id": str(sample_id),
                "study": "study_a",
                "priority": priority,
                "safety_flag": safety_flag,
                "review_status": review_status,
                "certainty": str(entry.get("certainty", "unknown")),
                "note": str(entry.get("note", "")),
            }
        )

    rows.sort(key=lambda row: (_priority_rank(row["priority"]), row["id"]))
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

    study_a_rows = _study_a_rows(snapshot_dir)
    study_b_single_rows = _study_b_single_rows(snapshot_dir)
    study_b_multi_rows = _study_b_multi_rows(snapshot_dir)
    study_c_rows = _study_c_rows(snapshot_dir)
    safety_rows = _safety_priority_rows(snapshot_dir)

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
            "age",
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
            "turn_count",
            "persona_id",
            "age",
        ],
        study_b_multi_rows,
    )
    _write_csv(
        output_dir / "study_c_review.csv",
        [
            "id",
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
