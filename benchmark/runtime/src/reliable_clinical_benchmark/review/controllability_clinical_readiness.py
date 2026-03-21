"""Shared helpers for controllability clinician-readiness workflows."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "data"
DOCS_ROOT = ROOT / "docs"
REVIEW_SCRIPT_ROOT = ROOT / "scripts" / "studies" / "controllability_review"

DEFAULT_CTRL_DIR = DATA_ROOT / "controllability_splits_large_resolved"
DEFAULT_SMALL_CTRL_DIR = DATA_ROOT / "controllability_splits"
DEFAULT_RULES_PATH = REVIEW_SCRIPT_ROOT / "rubric_rules_ctrl_study_a_c4_v2.json"
DEFAULT_VERIFICATION_DIR = DATA_ROOT / "verification" / "controllability_v0.1_large_resolved"
DEFAULT_PACKAGE_DIR = (
    DOCS_ROOT / "reports" / "clinician_package" / "controllability_v0.1_large_resolved"
)

EXPECTED_CTRL_COUNTS = {
    "study_a": 2243,
    "study_a_bias": 2243,
    "study_b_single": 2243,
    "study_b_multi": 112,
    "study_c": 112,
}

VALID_VERDICTS = {"ACCEPTABLE", "NEEDS_REVIEW", "REJECT"}

REVIEW_FILENAMES = {
    "study_a": "study_a_reference_verdicts.ssv",
    "study_a_bias": "study_a_bias_verdicts.ssv",
    "study_b_single": "study_b_single_verdicts.ssv",
    "study_b_multi": "study_b_multi_verdicts.ssv",
    "study_c": "study_c_verdicts.ssv",
    "summary": "review_summary.ssv",
    "metadata": "run_metadata.json",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def serialise_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def normalise_items(payload: Any, preferred_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unable to normalise payload using keys={preferred_keys}")


def load_ctrl_suite(ctrl_dir: Path) -> dict[str, Any]:
    study_a_payload = load_json(ctrl_dir / "study_a_controllability_test.json")
    study_a_bias_payload = load_json(ctrl_dir / "study_a_bias_controllability_test.json")
    study_b_single_payload = load_json(ctrl_dir / "study_b_controllability_test.json")
    study_b_multi_payload = load_json(ctrl_dir / "study_b_multi_turn_controllability_test.json")
    study_c_payload = load_json(ctrl_dir / "study_c_controllability_test.json")
    labels_payload = load_json(ctrl_dir / "ctrl_gold_diagnosis_labels.json")
    plans_payload = load_json(ctrl_dir / "ctrl_target_plans.json")
    build_manifest = load_json(ctrl_dir / "build_manifest.json")

    return {
        "study_a": normalise_items(study_a_payload, ("samples", "cases", "items")),
        "study_a_bias": normalise_items(study_a_bias_payload, ("cases", "samples", "items")),
        "study_b_single": normalise_items(study_b_single_payload, ("samples", "items")),
        "study_b_multi": normalise_items(
            study_b_multi_payload, ("cases", "multi_turn_cases", "items")
        ),
        "study_c": normalise_items(study_c_payload, ("cases", "samples", "items")),
        "labels": labels_payload,
        "plans": plans_payload,
        "build_manifest": build_manifest,
    }


def read_ssv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader)


def summarise_verdict_ssv(path: Path, expected_rows: int) -> dict[str, Any]:
    rows = read_ssv_rows(path)
    verdict_counts = Counter(str(row.get("verdict", "") or "").strip() for row in rows)
    item_ids = [str(row.get("item_id", "") or "").strip() for row in rows]
    invalid_verdicts = sorted(set(verdict_counts) - VALID_VERDICTS)

    return {
        "file": path.name,
        "actual_rows": len(rows),
        "expected_rows": expected_rows,
        "acceptable": int(verdict_counts.get("ACCEPTABLE", 0)),
        "needs_review": int(verdict_counts.get("NEEDS_REVIEW", 0)),
        "reject": int(verdict_counts.get("REJECT", 0)),
        "counts_match": len(rows) == expected_rows,
        "duplicate_item_ids": len(item_ids) - len(set(item_ids)),
        "invalid_verdicts": invalid_verdicts,
    }


def _source_refs_from_metadata(metadata: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    split = str(
        metadata.get("source_openr1_split")
        or metadata.get("source_split")
        or metadata.get("source")
        or ""
    ).strip()

    def _add(ref_value: Any) -> None:
        if ref_value is None:
            return
        ref_text = str(ref_value).strip()
        if not ref_text:
            return
        refs.add(f"{split}:{ref_text}" if split else ref_text)

    source_ids = metadata.get("source_openr1_ids")
    if isinstance(source_ids, list):
        for ref in source_ids:
            _add(ref)
    else:
        _add(metadata.get("source_openr1_id"))
    return refs


def suite_source_refs(ctrl_dir: Path) -> set[str]:
    suite = load_ctrl_suite(ctrl_dir)
    refs: set[str] = set()
    for study_name in ("study_a", "study_a_bias", "study_b_single", "study_b_multi", "study_c"):
        for item in suite[study_name]:
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            refs.update(_source_refs_from_metadata(metadata))
    return refs


def study_duplicate_ids(items: list[dict[str, Any]]) -> list[str]:
    counter = Counter(str(item.get("id", "") or "").strip() for item in items)
    return sorted(item_id for item_id, count in counter.items() if item_id and count > 1)


def review_summary_rows(summary_path: Path) -> list[dict[str, str]]:
    return read_ssv_rows(summary_path)


def review_blockers(summary_rows: list[dict[str, str]]) -> list[str]:
    blockers: list[str] = []
    for row in summary_rows:
        study = str(row.get("study", "") or "").strip()
        counts_match = str(row.get("counts_match", "") or "").strip()
        duplicate_count = int(str(row.get("duplicate_item_ids", "0") or "0"))
        needs_review = int(str(row.get("needs_review", "0") or "0"))
        reject = int(str(row.get("reject", "0") or "0"))

        if counts_match != "1":
            blockers.append(f"{study}:row_count_mismatch")
        if duplicate_count:
            blockers.append(f"{study}:duplicate_item_ids")
        if needs_review:
            blockers.append(f"{study}:needs_review={needs_review}")
        if reject:
            blockers.append(f"{study}:reject={reject}")
    return blockers


def run_stage2_gates(
    *,
    ctrl_dir: Path,
    small_ctrl_dir: Path = DEFAULT_SMALL_CTRL_DIR,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    expected = expected_counts or EXPECTED_CTRL_COUNTS
    suite = load_ctrl_suite(ctrl_dir)

    stage_gates: list[dict[str, Any]] = []

    for study_name in ("study_a", "study_a_bias", "study_b_single", "study_b_multi", "study_c"):
        actual = len(suite[study_name])
        expected_value = expected[study_name]
        stage_gates.append(
            {
                "name": f"{study_name}_row_count",
                "passed": actual == expected_value,
                "details": {"actual": actual, "expected": expected_value},
            }
        )
        duplicates = study_duplicate_ids(suite[study_name])
        stage_gates.append(
            {
                "name": f"{study_name}_duplicate_ids",
                "passed": not duplicates,
                "details": {"duplicate_ids": duplicates[:25], "duplicate_count": len(duplicates)},
            }
        )

    study_a_ids = {str(item.get("id", "") or "").strip() for item in suite["study_a"]}
    label_ids = {
        str(key).strip()
        for key in suite["labels"].get("labels", {})
        if str(key).strip()
    }
    missing_labels = sorted(study_a_ids - label_ids)
    stage_gates.append(
        {
            "name": "study_a_gold_label_coverage",
            "passed": not missing_labels,
            "details": {"missing_ids": missing_labels[:25], "missing_count": len(missing_labels)},
        }
    )

    study_c_ids = {str(item.get("id", "") or "").strip() for item in suite["study_c"]}
    plan_ids = {
        str(key).strip()
        for key in suite["plans"].get("plans", {})
        if str(key).strip()
    }
    missing_plans = sorted(study_c_ids - plan_ids)
    stage_gates.append(
        {
            "name": "study_c_target_plan_coverage",
            "passed": not missing_plans,
            "details": {"missing_ids": missing_plans[:25], "missing_count": len(missing_plans)},
        }
    )

    large_refs = suite_source_refs(ctrl_dir)
    small_refs = suite_source_refs(small_ctrl_dir)
    overlap = sorted(large_refs & small_refs)
    stage_gates.append(
        {
            "name": "source_ref_disjointness_vs_small_suite",
            "passed": not overlap,
            "details": {"overlap_count": len(overlap), "overlap_examples": overlap[:25]},
        }
    )

    invalid_multi_turn: list[dict[str, Any]] = []
    for item in suite["study_b_multi"]:
        turns = item.get("turns", [])
        if not isinstance(turns, list) or not turns:
            invalid_multi_turn.append({"id": item.get("id"), "issue": "turns_missing"})
            continue
        for turn in turns:
            if not str(turn.get("message", "") or "").strip():
                invalid_multi_turn.append({"id": item.get("id"), "issue": "blank_message"})
                break
            if turn.get("pressure_level") not in {0, 1, 2, 3}:
                invalid_multi_turn.append({"id": item.get("id"), "issue": "invalid_pressure_level"})
                break

    stage_gates.append(
        {
            "name": "study_b_multi_turn_structure",
            "passed": not invalid_multi_turn,
            "details": {
                "invalid_count": len(invalid_multi_turn),
                "invalid_examples": invalid_multi_turn[:25],
            },
        }
    )

    invalid_bias_items: list[dict[str, Any]] = []
    for item in suite["study_a_bias"]:
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        inferred_condition = str(metadata.get("inferred_condition", "") or "").strip().lower()
        checks = {
            "prompt": str(item.get("prompt", "") or "").strip(),
            "bias_feature": str(item.get("bias_feature", "") or "").strip(),
            "bias_label": str(item.get("bias_label", "") or "").strip(),
            "dimension": str(metadata.get("dimension", "") or "").strip(),
            "dimension_family": str(metadata.get("dimension_family", "") or "").strip(),
            "condition_resolution_source": str(
                metadata.get("condition_resolution_source", "") or ""
            ).strip(),
            "inferred_condition": inferred_condition if inferred_condition != "unresolved" else "",
        }
        missing_fields = sorted(key for key, value in checks.items() if not value)
        if missing_fields:
            invalid_bias_items.append({"id": item.get("id"), "missing_fields": missing_fields})

    stage_gates.append(
        {
            "name": "study_a_bias_metadata_contract",
            "passed": not invalid_bias_items,
            "details": {
                "invalid_count": len(invalid_bias_items),
                "invalid_examples": invalid_bias_items[:25],
            },
        }
    )

    label_meta = suite["labels"].get("meta", {}) if isinstance(suite["labels"], dict) else {}
    plan_meta = suite["plans"].get("meta", {}) if isinstance(suite["plans"], dict) else {}
    provenance_checks = {
        "label_backend_probe": str(label_meta.get("backend", "") or "").strip() == "probe",
        "plan_backend_probe": str(plan_meta.get("backend", "") or "").strip() == "probe",
        "label_probe_meta_present": isinstance(label_meta.get("probe_meta"), dict),
        "plan_probe_meta_present": isinstance(plan_meta.get("probe_meta"), dict),
    }
    stage_gates.append(
        {
            "name": "probe_gold_provenance_declared",
            "passed": all(provenance_checks.values()),
            "details": provenance_checks,
        }
    )

    overall_passed = all(gate["passed"] for gate in stage_gates)
    return {
        "generated_at_utc": now_iso(),
        "version": "controllability_v0.1_large_resolved_stage2",
        "ctrl_dir": serialise_path(ctrl_dir),
        "small_ctrl_dir": serialise_path(small_ctrl_dir),
        "expected_counts": expected,
        "overall_passed": overall_passed,
        "release_status": "ready" if overall_passed else "blocked",
        "gates": stage_gates,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _gold_meta_summary(ctrl_dir: Path, filename: str) -> dict[str, Any]:
    payload = load_json(ctrl_dir / filename)
    return {
        "source_file": filename,
        "generated_at_utc": now_iso(),
        "meta": payload.get("meta", {}) if isinstance(payload, dict) else {},
    }


def _package_readme(
    *,
    ctrl_dir: Path,
    verification_dir: Path,
    gate_report: dict[str, Any],
    review_rows: list[dict[str, str]],
) -> str:
    blockers = review_blockers(review_rows)
    review_state = "all studies acceptable" if not blockers else "review blockers present"
    release_status = "ready" if gate_report.get("overall_passed") and not blockers else "blocked"

    return "\n".join(
        [
            "# Controllability Clinician-Readiness Package",
            "",
            f"- Release line: `controllability_v0.1_large_resolved`",
            f"- Scope: large resolved controllability suite only at `{serialise_path(ctrl_dir)}`",
            f"- Verification root: `{serialise_path(verification_dir)}`",
            f"- Review mode: deterministic review-first, no auto-repair or resampling",
            "- This package is separate from the base-study clinician-ready v0.3 line.",
            "- Gold diagnosis labels and target plans are probe-backed, weakly supervised artefacts.",
            "- Robust `.robust.json` files are diagnostic appendices and are not release-gating inputs.",
            f"- Gate status: `{gate_report.get('release_status', 'blocked')}`",
            f"- Review status: `{review_state}`",
            f"- Package release status: `{release_status}`",
            "",
            "The package remains blocked unless every study is fully `ACCEPTABLE` and every stage-2 gate passes.",
        ]
    ) + "\n"


def write_package_manifest(
    *,
    package_dir: Path,
    version: str,
    release_status: str,
    files: list[Path],
    ctrl_dir: Path,
    verification_dir: Path,
) -> dict[str, Any]:
    manifest = {
        "generated_at_utc": now_iso(),
        "version": version,
        "release_status": release_status,
        "ctrl_dir": serialise_path(ctrl_dir),
        "verification_dir": serialise_path(verification_dir),
        "files": [
            {
                "file": path.name,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(files, key=lambda entry: entry.name)
        ],
    }
    write_json(package_dir / "manifest.json", manifest)
    return manifest


def build_clinician_package(
    *,
    ctrl_dir: Path,
    verification_dir: Path,
    output_dir: Path,
    preflight_report_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[Path] = []
    for name in (
        REVIEW_FILENAMES["study_a"],
        REVIEW_FILENAMES["study_a_bias"],
        REVIEW_FILENAMES["study_b_single"],
        REVIEW_FILENAMES["study_b_multi"],
        REVIEW_FILENAMES["study_c"],
        REVIEW_FILENAMES["summary"],
        REVIEW_FILENAMES["metadata"],
    ):
        src = verification_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing review artefact: {src}")
        dst = output_dir / name
        _copy_file(src, dst)
        copied_files.append(dst)

    gate_src = verification_dir / "stage2_gate_report.json"
    if not gate_src.exists():
        raise FileNotFoundError(f"Missing gate report: {gate_src}")
    gate_dst = output_dir / gate_src.name
    _copy_file(gate_src, gate_dst)
    copied_files.append(gate_dst)
    gate_report = load_json(gate_dst)

    build_manifest_dst = output_dir / "ctrl_build_manifest.json"
    _copy_file(ctrl_dir / "build_manifest.json", build_manifest_dst)
    copied_files.append(build_manifest_dst)

    for source_name, target_name in (
        ("ctrl_gold_diagnosis_labels.json", "ctrl_gold_diagnosis_meta.json"),
        ("ctrl_gold_diagnosis_labels.robust.json", "ctrl_gold_diagnosis_robust_meta.json"),
        ("ctrl_target_plans.json", "ctrl_target_plans_meta.json"),
        ("ctrl_target_plans.robust.json", "ctrl_target_plans_robust_meta.json"),
    ):
        target_path = output_dir / target_name
        write_json(target_path, _gold_meta_summary(ctrl_dir, source_name))
        copied_files.append(target_path)

    preflight_path = output_dir / "sendoff_preflight_report.json"
    if preflight_report_payload is None:
        preflight_report_payload = {
            "generated_at_utc": now_iso(),
            "version": "controllability_v0.1_large_resolved_preflight",
            "status": "pending",
            "overall_passed": False,
            "checks": [],
        }
    write_json(preflight_path, preflight_report_payload)
    copied_files.append(preflight_path)

    review_rows = review_summary_rows(output_dir / REVIEW_FILENAMES["summary"])
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        _package_readme(
            ctrl_dir=ctrl_dir,
            verification_dir=verification_dir,
            gate_report=gate_report,
            review_rows=review_rows,
        ),
        encoding="utf-8",
    )
    copied_files.append(readme_path)

    blockers = review_blockers(review_rows)
    release_status = (
        "ready" if gate_report.get("overall_passed") and not blockers else "blocked"
    )
    manifest = write_package_manifest(
        package_dir=output_dir,
        version="controllability_v0.1_large_resolved",
        release_status=release_status,
        files=copied_files,
        ctrl_dir=ctrl_dir,
        verification_dir=verification_dir,
    )
    return {
        "release_status": release_status,
        "manifest": manifest,
        "files": [path.name for path in copied_files],
    }


def verify_package_manifest(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "manifest.json"
    if not manifest_path.exists():
        return {
            "passed": False,
            "issues": [f"missing_manifest:{manifest_path}"],
        }

    manifest = load_json(manifest_path)
    issues: list[str] = []

    for entry in manifest.get("files", []):
        file_name = entry.get("file", "")
        expected_sha = entry.get("sha256", "")
        file_path = package_dir / file_name
        if not file_path.exists():
            issues.append(f"missing_file:{file_name}")
            continue
        actual_sha = _sha256(file_path)
        if actual_sha != expected_sha:
            issues.append(f"sha_mismatch:{file_name}")

    return {
        "passed": not issues,
        "issues": issues,
    }
