#!/usr/bin/env python3
"""Deterministic v4.1 Study A resampling and full revalidation runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reliable_clinical_benchmark.review import (
    iter_openr1_candidates,
    load_rules,
    now_iso,
    score_study_a,
)


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = RUNTIME_ROOT.parents[1]
DATA_ROOT = RUNTIME_ROOT / "data"

DEFAULT_INPUT_ROOT = DATA_ROOT / "frozen_splits" / "v0.3_postclinician_audit"
DEFAULT_V4_VERIFICATION_DIR = DATA_ROOT / "verification" / "v4"
DEFAULT_V4_STUDY_A_SSV = DEFAULT_V4_VERIFICATION_DIR / "study_a_reference_verdicts.ssv"
DEFAULT_V4_RULES = DEFAULT_V4_VERIFICATION_DIR / "rubric_rules_v2.json"

DEFAULT_OUT_FROZEN = DATA_ROOT / "frozen_splits" / "v4_1_resampled"
DEFAULT_OUT_VERIFICATION = DATA_ROOT / "verification" / "v4_1"

REQUIRED_COMMITS = [
    "3cbc5ed",
    "bec71fc",
    "3f99a38",
    "fd3a955",
    "582c5f3",
    "c13caf1",
    "8fbe579",
]

VALID_VERDICTS = {"ACCEPTABLE", "NEEDS_REVIEW", "REJECT"}
TARGET_VERDICTS = {"NEEDS_REVIEW", "REJECT"}
TARGET_REPLACEMENT_COUNT = 768


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), check=False, capture_output=True, text=True)


def _run_checked(cmd: list[str], cwd: Path) -> str:
    proc = _run(cmd, cwd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc.stdout


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _payload_hash(payload: Any) -> str:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return _sha256_bytes(encoded)


def _normalise_items(payload: Any, preferred_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unable to normalise payload using keys={preferred_keys}")


def _read_study_a_targets(ssv_path: Path) -> tuple[list[str], dict[str, str]]:
    if not ssv_path.exists():
        raise RuntimeError(f"Missing Study A verdict SSV: {ssv_path}")

    targets: list[str] = []
    verdict_by_id: dict[str, str] = {}
    with ssv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            item_id = str(row.get("item_id", "") or "").strip()
            verdict = str(row.get("verdict", "") or "").strip()
            if not item_id:
                continue
            if verdict not in VALID_VERDICTS:
                raise RuntimeError(f"Invalid verdict for {item_id}: {verdict}")
            verdict_by_id[item_id] = verdict
            if verdict in TARGET_VERDICTS:
                targets.append(item_id)

    if len(targets) != TARGET_REPLACEMENT_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_REPLACEMENT_COUNT} Study A non-acceptable targets, found {len(targets)}"
        )

    return targets, verdict_by_id


def _extract_used_source_ids(samples: list[dict[str, Any]]) -> set[int]:
    used: set[int] = set()
    for sample in samples:
        metadata = sample.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        source_ids = metadata.get("source_openr1_ids", [])
        if not isinstance(source_ids, list):
            continue
        for source_id in source_ids:
            if isinstance(source_id, int):
                used.add(source_id)
    return used


def _build_resampled_study_a(
    *,
    study_a_samples: list[dict[str, Any]],
    labels: dict[str, str],
    target_ids_in_order: list[str],
    verdict_by_id: dict[str, str],
    rules: dict[str, Any],
) -> dict[str, Any]:
    used_source_ids = _extract_used_source_ids(study_a_samples)

    replacements: dict[str, dict[str, Any]] = {}
    replacement_labels: dict[str, str] = {}
    replacement_log_rows: list[dict[str, Any]] = []

    pool_scanned = 0
    pool_exhausted = False

    candidates = iter_openr1_candidates(used_source_ids=used_source_ids)

    for item_id in target_ids_in_order:
        while True:
            try:
                candidate = next(candidates)
            except StopIteration as exc:
                pool_exhausted = True
                raise RuntimeError(
                    "OpenR1 candidate pool exhausted before all targets were filled "
                    f"(filled={len(replacements)} target_total={len(target_ids_in_order)})"
                ) from exc

            pool_scanned += 1

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
                },
            }
            diagnosis_label = str(candidate.get("diagnosis_label", "") or "").strip()
            if not diagnosis_label:
                continue

            review = score_study_a(candidate_item, diagnosis_label, rules)
            if review["verdict"] != "ACCEPTABLE":
                continue

            replacements[item_id] = candidate_item
            replacement_labels[item_id] = diagnosis_label
            used_source_ids.add(int(candidate["openr1_id"]))

            replacement_log_rows.append(
                {
                    "replaced_item_id": item_id,
                    "previous_verdict": verdict_by_id[item_id],
                    "candidate_split": candidate["split"],
                    "candidate_openr1_id": candidate["openr1_id"],
                    "candidate_diagnosis_label": diagnosis_label,
                    "candidate_verdict": review["verdict"],
                    "candidate_reason_codes": "|".join(review.get("reason_codes", [])),
                    "pool_scan_position": pool_scanned,
                    "review_timestamp_utc": now_iso(),
                }
            )
            break

    rebuilt_samples: list[dict[str, Any]] = []
    for sample in study_a_samples:
        item_id = str(sample.get("id", "") or "").strip()
        if item_id in replacements:
            rebuilt_samples.append(replacements[item_id])
        else:
            rebuilt_samples.append(sample)

    rebuilt_labels = dict(labels)
    rebuilt_labels.update(replacement_labels)

    sample_ids = [str(item.get("id", "") or "").strip() for item in rebuilt_samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate Study A IDs detected after resampling")
    if len(rebuilt_samples) != 2000:
        raise RuntimeError(f"Expected 2000 Study A rows after resampling, found {len(rebuilt_samples)}")

    label_keys = set(rebuilt_labels.keys())
    sample_id_set = set(sample_ids)
    if label_keys != sample_id_set:
        missing = sorted(sample_id_set - label_keys)[:10]
        extra = sorted(label_keys - sample_id_set)[:10]
        raise RuntimeError(
            "Label key mismatch after resampling: "
            f"missing={missing} extra={extra}"
        )
    missing_labels = [item_id for item_id in sample_ids if not str(rebuilt_labels.get(item_id, "") or "").strip()]
    if missing_labels:
        raise RuntimeError(f"Missing/empty labels after resampling for IDs: {missing_labels[:10]}")

    retained_count = len(study_a_samples) - len(target_ids_in_order)

    return {
        "samples": rebuilt_samples,
        "labels": rebuilt_labels,
        "replacement_log_rows": replacement_log_rows,
        "targets": len(target_ids_in_order),
        "retained": retained_count,
        "filled": len(replacements),
        "pool_scanned": pool_scanned,
        "pool_exhausted": int(pool_exhausted),
    }


def _copy_baseline_tree(src: Path, dst: Path, clean: bool) -> None:
    if dst.exists() and clean:
        shutil.rmtree(dst)
    if dst.exists() and not clean:
        raise RuntimeError(f"Output snapshot already exists: {dst} (use --clean)")
    shutil.copytree(src, dst)


def _copy_rules_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_replacement_log(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "replaced_item_id",
        "previous_verdict",
        "candidate_split",
        "candidate_openr1_id",
        "candidate_diagnosis_label",
        "candidate_verdict",
        "candidate_reason_codes",
        "pool_scan_position",
        "review_timestamp_utc",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row_count_for_file(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _load_json(path)
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            for key in (
                "samples",
                "cases",
                "labels",
                "files",
                "rows",
                "items",
                "gates",
                "checks",
            ):
                value = payload.get(key)
                if isinstance(value, (list, dict)):
                    return len(value)
            return len(payload)
        return None
    if suffix in {".ssv", ".csv"}:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    return None


def _write_manifest(snapshot_root: Path, *, created_at_utc: str) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in sorted(p for p in snapshot_root.rglob("*") if p.is_file() and p.name != "manifest.json"):
        rel = path.relative_to(snapshot_root).as_posix()
        entries.append(
            {
                "file": rel,
                "sha256": _sha256_file(path),
                "row_count": _row_count_for_file(path),
            }
        )

    manifest = {
        "created_at_utc": created_at_utc,
        "version": "v4_1_resampled",
        "description": "Study A non-acceptable rows deterministically resampled from OpenR1-Psy and validated with v4 rubric v2 scorer.",
        "supersedes": "v0.3_postclinician_audit",
        "files": entries,
    }
    _write_json(snapshot_root / "manifest.json", manifest)
    return manifest


def _write_snapshot_readme(
    snapshot_root: Path,
    *,
    targets: int,
    retained: int,
    filled: int,
    pool_scanned: int,
) -> None:
    readme = "\n".join(
        [
            "# Frozen Snapshot v4.1 (Study A Resampled)",
            "",
            "## Scope",
            "- Baseline snapshot: `v0.3_postclinician_audit`.",
            "- Study A rows replaced: non-acceptable (`NEEDS_REVIEW` + `REJECT`) from `verification/v4/study_a_reference_verdicts.ssv`.",
            "- Study B/Study C retained unchanged from baseline.",
            "",
            "## Deterministic policy",
            "- Preserve Study A IDs and row order.",
            "- Retain all baseline `ACCEPTABLE` rows unchanged.",
            "- Replace only targeted rows using OpenR1-Psy candidate stream in split/index order (`train` then `test`).",
            "- Candidate acceptance authority: current in-repo `score_study_a` rubric-v2 scorer.",
            "- Exclude source IDs already used anywhere in baseline Study A and by earlier accepted replacements.",
            "",
            "## Result",
            f"- Targets: {targets}",
            f"- Retained Study A rows: {retained}",
            f"- Filled replacements: {filled}",
            f"- Candidate pool scanned: {pool_scanned}",
            "",
            "See `manifest.json` for checksums and row counts.",
        ]
    )
    (snapshot_root / "README.md").write_text(readme + "\n", encoding="utf-8")


def _git_head_summary() -> dict[str, str]:
    commit = _run_checked(["git", "rev-parse", "HEAD"], REPO_ROOT).strip()
    branch = _run_checked(["git", "rev-parse", "--abbrev-ref", "HEAD"], REPO_ROOT).strip()
    return {"commit": commit, "branch": branch}


def _collect_commit_metadata(commit: str) -> dict[str, Any]:
    _run_checked(["git", "merge-base", "--is-ancestor", commit, "HEAD"], REPO_ROOT)

    subject = _run_checked(["git", "show", "--no-patch", "--format=%s", commit], REPO_ROOT).strip()
    authored = _run_checked(
        ["git", "show", "--no-patch", "--format=%ad", "--date=iso-strict", commit],
        REPO_ROOT,
    ).strip()
    files_out = _run_checked(["git", "show", "--pretty=format:", "--name-only", commit], REPO_ROOT)
    files = [line.strip() for line in files_out.splitlines() if line.strip()]

    return {
        "commit": commit,
        "subject": subject,
        "authored_at": authored,
        "files": files,
    }


def _write_history_audit(path: Path, required_commits: list[str]) -> list[dict[str, Any]]:
    head = _git_head_summary()
    rows = [_collect_commit_metadata(commit) for commit in required_commits]

    lines = [
        "# V4.1 History Audit",
        "",
        "## Scope",
        "- Required lineage commits are present in reachable history from current HEAD.",
        f"- Current branch: `{head['branch']}`",
        f"- Current HEAD: `{head['commit']}`",
        "",
        "## Required Commit Checks",
    ]

    for row in rows:
        lines.extend(
            [
                f"### {row['commit']}",
                f"- Subject: {row['subject']}",
                f"- Authored: {row['authored_at']}",
                "- Touched files:",
            ]
        )
        if row["files"]:
            lines.extend([f"  - `{file}`" for file in row["files"]])
        else:
            lines.append("  - *(no file changes)*")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return rows


def _run_validation_chain(runtime_root: Path) -> list[dict[str, Any]]:
    commands = [
        [sys.executable, "-m", "pytest", "tests/unit/data/test_frozen_snapshot_v03_manifest.py", "-q"],
        [sys.executable, "-m", "pytest", "tests/unit/data/test_clinician_package_v03_manifest.py", "-q"],
        [sys.executable, "scripts/studies/clinician_sendoff/run_stage2_gates.py"],
        [sys.executable, "scripts/studies/clinician_sendoff/run_sendoff_preflight.py"],
    ]

    executed: list[dict[str, Any]] = []
    for cmd in commands:
        proc = _run(cmd, runtime_root)
        record = {
            "command": " ".join(cmd),
            "exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
        executed.append(record)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Validation chain failed: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
    return executed


def _run_v4_review(input_root: Path, out_dir: Path, rules_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/studies/v4_review/run_v4_cross_study_review.py",
        "--clean",
        "--input-root",
        str(input_root),
        "--out-dir",
        str(out_dir),
        "--rules",
        str(rules_path),
    ]
    proc = _run(cmd, RUNTIME_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"v4 review runner failed\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return {
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _summarise_study_verdicts(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = list(reader)
    verdicts = Counter(str(row.get("verdict", "") or "").strip() for row in rows)
    return {
        "rows": len(rows),
        "acceptable": int(verdicts.get("ACCEPTABLE", 0)),
        "needs_review": int(verdicts.get("NEEDS_REVIEW", 0)),
        "reject": int(verdicts.get("REJECT", 0)),
    }


def _enforce_v4_1_quality_gate(verification_root: Path) -> dict[str, dict[str, int]]:
    summary = {
        "study_a": _summarise_study_verdicts(verification_root / "study_a_reference_verdicts.ssv"),
        "study_b_single": _summarise_study_verdicts(verification_root / "study_b_single_verdicts.ssv"),
        "study_b_multi": _summarise_study_verdicts(verification_root / "study_b_multi_verdicts.ssv"),
        "study_c": _summarise_study_verdicts(verification_root / "study_c_verdicts.ssv"),
    }

    if summary["study_a"]["rows"] != 2000:
        raise RuntimeError("Study A row count mismatch in v4_1 verification output")
    if summary["study_a"]["acceptable"] != 2000:
        raise RuntimeError(f"Study A quality gate failed: {summary['study_a']}")
    if summary["study_a"]["needs_review"] != 0 or summary["study_a"]["reject"] != 0:
        raise RuntimeError(f"Study A must be all ACCEPTABLE in v4_1: {summary['study_a']}")

    for study, expected_rows in (("study_b_single", 2000), ("study_b_multi", 120), ("study_c", 100)):
        if summary[study]["rows"] != expected_rows:
            raise RuntimeError(f"{study} row-count mismatch: {summary[study]}")
        if summary[study]["needs_review"] != 0 or summary[study]["reject"] != 0:
            raise RuntimeError(f"{study} verdict regression: {summary[study]}")

    return summary


def _write_v4_1_notes(
    path: Path,
    *,
    before_counts: dict[str, int],
    after_counts: dict[str, dict[str, int]],
    resampling: dict[str, Any],
    validation_records: list[dict[str, Any]],
    runner_record: dict[str, Any],
) -> None:
    lines = [
        "# V4.1 Update Notes",
        "",
        "## Summary",
        "- Built deterministic Study A resampling from OpenR1-Psy using current in-repo v4 scorer behaviour.",
        "- Replaced non-acceptable Study A rows from v4 baseline outputs.",
        "- Re-ran full v4 review on `frozen_splits/v4_1_resampled` into `verification/v4_1`.",
        "",
        "## Before vs After (Study A)",
        f"- Before (`verification/v4`): ACCEPTABLE={before_counts['acceptable']}, NEEDS_REVIEW={before_counts['needs_review']}, REJECT={before_counts['reject']}",
        f"- After (`verification/v4_1`): ACCEPTABLE={after_counts['study_a']['acceptable']}, NEEDS_REVIEW={after_counts['study_a']['needs_review']}, REJECT={after_counts['study_a']['reject']}",
        "",
        "## Resampling Provenance",
        f"- Targets: {resampling['targets']}",
        f"- Filled: {resampling['filled']}",
        f"- Retained unchanged: {resampling['retained']}",
        f"- Candidate pool scanned: {resampling['pool_scanned']}",
        f"- Pool exhausted: {resampling['exhausted']}",
        "",
        "## Command Audit",
    ]
    for record in validation_records:
        lines.append(f"- `{record['command']}` -> exit {record['exit_code']}")
    lines.append(f"- `{runner_record['command']}` -> exit {runner_record['exit_code']}")

    lines.extend(
        [
            "",
            "## Final Distribution",
            "| Study | Rows | ACCEPTABLE | NEEDS_REVIEW | REJECT |",
            "|---|---:|---:|---:|---:|",
            f"| study_a | {after_counts['study_a']['rows']} | {after_counts['study_a']['acceptable']} | {after_counts['study_a']['needs_review']} | {after_counts['study_a']['reject']} |",
            f"| study_b_single | {after_counts['study_b_single']['rows']} | {after_counts['study_b_single']['acceptable']} | {after_counts['study_b_single']['needs_review']} | {after_counts['study_b_single']['reject']} |",
            f"| study_b_multi | {after_counts['study_b_multi']['rows']} | {after_counts['study_b_multi']['acceptable']} | {after_counts['study_b_multi']['needs_review']} | {after_counts['study_b_multi']['reject']} |",
            f"| study_c | {after_counts['study_c']['rows']} | {after_counts['study_c']['acceptable']} | {after_counts['study_c']['needs_review']} | {after_counts['study_c']['reject']} |",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_before_study_a_counts(v4_ssv: Path) -> dict[str, int]:
    with v4_ssv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = list(reader)
    verdicts = Counter(str(row.get("verdict", "") or "").strip() for row in rows)
    return {
        "rows": len(rows),
        "acceptable": int(verdicts.get("ACCEPTABLE", 0)),
        "needs_review": int(verdicts.get("NEEDS_REVIEW", 0)),
        "reject": int(verdicts.get("REJECT", 0)),
    }


def _augment_run_metadata(path: Path, *, resampling: dict[str, Any]) -> None:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid run_metadata format: {path}")

    payload["resampling"] = {
        "targets": int(resampling["targets"]),
        "filled": int(resampling["filled"]),
        "pool_scanned": int(resampling["pool_scanned"]),
        "exhausted": int(resampling["exhausted"]),
        "retained": int(resampling["retained"]),
    }
    payload["last_run_utc"] = now_iso()
    _write_json(path, payload)


def _clear_verification_out(path: Path, clean: bool) -> None:
    if path.exists() and clean:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic v4.1 Study A resampling and revalidation")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--v4-study-a-ssv", type=Path, default=DEFAULT_V4_STUDY_A_SSV)
    parser.add_argument("--rules", type=Path, default=DEFAULT_V4_RULES)
    parser.add_argument("--out-frozen", type=Path, default=DEFAULT_OUT_FROZEN)
    parser.add_argument("--out-verification", type=Path, default=DEFAULT_OUT_VERIFICATION)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-validation-chain", action="store_true")
    parser.add_argument("--skip-v4-run", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root
    v4_ssv = args.v4_study_a_ssv
    rules_path = args.rules
    out_frozen = args.out_frozen
    out_verification = args.out_verification

    if not input_root.exists():
        raise RuntimeError(f"Missing input root: {input_root}")
    if not rules_path.exists():
        raise RuntimeError(f"Missing rules file: {rules_path}")

    rules = load_rules(rules_path)

    _clear_verification_out(out_verification, clean=args.clean)

    history_path = out_verification / "V4_1_HISTORY_AUDIT.md"
    _write_history_audit(history_path, REQUIRED_COMMITS)

    target_ids, verdict_by_id = _read_study_a_targets(v4_ssv)

    study_a_payload = _load_json(input_root / "study_a_test.json")
    study_a_samples = _normalise_items(study_a_payload, ("samples", "items", "cases"))

    labels_payload = _load_json(input_root / "gold_diagnosis_labels.json")
    labels = labels_payload.get("labels", {}) if isinstance(labels_payload, dict) else {}
    if not isinstance(labels, dict):
        raise RuntimeError("gold_diagnosis_labels.json must include a labels object")

    resampled_once = _build_resampled_study_a(
        study_a_samples=study_a_samples,
        labels=labels,
        target_ids_in_order=target_ids,
        verdict_by_id=verdict_by_id,
        rules=rules,
    )

    resampled_twice = _build_resampled_study_a(
        study_a_samples=study_a_samples,
        labels=labels,
        target_ids_in_order=target_ids,
        verdict_by_id=verdict_by_id,
        rules=rules,
    )

    study_a_meta = study_a_payload.get("meta", {}) if isinstance(study_a_payload, dict) else {}
    if not isinstance(study_a_meta, dict):
        study_a_meta = {}

    study_a_out_payload = {
        "samples": resampled_once["samples"],
        "meta": {
            **study_a_meta,
            "snapshot_version": "v4_1_resampled",
            "resampling_policy": "replace_non_acceptable_only",
            "resampling_targets": resampled_once["targets"],
        },
    }
    labels_meta = labels_payload.get("meta", {}) if isinstance(labels_payload, dict) else {}
    if not isinstance(labels_meta, dict):
        labels_meta = {}

    labels_out_payload = {
        "labels": resampled_once["labels"],
        "meta": {
            **labels_meta,
            "snapshot_version": "v4_1_resampled",
            "resampling_targets": resampled_once["targets"],
        },
    }

    study_a_second_payload = {
        "samples": resampled_twice["samples"],
        "meta": {
            **study_a_meta,
            "snapshot_version": "v4_1_resampled",
            "resampling_policy": "replace_non_acceptable_only",
            "resampling_targets": resampled_twice["targets"],
        },
    }
    labels_second_payload = {
        "labels": resampled_twice["labels"],
        "meta": {
            **labels_meta,
            "snapshot_version": "v4_1_resampled",
            "resampling_targets": resampled_twice["targets"],
        },
    }

    determinism_ok = (
        _payload_hash(study_a_out_payload) == _payload_hash(study_a_second_payload)
        and _payload_hash(labels_out_payload) == _payload_hash(labels_second_payload)
    )
    if not determinism_ok:
        raise RuntimeError("Determinism check failed: second in-memory resampling run diverged")

    _copy_baseline_tree(input_root, out_frozen, clean=args.clean)

    _write_json(out_frozen / "study_a_test.json", study_a_out_payload)
    _write_json(out_frozen / "gold_diagnosis_labels.json", labels_out_payload)

    study_a_dir = out_frozen / "study_a"
    if study_a_dir.exists():
        _write_json(study_a_dir / "study_a_test.json", study_a_out_payload)
        _write_json(study_a_dir / "gold_diagnosis_labels.json", labels_out_payload)

    _write_snapshot_readme(
        out_frozen,
        targets=resampled_once["targets"],
        retained=resampled_once["retained"],
        filled=resampled_once["filled"],
        pool_scanned=resampled_once["pool_scanned"],
    )
    source_manifest = _load_json(input_root / "manifest.json")
    created_at_utc = str(source_manifest.get("created_at_utc", "") if isinstance(source_manifest, dict) else "")
    if not created_at_utc:
        created_at_utc = "1970-01-01T00:00:00+00:00"
    _write_manifest(out_frozen, created_at_utc=created_at_utc)

    rules_out_path = out_verification / "rubric_rules_v2.json"
    _copy_rules_file(rules_path, rules_out_path)

    replacement_log_path = out_verification / "study_a_resampling_log.ssv"
    _write_replacement_log(replacement_log_path, resampled_once["replacement_log_rows"])

    validation_records: list[dict[str, Any]] = []
    if not args.skip_validation_chain:
        validation_records = _run_validation_chain(RUNTIME_ROOT)

    runner_record = {
        "command": "(skipped)",
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
    }
    after_counts: dict[str, dict[str, int]] = {
        "study_a": {"rows": 0, "acceptable": 0, "needs_review": 0, "reject": 0},
        "study_b_single": {"rows": 0, "acceptable": 0, "needs_review": 0, "reject": 0},
        "study_b_multi": {"rows": 0, "acceptable": 0, "needs_review": 0, "reject": 0},
        "study_c": {"rows": 0, "acceptable": 0, "needs_review": 0, "reject": 0},
    }

    if not args.skip_v4_run:
        runner_record = _run_v4_review(out_frozen, out_verification, rules_out_path)
        after_counts = _enforce_v4_1_quality_gate(out_verification)

        resampling_metadata = {
            "targets": resampled_once["targets"],
            "filled": resampled_once["filled"],
            "pool_scanned": resampled_once["pool_scanned"],
            "exhausted": resampled_once["pool_exhausted"],
            "retained": resampled_once["retained"],
        }
        _augment_run_metadata(out_verification / "run_metadata.json", resampling=resampling_metadata)

    before_counts = _read_before_study_a_counts(v4_ssv)

    _write_v4_1_notes(
        out_verification / "V4_1_UPDATE_NOTES.md",
        before_counts=before_counts,
        after_counts=after_counts,
        resampling={
            "targets": resampled_once["targets"],
            "retained": resampled_once["retained"],
            "filled": resampled_once["filled"],
            "pool_scanned": resampled_once["pool_scanned"],
            "exhausted": resampled_once["pool_exhausted"],
        },
        validation_records=validation_records,
        runner_record=runner_record,
    )

    if not args.skip_v4_run:
        print("study | rows | acceptable | needs_review | reject")
        print("----- | ---- | ---------- | ------------ | ------")
        for study in ("study_a", "study_b_single", "study_b_multi", "study_c"):
            row = after_counts[study]
            print(
                f"{study} | {row['rows']} | {row['acceptable']} | "
                f"{row['needs_review']} | {row['reject']}"
            )

    print(
        "v4.1 resampling complete: "
        f"targets={resampled_once['targets']} filled={resampled_once['filled']} "
        f"pool_scanned={resampled_once['pool_scanned']} determinism_ok={determinism_ok}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
