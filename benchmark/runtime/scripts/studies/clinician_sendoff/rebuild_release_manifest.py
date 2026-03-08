#!/usr/bin/env python3
"""Regenerate or verify release manifest row counts and hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_DIR = ROOT / "data" / "releases" / "clinician_readiness_v0.3_2026-02-16"

JSON_ROW_KEYS = [
    "labels",
    "mapping",
    "cases",
    "samples",
    "plans",
    "case_evidence",
    "canonical_labels",
    "files",
    "gates",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def _count_json_rows(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(payload, list):
        return len(payload)

    if isinstance(payload, dict):
        for key in JSON_ROW_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
            if isinstance(value, dict):
                return len(value)
        return len(payload)

    return None


def _compute_row_count(path: Path) -> int | None:
    if path.suffix.lower() == ".json":
        return _count_json_rows(path)
    if path.suffix.lower() == ".csv":
        return _count_csv_rows(path)
    return None


def _discover_release_entries(release_dir: Path, ordered_paths: list[str]) -> list[dict]:
    files = sorted(
        p for p in release_dir.rglob("*") if p.is_file() and p.name != "manifest.json"
    )

    entries_by_path = {}
    for file_path in files:
        rel_path = file_path.relative_to(release_dir).as_posix()
        entries_by_path[rel_path] = {
            "path": rel_path,
            "sha256": _sha256(file_path),
            "row_count": _compute_row_count(file_path),
            "bytes": file_path.stat().st_size,
        }

    ordered_entries = []
    seen = set()
    for rel_path in ordered_paths:
        if rel_path in entries_by_path:
            ordered_entries.append(entries_by_path[rel_path])
            seen.add(rel_path)

    for rel_path in sorted(entries_by_path):
        if rel_path not in seen:
            ordered_entries.append(entries_by_path[rel_path])

    return ordered_entries


def _current_branch() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _build_manifest(release_dir: Path, existing_manifest: dict | None) -> dict:
    ordered_paths = []
    if existing_manifest:
        ordered_paths = [entry.get("path", "") for entry in existing_manifest.get("files", [])]

    entries = _discover_release_entries(release_dir, ordered_paths)

    dataset_release = (
        existing_manifest.get("dataset_release")
        if isinstance(existing_manifest, dict)
        else release_dir.name
    )
    source_branch = (
        existing_manifest.get("source_branch")
        if isinstance(existing_manifest, dict)
        else _current_branch()
    )

    return {
        "dataset_release": dataset_release,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_branch": source_branch,
        "files": entries,
    }


def _normalise_manifest(manifest: dict) -> dict:
    if not isinstance(manifest, dict):
        return {}
    files = manifest.get("files", [])
    by_path = {}
    for entry in files:
        if isinstance(entry, dict) and "path" in entry:
            by_path[entry["path"]] = {
                "sha256": entry.get("sha256"),
                "row_count": entry.get("row_count"),
                "bytes": entry.get("bytes"),
            }
    return by_path


def _verify(expected: dict, existing: dict) -> tuple[bool, list[str]]:
    expected_map = _normalise_manifest(expected)
    existing_map = _normalise_manifest(existing)

    mismatches = []

    expected_paths = set(expected_map)
    existing_paths = set(existing_map)

    missing = sorted(expected_paths - existing_paths)
    extra = sorted(existing_paths - expected_paths)
    if missing:
        mismatches.append(f"Missing manifest entries: {missing}")
    if extra:
        mismatches.append(f"Unexpected manifest entries: {extra}")

    for path in sorted(expected_paths & existing_paths):
        exp = expected_map[path]
        act = existing_map[path]
        for key in ("sha256", "row_count", "bytes"):
            if exp.get(key) != act.get(key):
                mismatches.append(
                    f"{path}: {key} expected={exp.get(key)!r} actual={act.get(key)!r}"
                )

    return not mismatches, mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild or verify release manifest metadata.")
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=DEFAULT_RELEASE_DIR,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Fail if the existing manifest does not match recomputed values.",
    )
    args = parser.parse_args()

    release_dir = args.release_dir
    manifest_path = args.manifest or (release_dir / "manifest.json")

    if not release_dir.exists():
        print(f"Release directory not found: {release_dir}")
        return 1

    existing_manifest = None
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_manifest = _build_manifest(release_dir, existing_manifest)

    if args.verify:
        if existing_manifest is None:
            print(f"Manifest not found for verification: {manifest_path}")
            return 1
        ok, mismatches = _verify(expected_manifest, existing_manifest)
        if ok:
            print("PASS: release manifest matches recomputed hashes and row counts.")
            return 0

        print("FAIL: release manifest mismatch detected.")
        for line in mismatches:
            print(f"  - {line}")
        return 1

    manifest_path.write_text(json.dumps(expected_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rebuilt manifest: {manifest_path}")
    print(f"Entries: {len(expected_manifest['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
