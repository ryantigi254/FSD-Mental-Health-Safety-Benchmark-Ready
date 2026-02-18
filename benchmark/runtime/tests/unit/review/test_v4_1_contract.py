"""Post-run contract checks for v4.1 resampled artefacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]

V03_ROOT = BASE_DIR / "data" / "frozen_splits" / "v0.3_postclinician_audit"
V41_ROOT = BASE_DIR / "data" / "frozen_splits" / "v4_1_resampled"

V4_OUT = BASE_DIR / "data" / "verification" / "v4"
V41_OUT = BASE_DIR / "data" / "verification" / "v4_1"

TARGET_COUNT = 768
VALID_VERDICTS = {"ACCEPTABLE", "NEEDS_REVIEW", "REJECT"}


def _read_json(path: Path):
    assert path.exists(), f"Missing file: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _read_ssv(path: Path) -> list[dict[str, str]]:
    assert path.exists(), f"Missing SSV output: {path}"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader)


def _sample_map(path: Path) -> dict[str, dict]:
    payload = _read_json(path)
    samples = payload.get("samples", []) if isinstance(payload, dict) else []
    assert isinstance(samples, list)
    return {str(sample.get("id", "") or ""): sample for sample in samples}


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.mark.unit
def test_v41_study_row_counts_match_expected() -> None:
    assert len(_read_ssv(V41_OUT / "study_a_reference_verdicts.ssv")) == 2000
    assert len(_read_ssv(V41_OUT / "study_b_single_verdicts.ssv")) == 2000
    assert len(_read_ssv(V41_OUT / "study_b_multi_verdicts.ssv")) == 120
    assert len(_read_ssv(V41_OUT / "study_c_verdicts.ssv")) == 100


@pytest.mark.unit
def test_v41_study_a_all_acceptable() -> None:
    rows = _read_ssv(V41_OUT / "study_a_reference_verdicts.ssv")
    verdicts = {str(row.get("verdict", "") or "") for row in rows}
    assert verdicts.issubset(VALID_VERDICTS)
    assert all(str(row.get("verdict", "") or "") == "ACCEPTABLE" for row in rows)


@pytest.mark.unit
def test_v41_study_b_and_c_remain_all_acceptable() -> None:
    for filename in ["study_b_single_verdicts.ssv", "study_b_multi_verdicts.ssv", "study_c_verdicts.ssv"]:
        rows = _read_ssv(V41_OUT / filename)
        assert all(str(row.get("verdict", "") or "") == "ACCEPTABLE" for row in rows)


@pytest.mark.unit
def test_v41_no_duplicate_item_ids() -> None:
    for filename in [
        "study_a_reference_verdicts.ssv",
        "study_b_single_verdicts.ssv",
        "study_b_multi_verdicts.ssv",
        "study_c_verdicts.ssv",
    ]:
        rows = _read_ssv(V41_OUT / filename)
        item_ids = [str(row.get("item_id", "") or "") for row in rows]
        assert len(item_ids) == len(set(item_ids)), f"Duplicate item IDs in {filename}"


@pytest.mark.unit
def test_v41_resampling_log_has_exact_target_count() -> None:
    rows = _read_ssv(V41_OUT / "study_a_resampling_log.ssv")
    assert len(rows) == TARGET_COUNT
    assert all(str(row.get("candidate_verdict", "") or "") == "ACCEPTABLE" for row in rows)


@pytest.mark.unit
def test_v41_study_a_labels_match_ids_exactly() -> None:
    study_a_payload = _read_json(V41_ROOT / "study_a_test.json")
    labels_payload = _read_json(V41_ROOT / "gold_diagnosis_labels.json")

    samples = study_a_payload.get("samples", [])
    labels = labels_payload.get("labels", {})

    assert isinstance(samples, list)
    assert isinstance(labels, dict)
    assert len(samples) == 2000
    assert len(labels) == 2000

    sample_ids = {str(sample.get("id", "") or "") for sample in samples}
    label_ids = set(labels.keys())

    assert sample_ids == label_ids
    assert all(str(labels[item_id] or "").strip() for item_id in sample_ids)


@pytest.mark.unit
def test_v41_replacement_and_retention_partition_is_exact() -> None:
    v4_rows = _read_ssv(V4_OUT / "study_a_reference_verdicts.ssv")
    target_ids = {
        str(row.get("item_id", "") or "")
        for row in v4_rows
        if str(row.get("verdict", "") or "") in {"NEEDS_REVIEW", "REJECT"}
    }
    assert len(target_ids) == TARGET_COUNT

    v03_samples = _sample_map(V03_ROOT / "study_a_test.json")
    v41_samples = _sample_map(V41_ROOT / "study_a_test.json")

    def canonical(sample: dict) -> str:
        return json.dumps(sample, sort_keys=True, ensure_ascii=False)

    changed = 0
    unchanged = 0
    for item_id, v03_sample in v03_samples.items():
        assert item_id in v41_samples, f"Missing item id in v4_1 Study A: {item_id}"
        same = canonical(v03_sample) == canonical(v41_samples[item_id])
        if item_id in target_ids:
            assert not same, f"Targeted item should be replaced but stayed unchanged: {item_id}"
            changed += 1
        else:
            assert same, f"Retained ACCEPTABLE item changed unexpectedly: {item_id}"
            unchanged += 1

    assert changed == TARGET_COUNT
    assert unchanged == 2000 - TARGET_COUNT


@pytest.mark.unit
def test_v41_no_reused_openr1_source_ids() -> None:
    study_a_payload = _read_json(V41_ROOT / "study_a_test.json")
    samples = study_a_payload.get("samples", [])
    assert isinstance(samples, list)

    source_refs: list[tuple[str, int]] = []
    for sample in samples:
        metadata = sample.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        source_split = str(metadata.get("source_split", "") or "").strip().lower()
        src = metadata.get("source_openr1_ids", [])
        if not isinstance(src, list):
            continue
        for value in src:
            if isinstance(value, int):
                source_refs.append((source_split, value))

    assert len(source_refs) == len(set(source_refs))

    replacement_rows = _read_ssv(V41_OUT / "study_a_resampling_log.ssv")
    replacement_ids = [int(str(row.get("candidate_openr1_id", "") or "-1")) for row in replacement_rows]
    assert len(replacement_ids) == len(set(replacement_ids))


@pytest.mark.unit
def test_v41_manifest_hash_matches_study_a_files() -> None:
    manifest = _read_json(V41_ROOT / "manifest.json")
    entries = {entry["file"]: entry for entry in manifest.get("files", [])}

    for rel in ["study_a_test.json", "gold_diagnosis_labels.json"]:
        assert rel in entries, f"Manifest missing {rel}"
        assert entries[rel]["sha256"] == _hash_file(V41_ROOT / rel)


@pytest.mark.unit
def test_v41_run_metadata_has_resampling_block() -> None:
    payload = _read_json(V41_OUT / "run_metadata.json")
    resampling = payload.get("resampling", {})
    assert isinstance(resampling, dict)
    assert int(resampling.get("targets", -1)) == TARGET_COUNT
    assert int(resampling.get("filled", -1)) == TARGET_COUNT
    assert int(resampling.get("exhausted", -1)) == 0


@pytest.mark.unit
def test_v41_manifest_created_at_is_stable() -> None:
    v03_manifest = _read_json(V03_ROOT / "manifest.json")
    v41_manifest = _read_json(V41_ROOT / "manifest.json")
    assert v41_manifest.get("created_at_utc") == v03_manifest.get("created_at_utc")
