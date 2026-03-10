"""Post-run contract checks for v4.1 resampled artefacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parents[3]

V41_ROOT = BASE_DIR / "data" / "frozen_splits" / "v4_1_resampled"
V41_OUT = BASE_DIR / "data" / "verification" / "v4_1"
LATEST_RELEASE_ROOT = BASE_DIR / "data" / "releases" / "clinician_readiness_v4_2026-02-22"

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


def _group_bias_by_slot(path: Path) -> dict[tuple[str, ...], list[dict]]:
    payload = _read_json(path)
    grouped: dict[str, list[dict]] = {}
    for case in payload.get("cases", []):
        grouped.setdefault(str(case.get("pair_group_id", "") or ""), []).append(case)

    by_slot: dict[tuple[str, ...], list[dict]] = {}
    for rows in grouped.values():
        slot = tuple(sorted(str(row.get("id", "") or "") for row in rows))
        by_slot[slot] = sorted(rows, key=lambda row: str(row.get("id", "") or ""))
    return by_slot


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
def test_v41_bias_replacement_log_has_exact_target_count() -> None:
    rows = _read_ssv(V41_OUT / "study_a_bias_replacement_log.ssv")
    assert len(rows) == 114


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
def test_v41_bias_retention_and_replacement_partition_is_exact() -> None:
    release_groups = _group_bias_by_slot(LATEST_RELEASE_ROOT / "adversarial_bias" / "biased_vignettes.json")
    current_groups = _group_bias_by_slot(V41_ROOT / "adversarial_bias" / "biased_vignettes.json")
    study_a_samples = _read_json(LATEST_RELEASE_ROOT / "openr1_psy_splits" / "study_a_test.json").get("samples", [])

    study_a_pairs = set()
    for sample in study_a_samples:
        metadata = sample.get("metadata", {}) or {}
        split_name = str(metadata.get("source_split", "") or "").strip().lower()
        if split_name not in {"test", "train"}:
            continue
        for source_id in metadata.get("source_openr1_ids", []) or []:
            study_a_pairs.add((split_name, int(source_id)))

    retained = 0
    replaced = 0
    for slot_key, old_rows in release_groups.items():
        new_rows = current_groups[slot_key]
        old_meta = old_rows[0].get("metadata", {}) or {}
        old_pair = (
            str(old_meta.get("source_openr1_split", "") or "").strip().lower(),
            int(old_meta.get("source_openr1_id", -1)),
        )
        if old_pair in study_a_pairs:
            replaced += 1
            assert json.dumps(old_rows, sort_keys=True, ensure_ascii=False) != json.dumps(
                new_rows, sort_keys=True, ensure_ascii=False
            )
        else:
            retained += 1
            assert json.dumps(old_rows, sort_keys=True, ensure_ascii=False) == json.dumps(
                new_rows, sort_keys=True, ensure_ascii=False
            )

    assert retained == 886
    assert replaced == 114


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

    replacement_rows = _read_ssv(V41_OUT / "study_a_bias_replacement_log.ssv")
    replacement_ids = [int(str(row.get("replacement_source_openr1_id", "") or "-1")) for row in replacement_rows]
    assert len(replacement_ids) == len(set(replacement_ids))


@pytest.mark.unit
def test_v41_manifest_hash_matches_study_a_files() -> None:
    manifest = _read_json(V41_ROOT / "manifest.json")
    entries = {entry["file"]: entry for entry in manifest.get("files", [])}

    for rel in [
        "study_a_test.json",
        "gold_diagnosis_labels.json",
        "study_b_test.json",
        "study_b_multi_turn_test.json",
        "adversarial_bias/biased_vignettes.json",
    ]:
        assert rel in entries, f"Manifest missing {rel}"
        assert entries[rel]["sha256"] == _hash_file(V41_ROOT / rel)


@pytest.mark.unit
def test_v41_run_metadata_has_resampling_block() -> None:
    payload = _read_json(V41_OUT / "refresh_run_metadata.json")
    study_a_bias = payload.get("study_a_bias", {})
    study_b = payload.get("study_b", {})
    study_c = payload.get("study_c", {})
    assert int(study_a_bias.get("replaced_groups", -1)) == 114
    assert int(study_a_bias.get("retained_groups", -1)) == 886
    assert int(study_b.get("single_turn_total", -1)) == 2000
    assert int(study_b.get("multi_turn_total", -1)) == 120
    assert int(study_c.get("replaced_cases", -1)) > 0


@pytest.mark.unit
def test_v41_cross_study_source_disjointness_report_is_clean() -> None:
    payload = _read_json(V41_OUT / "cross_study_source_disjointness.json")
    assert payload["study_a_bias_vs_main_overlap"] == []
    assert payload["study_b_multi_vs_study_a_overlap"] == []
    assert payload["study_b_multi_vs_study_a_bias_overlap"] == []
    assert payload["study_b_multi_vs_study_b_single_overlap"] == []
    assert payload["study_b_multi_vs_study_c_overlap"] == []
    assert payload["study_c_vs_study_a_overlap"] == []
    assert payload["study_c_vs_study_a_bias_overlap"] == []
    assert payload["study_c_vs_study_b_single_overlap"] == []
    assert payload["study_c_vs_study_b_multi_overlap"] == []
    assert payload["study_b_multi_internal_source_duplicates"] == {}
    assert payload["study_b_multi_duplicate_signatures"] == []


@pytest.mark.unit
def test_v41_manifest_created_at_is_stable() -> None:
    v41_manifest = _read_json(V41_ROOT / "manifest.json")
    assert str(v41_manifest.get("created_at_utc", "") or "").strip()
    assert v41_manifest.get("version") == "v4_1_resampled"
