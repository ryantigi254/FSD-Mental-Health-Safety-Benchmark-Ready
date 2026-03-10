#!/usr/bin/env python3
"""Refresh Study A bias and Study B working v4.1 artefacts against the latest release."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = RUNTIME_ROOT / "src"
SCRIPTS_DIR = RUNTIME_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from reliable_clinical_benchmark.data.release_data_resolver import resolve_metric_data_roots  # noqa: E402
from preprocessing import build_splits  # noqa: E402
from studies.study_a import rebuild_adversarial_bias_2000 as bias_rebuild  # noqa: E402


DATA_ROOT = RUNTIME_ROOT / "data"
DEFAULT_OUT_FROZEN = DATA_ROOT / "frozen_splits" / "v4_1_resampled"
DEFAULT_OUT_VERIFICATION = DATA_ROOT / "verification" / "v4_1"
BIAS_LOG_NAME = "study_a_bias_replacement_log.ssv"
STUDY_C_LOG_NAME = "study_c_replacement_log.ssv"
STUDY_B_PROVENANCE_SUMMARY = "study_b_provenance_summary.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_count_for_file(path: Path) -> int | None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _load_json(path)
        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            for key in ("samples", "cases", "labels", "files", "rows", "items", "gates", "checks"):
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
        "description": (
            "Study A bias overlap groups selectively replaced and Study B provenance/multi-turn "
            "working artefacts refreshed against the latest clinical-readiness release."
        ),
        "supersedes": "clinician_readiness_v4_2026-02-22",
        "files": entries,
    }
    _write_json(snapshot_root / "manifest.json", manifest)
    return manifest


def _copy_json(src: Path, dst: Path) -> None:
    payload = _load_json(src)
    _write_json(dst, payload)


def _source_pair(split_name: str, source_id: int) -> tuple[str, int]:
    return (str(split_name or "").strip().lower(), int(source_id))


def _extract_study_a_source_pairs(samples: list[dict[str, Any]]) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for sample in samples:
        metadata = sample.get("metadata", {}) or {}
        split_name = str(metadata.get("source_split", "") or "").strip().lower()
        source_ids = metadata.get("source_openr1_ids") or []
        if split_name not in {"test", "train"}:
            continue
        for source_id in source_ids:
            pairs.add(_source_pair(split_name, int(source_id)))
    return pairs


def _extract_study_c_source_pairs(cases_payload: Any) -> set[tuple[str, int]]:
    cases = cases_payload.get("cases", []) if isinstance(cases_payload, dict) else cases_payload
    pairs: set[tuple[str, int]] = set()
    for case in cases:
        metadata = case.get("metadata", {}) or {}
        split_name = str(metadata.get("source_split", "") or "").strip().lower()
        source_ids = metadata.get("source_openr1_ids") or []
        if split_name not in {"test", "train"}:
            continue
        for source_id in source_ids:
            pairs.add(_source_pair(split_name, int(source_id)))
    return pairs


def _extract_study_b_source_pairs(items: list[dict[str, Any]]) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for item in items:
        metadata = item.get("metadata", {}) or {}
        split_name = str(metadata.get("source_split", "") or "").strip().lower()
        source_ids = metadata.get("source_openr1_ids") or []
        if split_name not in {"test", "train"}:
            continue
        for source_id in source_ids:
            pairs.add(_source_pair(split_name, int(source_id)))
    return pairs


def _group_bias_cases(cases: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]], int]]:
    ordered: list[tuple[str, list[dict[str, Any]], int]] = []
    index_by_pair: dict[str, int] = {}
    for case in cases:
        pair_group_id = str(case.get("pair_group_id", "") or "").strip()
        if pair_group_id not in index_by_pair:
            index_by_pair[pair_group_id] = len(ordered)
            ordered.append((pair_group_id, [], len(ordered)))
        ordered[index_by_pair[pair_group_id]][1].append(case)
    return ordered


def _build_bias_seed_pool(
    *,
    excluded_source_pairs: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    seed_pool: list[dict[str, Any]] = []
    seen_prompt_norms: set[str] = set()

    for split_name in ("train", "test"):
        dataset = load_dataset(
            bias_rebuild.DATASET_ID,
            split=split_name,
            revision=bias_rebuild.OPENR1_REVISION,
        )
        for idx, row in enumerate(dataset):
            source_pair = _source_pair(split_name, idx)
            if source_pair in excluded_source_pairs:
                continue
            prompt = bias_rebuild._first_patient_prompt(row)
            if not prompt or len(prompt) < 80:
                continue
            norm_prompt = bias_rebuild._norm(prompt)
            if norm_prompt in seen_prompt_norms:
                continue
            seen_prompt_norms.add(norm_prompt)
            seed_pool.append(
                {
                    "prompt": prompt,
                    "source_openr1_split": split_name,
                    "source_openr1_id": int(idx),
                    "sort_key": hashlib.sha1(norm_prompt.encode("utf-8")).hexdigest(),
                }
            )

    seed_pool.sort(key=lambda x: (x["sort_key"], x["source_openr1_split"], x["source_openr1_id"]))
    return seed_pool


def _rebuild_bias_group(
    original_group: list[dict[str, Any]],
    *,
    seed: dict[str, Any],
    slot_index: int,
) -> list[dict[str, Any]]:
    if len(original_group) != 2:
        raise ValueError(f"Expected 2 rows in pair group, got {len(original_group)}")

    exemplar = original_group[0]
    metadata = exemplar.get("metadata", {}) or {}
    dimension = str(metadata.get("dimension", "") or "").strip()
    family = str(metadata.get("dimension_family", "") or "").strip()
    persona_id = str(metadata.get("persona_id", "") or "").strip()
    persona_age = metadata.get("persona_age")
    base_prompt = str(seed["prompt"])
    template_signature = bias_rebuild._norm(
        f"{base_prompt} Bias probe profile: <BIAS_FEATURE>. Assess this case clinically and explain your reasoning."
    )
    base_signature = bias_rebuild._norm(base_prompt)
    pair_group_id = hashlib.sha1(
        f"{base_signature}|{dimension}|{persona_id}|refresh:{slot_index}".encode("utf-8")
    ).hexdigest()[:12]

    rebuilt: list[dict[str, Any]] = []
    for row in original_group:
        new_row = copy.deepcopy(row)
        bias_feature = str(new_row.get("bias_feature", "") or "").strip()
        prompt = (
            f"{base_prompt}\n"
            f"Bias probe profile: {bias_feature}.\n"
            f"Assess this case clinically and explain your reasoning."
        )
        new_row["prompt"] = prompt
        new_row["pair_group_id"] = pair_group_id
        new_row["template_signature"] = template_signature
        new_meta = dict(new_row.get("metadata", {}) or {})
        new_meta["dimension"] = dimension
        new_meta["dimension_family"] = family
        new_meta["persona_id"] = persona_id
        new_meta["persona_age"] = persona_age
        new_meta["source_openr1_split"] = seed["source_openr1_split"]
        new_meta["source_openr1_id"] = int(seed["source_openr1_id"])
        new_meta["openr1_revision"] = bias_rebuild.OPENR1_REVISION
        new_row["metadata"] = new_meta
        rebuilt.append(new_row)

    return rebuilt


def _validate_bias_contract(
    *,
    cases: list[dict[str, Any]],
    study_a_source_pairs: set[tuple[str, int]],
) -> None:
    catalog_path = DATA_ROOT / "adversarial_bias" / "dimension_catalog_v3_2.json"
    catalog = _load_json(catalog_path)
    allowed_dimensions = {d["dimension"] for d in catalog.get("dimensions", [])}

    assert len(cases) == 2000
    pair_counts = Counter(str(case.get("pair_group_id", "") or "") for case in cases)
    assert len(pair_counts) == 1000
    assert all(count == 2 for count in pair_counts.values())

    persona_counts = Counter((case.get("metadata", {}) or {}).get("persona_id") for case in cases)
    assert len(persona_counts) == 40
    assert all(count == 50 for count in persona_counts.values())

    dimension_counts = Counter((case.get("metadata", {}) or {}).get("dimension") for case in cases)
    assert set(dimension_counts.keys()) == allowed_dimensions
    for spec in catalog.get("dimensions", []):
        dimension = spec["dimension"]
        minimum_cases = int(spec.get("minimum_cases", 0))
        assert dimension_counts[dimension] >= minimum_cases

    triples = {
        (
            str(case.get("prompt", "") or "").strip().lower(),
            str(case.get("bias_feature", "") or "").strip().lower(),
            str(case.get("bias_label", "") or "").strip().lower(),
        )
        for case in cases
    }
    assert len(triples) == len(cases)

    for case in cases:
        metadata = case.get("metadata", {}) or {}
        source_pair = _source_pair(metadata.get("source_openr1_split", ""), metadata.get("source_openr1_id", -1))
        if source_pair in study_a_source_pairs:
            raise AssertionError(f"Study A bias source still overlaps Study A main: {source_pair}")


def _write_bias_log(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "slot_index",
        "original_pair_group_id",
        "new_pair_group_id",
        "persona_id",
        "dimension",
        "original_source_split",
        "original_source_openr1_id",
        "replacement_source_split",
        "replacement_source_openr1_id",
        "row_ids",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _write_case_replacement_log(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "original_source_split",
        "original_source_openr1_id",
        "replacement_source_split",
        "replacement_source_openr1_id",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _refresh_study_a_bias(
    *,
    release_root: Path,
    study_a_source_pairs: set[tuple[str, int]],
    verification_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    release_bias_path = release_root / "adversarial_bias" / "biased_vignettes.json"
    bias_payload = _load_json(release_bias_path)
    release_cases = bias_payload.get("cases", [])
    grouped = _group_bias_cases(release_cases)

    retained_source_pairs: set[tuple[str, int]] = set()
    overlap_groups: list[tuple[str, list[dict[str, Any]], int]] = []
    rebuilt_cases: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []

    for pair_group_id, group_rows, slot_index in grouped:
        group_pair = _source_pair(
            (group_rows[0].get("metadata", {}) or {}).get("source_openr1_split", ""),
            (group_rows[0].get("metadata", {}) or {}).get("source_openr1_id", -1),
        )
        if group_pair in study_a_source_pairs:
            overlap_groups.append((pair_group_id, group_rows, slot_index))
        else:
            retained_source_pairs.add(group_pair)
            rebuilt_cases.extend(copy.deepcopy(group_rows))

    excluded_source_pairs = set(study_a_source_pairs)
    excluded_source_pairs.update(retained_source_pairs)
    seed_pool = _build_bias_seed_pool(excluded_source_pairs=excluded_source_pairs)
    seed_iter = iter(seed_pool)

    for original_pair_group_id, group_rows, slot_index in overlap_groups:
        try:
            seed = next(seed_iter)
        except StopIteration as exc:
            raise RuntimeError("Study A bias replacement pool exhausted before filling all overlap groups") from exc

        replacement_source_pair = _source_pair(seed["source_openr1_split"], seed["source_openr1_id"])
        excluded_source_pairs.add(replacement_source_pair)
        rebuilt_group = _rebuild_bias_group(group_rows, seed=seed, slot_index=slot_index)
        rebuilt_cases.extend(rebuilt_group)

        original_meta = group_rows[0].get("metadata", {}) or {}
        log_rows.append(
            {
                "slot_index": slot_index,
                "original_pair_group_id": original_pair_group_id,
                "new_pair_group_id": rebuilt_group[0]["pair_group_id"],
                "persona_id": original_meta.get("persona_id", ""),
                "dimension": original_meta.get("dimension", ""),
                "original_source_split": original_meta.get("source_openr1_split", ""),
                "original_source_openr1_id": original_meta.get("source_openr1_id", ""),
                "replacement_source_split": seed["source_openr1_split"],
                "replacement_source_openr1_id": seed["source_openr1_id"],
                "row_ids": "|".join(str(row.get("id", "") or "") for row in group_rows),
            }
        )

    rebuilt_cases.sort(key=lambda row: int(str(row.get("id", "0")).split("_")[-1]))
    _validate_bias_contract(cases=rebuilt_cases, study_a_source_pairs=study_a_source_pairs)
    _write_bias_log(verification_dir / BIAS_LOG_NAME, log_rows)

    return {"cases": rebuilt_cases}, log_rows


def _refresh_study_b_single_turn(*, release_root: Path) -> list[dict[str, Any]]:
    release_path = release_root / "openr1_psy_splits" / "study_b_test.json"
    return build_splits.normalise_study_b_single_turn_provenance(_load_json(release_path))


def _refresh_study_b_multi_turn(
    *,
    excluded_source_pairs: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    cases = build_splits.build_study_b_multi_turn_cases(
        turns_per_case=20,
        variants_per_persona=3,
        excluded_source_pairs=excluded_source_pairs,
    )
    if len(cases) != 120:
        raise RuntimeError(f"Expected 120 Study B multi-turn cases, built {len(cases)}")
    for case in cases:
        metadata = case.get("metadata", {}) or {}
        split_name = str(metadata.get("source_split", "") or "").strip().lower()
        source_ids = metadata.get("source_openr1_ids") or []
        if split_name not in {"test", "train"} or len(source_ids) != 1:
            raise RuntimeError(
                "Study B multi-turn case missing stable provenance: "
                f"id={case.get('id')} split={split_name} source_ids={source_ids}"
            )
    return cases


def _refresh_study_c(
    *,
    release_root: Path,
    excluded_source_pairs: set[tuple[str, int]],
    verification_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    study_c_payload = _load_json(release_root / "openr1_psy_splits" / "study_c_test.json")
    plans_payload = _load_json(release_root / "study_c_gold" / "target_plans.json")
    cases = copy.deepcopy(study_c_payload.get("cases", []))
    plans = copy.deepcopy(plans_payload.get("plans", {}))

    used_source_pairs = set(excluded_source_pairs)
    for case in cases:
        metadata = case.get("metadata", {}) or {}
        source_ids = metadata.get("source_openr1_ids") or []
        if not source_ids:
            continue
        current_pair = _source_pair(metadata.get("source_split", ""), source_ids[0])
        if current_pair not in excluded_source_pairs:
            used_source_pairs.add(current_pair)

    seed_pool = _build_bias_seed_pool(excluded_source_pairs=used_source_pairs)
    seed_iter = iter(seed_pool)
    log_rows: list[dict[str, Any]] = []

    for case in cases:
        metadata = case.get("metadata", {}) or {}
        source_ids = metadata.get("source_openr1_ids") or []
        if not source_ids:
            continue
        current_pair = _source_pair(metadata.get("source_split", ""), source_ids[0])
        if current_pair not in excluded_source_pairs:
            continue

        try:
            seed = next(seed_iter)
        except StopIteration as exc:
            raise RuntimeError("Study C replacement pool exhausted before all overlapping cases were refreshed") from exc

        metadata["source_split"] = seed["source_openr1_split"]
        metadata["source_openr1_ids"] = [int(seed["source_openr1_id"])]

        case_id = str(case.get("id", "") or "")
        if case_id in plans:
            plans[case_id]["source_split"] = seed["source_openr1_split"]
            plans[case_id]["source_openr1_id"] = int(seed["source_openr1_id"])

        log_rows.append(
            {
                "case_id": case_id,
                "original_source_split": current_pair[0],
                "original_source_openr1_id": current_pair[1],
                "replacement_source_split": seed["source_openr1_split"],
                "replacement_source_openr1_id": int(seed["source_openr1_id"]),
            }
        )

    _write_case_replacement_log(verification_dir / STUDY_C_LOG_NAME, log_rows)

    meta = copy.deepcopy(plans_payload.get("meta", {}))
    split_counts = Counter(str(entry.get("source_split", "") or "").strip() for entry in plans.values())
    meta["source_split_counts"] = {key: value for key, value in sorted(split_counts.items())}
    meta["source_split_counts"]["total"] = len(plans)

    return {"cases": cases}, {"meta": meta, "plans": plans}, log_rows


def _write_snapshot_readme(snapshot_root: Path) -> None:
    lines = [
        "# Frozen Snapshot v4.1 (Bias + Study B Refresh)",
        "",
        "## Scope",
        "- Baseline authority: latest packaged clinical-readiness release from `data/releases/LATEST.md`.",
        "- Study A main retained unchanged.",
        "- Study A bias retained every disjoint pair group and replaced only overlapping groups.",
        "- Study B single-turn retained row content and backfilled Ready-style provenance.",
        "- Study B multi-turn regenerated with explicit provenance and cross-study source disjointness.",
        "- Study C retained already-unique cases and replaced only overlapping provenance cases.",
    ]
    (snapshot_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_snapshot_note(snapshot_root: Path) -> None:
    note = (
        "Working snapshot only. This refresh updates in-repo v4.1 working artefacts under "
        "`benchmark/runtime/data` without creating a new packaged release directory."
    )
    (snapshot_root / "README_NOTE.txt").write_text(note + "\n", encoding="utf-8")


def _write_provenance_summary(
    *,
    verification_dir: Path,
    study_b_single: list[dict[str, Any]],
    study_b_multi: list[dict[str, Any]],
) -> None:
    summary = {
        "study_b_single_total": len(study_b_single),
        "study_b_single_real": sum(
            1
            for item in study_b_single
            if str((item.get("metadata", {}) or {}).get("source_split", "") or "").strip().lower()
            in {"test", "train"}
        ),
        "study_b_single_generated": sum(
            1
            for item in study_b_single
            if str((item.get("metadata", {}) or {}).get("source_split", "") or "").strip().lower() == "generated"
        ),
        "study_b_multi_total": len(study_b_multi),
        "study_b_multi_unique_source_pairs": len(_extract_study_b_source_pairs(study_b_multi)),
    }
    _write_json(verification_dir / STUDY_B_PROVENANCE_SUMMARY, summary)


def _sync_study_a_label_sidecars(root_dir: Path) -> None:
    labels_path = root_dir / "gold_diagnosis_labels.json"
    mapping_path = root_dir / "gold_labels_mapping.json"
    metadata_path = root_dir / "gold_diagnosis_metadata.json"
    if not (labels_path.exists() and mapping_path.exists() and metadata_path.exists()):
        return

    labels_payload = _load_json(labels_path)
    labels = labels_payload.get("labels", {})
    mapping_payload = _load_json(mapping_path)
    metadata_payload = _load_json(metadata_path)

    for sample_id, entry in mapping_payload.get("mapping", {}).items():
        if sample_id in labels:
            entry["gold_label"] = labels[sample_id]

    for sample_id, entry in metadata_payload.items():
        if entry.get("new_label") is not None and sample_id in labels:
            entry["new_label"] = labels[sample_id]

    _write_json(mapping_path, mapping_payload)
    _write_json(metadata_path, metadata_payload)


def _write_run_metadata(
    *,
    verification_dir: Path,
    release_root: Path,
    bias_log_rows: list[dict[str, Any]],
    study_c_log_rows: list[dict[str, Any]],
    study_b_single: list[dict[str, Any]],
    study_b_multi: list[dict[str, Any]],
) -> None:
    payload = {
        "source_release_root": str(release_root),
        "study_a_bias": {
            "replaced_groups": len(bias_log_rows),
            "retained_groups": 1000 - len(bias_log_rows),
            "replacement_log": BIAS_LOG_NAME,
        },
        "study_b": {
            "single_turn_total": len(study_b_single),
            "single_turn_real": sum(
                1
                for item in study_b_single
                if str((item.get("metadata", {}) or {}).get("source_split", "") or "").strip().lower()
                in {"test", "train"}
            ),
            "single_turn_generated": sum(
                1
                for item in study_b_single
                if str((item.get("metadata", {}) or {}).get("source_split", "") or "").strip().lower()
                == "generated"
            ),
            "multi_turn_total": len(study_b_multi),
            "multi_turn_unique_source_pairs": len(_extract_study_b_source_pairs(study_b_multi)),
        },
        "study_c": {
            "replaced_cases": len(study_c_log_rows),
            "retained_cases": 100 - len(study_c_log_rows),
            "replacement_log": STUDY_C_LOG_NAME,
        },
        "verification_outputs": {
            "cross_study_source_disjointness": "cross_study_source_disjointness.json",
            "study_b_provenance_summary": STUDY_B_PROVENANCE_SUMMARY,
        },
    }
    _write_json(verification_dir / "run_metadata.json", payload)
    _write_json(verification_dir / "refresh_run_metadata.json", payload)


def run_refresh(
    *,
    clean: bool = False,
    out_frozen: Path = DEFAULT_OUT_FROZEN,
    out_verification: Path = DEFAULT_OUT_VERIFICATION,
) -> int:
    metric_roots = resolve_metric_data_roots(RUNTIME_ROOT, data_source="latest_release")
    release_root = metric_roots.root

    out_frozen = out_frozen.resolve()
    out_verification = out_verification.resolve()
    if clean and out_verification.exists():
        shutil.rmtree(out_verification)
    out_verification.mkdir(parents=True, exist_ok=True)

    study_a_payload = _load_json(release_root / "openr1_psy_splits" / "study_a_test.json")
    study_a_source_pairs = _extract_study_a_source_pairs(study_a_payload.get("samples", []))
    refreshed_bias_payload, bias_log_rows = _refresh_study_a_bias(
        release_root=release_root,
        study_a_source_pairs=study_a_source_pairs,
        verification_dir=out_verification,
    )
    refreshed_study_b_single = _refresh_study_b_single_turn(release_root=release_root)

    reserved_pairs = set(study_a_source_pairs)
    reserved_pairs.update(_extract_study_b_source_pairs(refreshed_study_b_single))
    reserved_pairs.update(
        _source_pair(
            (case.get("metadata", {}) or {}).get("source_openr1_split", ""),
            (case.get("metadata", {}) or {}).get("source_openr1_id", -1),
        )
        for case in refreshed_bias_payload.get("cases", [])
    )
    refreshed_study_b_multi = _refresh_study_b_multi_turn(excluded_source_pairs=reserved_pairs)
    reserved_pairs.update(_extract_study_b_source_pairs(refreshed_study_b_multi))
    refreshed_study_c_payload, refreshed_study_c_plans, study_c_log_rows = _refresh_study_c(
        release_root=release_root,
        excluded_source_pairs=reserved_pairs,
        verification_dir=out_verification,
    )

    working_bias_path = DATA_ROOT / "adversarial_bias" / "biased_vignettes.json"
    frozen_bias_path = out_frozen / "adversarial_bias" / "biased_vignettes.json"
    _write_json(working_bias_path, refreshed_bias_payload)
    _write_json(frozen_bias_path, refreshed_bias_payload)

    working_study_b_single = DATA_ROOT / "openr1_psy_splits" / "study_b_test.json"
    frozen_study_b_single = out_frozen / "study_b_test.json"
    _write_json(working_study_b_single, refreshed_study_b_single)
    _write_json(frozen_study_b_single, refreshed_study_b_single)

    working_study_b_multi = DATA_ROOT / "openr1_psy_splits" / "study_b_multi_turn_test.json"
    frozen_study_b_multi = out_frozen / "study_b_multi_turn_test.json"
    _write_json(working_study_b_multi, refreshed_study_b_multi)
    _write_json(frozen_study_b_multi, refreshed_study_b_multi)

    _copy_json(release_root / "openr1_psy_splits" / "study_a_test.json", DATA_ROOT / "openr1_psy_splits" / "study_a_test.json")
    _copy_json(release_root / "study_a_gold" / "gold_diagnosis_labels.json", DATA_ROOT / "study_a_gold" / "gold_diagnosis_labels.json")
    _copy_json(release_root / "study_a_gold" / "gold_diagnosis_metadata.json", DATA_ROOT / "study_a_gold" / "gold_diagnosis_metadata.json")
    _copy_json(release_root / "study_a_gold" / "gold_labels_mapping.json", DATA_ROOT / "study_a_gold" / "gold_labels_mapping.json")
    _write_json(DATA_ROOT / "study_c_gold" / "target_plans.json", refreshed_study_c_plans)
    _copy_json(release_root / "study_c_gold" / "entity_evidence_map.json", DATA_ROOT / "study_c_gold" / "entity_evidence_map.json")
    _write_json(DATA_ROOT / "openr1_psy_splits" / "study_c_test.json", refreshed_study_c_payload)

    _copy_json(release_root / "openr1_psy_splits" / "study_a_test.json", out_frozen / "study_a_test.json")
    _copy_json(release_root / "openr1_psy_splits" / "study_a_test.json", out_frozen / "study_a" / "study_a_test.json")
    _copy_json(release_root / "study_a_gold" / "gold_diagnosis_labels.json", out_frozen / "gold_diagnosis_labels.json")
    _copy_json(release_root / "study_a_gold" / "gold_diagnosis_labels.json", out_frozen / "study_a" / "gold_diagnosis_labels.json")
    _copy_json(release_root / "study_a_gold" / "gold_diagnosis_metadata.json", out_frozen / "gold_diagnosis_metadata.json")
    _copy_json(release_root / "study_a_gold" / "gold_diagnosis_metadata.json", out_frozen / "study_a" / "gold_diagnosis_metadata.json")
    _copy_json(release_root / "study_a_gold" / "gold_labels_mapping.json", out_frozen / "gold_labels_mapping.json")
    _copy_json(release_root / "study_a_gold" / "gold_labels_mapping.json", out_frozen / "study_a" / "gold_labels_mapping.json")
    _write_json(out_frozen / "study_c_target_plans.json", refreshed_study_c_plans)
    _write_json(out_frozen / "study_c" / "study_c_target_plans.json", refreshed_study_c_plans)
    _copy_json(release_root / "study_c_gold" / "entity_evidence_map.json", out_frozen / "entity_evidence_map.json")
    _copy_json(release_root / "study_c_gold" / "entity_evidence_map.json", out_frozen / "study_c" / "entity_evidence_map.json")
    _write_json(out_frozen / "study_c_test.json", refreshed_study_c_payload)
    _write_json(out_frozen / "study_c" / "study_c_test.json", refreshed_study_c_payload)

    _write_snapshot_readme(out_frozen)
    _write_snapshot_note(out_frozen)
    _sync_study_a_label_sidecars(DATA_ROOT / "study_a_gold")
    _sync_study_a_label_sidecars(out_frozen)
    _sync_study_a_label_sidecars(out_frozen / "study_a")

    source_manifest = _load_json(out_frozen / "manifest.json") if (out_frozen / "manifest.json").exists() else {}
    created_at_utc = str(source_manifest.get("created_at_utc", "") or "")
    if not created_at_utc:
        release_manifest = _load_json(release_root / "manifest.json")
        created_at_utc = str(release_manifest.get("created_at_utc", "") or "")
    _write_manifest(out_frozen, created_at_utc=created_at_utc)
    _write_provenance_summary(
        verification_dir=out_verification,
        study_b_single=refreshed_study_b_single,
        study_b_multi=refreshed_study_b_multi,
    )
    _write_run_metadata(
        verification_dir=out_verification,
        release_root=release_root,
        bias_log_rows=bias_log_rows,
        study_c_log_rows=study_c_log_rows,
        study_b_single=refreshed_study_b_single,
        study_b_multi=refreshed_study_b_multi,
    )

    print(f"Using latest packaged release: {release_root}")
    print(f"Study A bias groups replaced: {len(bias_log_rows)}")
    print(f"Study B single-turn rows refreshed: {len(refreshed_study_b_single)}")
    print(f"Study B multi-turn rows regenerated: {len(refreshed_study_b_multi)}")
    print(f"Study C cases refreshed: {len(study_c_log_rows)}")
    print(f"Wrote verification artefacts to: {out_verification}")
    print(f"Wrote working frozen snapshot to: {out_frozen}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Remove the output verification directory before writing new artefacts.")
    parser.add_argument(
        "--out-frozen",
        type=Path,
        default=DEFAULT_OUT_FROZEN,
        help="Working frozen snapshot to refresh.",
    )
    parser.add_argument(
        "--out-verification",
        type=Path,
        default=DEFAULT_OUT_VERIFICATION,
        help="Verification directory for refresh logs.",
    )
    args = parser.parse_args()
    return run_refresh(
        clean=args.clean,
        out_frozen=args.out_frozen,
        out_verification=args.out_verification,
    )


if __name__ == "__main__":
    raise SystemExit(main())
