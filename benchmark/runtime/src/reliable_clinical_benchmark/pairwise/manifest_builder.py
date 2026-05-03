"""
Frozen case-manifest builders for pairwise evaluation.

The pairwise layer reads cached candidate generations only. It does not
regenerate candidate responses and it fails closed only when the required
cache pair for a slice is missing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


_MANIFEST_VERSION = "pairwise.cases.v1"

CORE_MAIN_LANE_CASE_BUDGETS: Dict[str, int] = {
    "study_a": 150,
    "study_a_bias": 300,
    "study_b": 250,
    "study_b_multiturn": 300,
    "study_c": 250,
}

SLICE_METADATA: Dict[str, Dict[str, Any]] = {
    "study_a": {"layer": "core", "notes": ["Study A cleaned cached generations."]},
    "study_a_bias": {
        "layer": "core",
        "notes": [
            "Study A bias processed outputs with legacy adversarial-bias prompts.",
        ],
    },
    "study_b": {
        "layer": "core",
        "notes": ["Study B control and injected single-turn cached generations."],
    },
    "study_b_multiturn": {
        "layer": "core",
        "notes": ["Study B multi-turn cached generations."],
    },
    "study_c": {
        "layer": "core",
        "notes": ["Study C summary and dialogue cached generations."],
    },
    "study_a_controllability": {
        "layer": "controllability",
        "notes": [
            "Three-arm controllability comparison within a matched model case.",
        ],
    },
    "study_a_bias_controllability": {
        "layer": "controllability",
        "notes": [
            "Three-arm controllability comparison within a matched model case.",
            "The Study A bias explicit-control arm is transparency-focused and should be interpreted cautiously.",
        ],
    },
    "study_b_controllability": {
        "layer": "controllability",
        "notes": [
            "Three-arm controllability comparison within a matched model case.",
        ],
    },
    "study_b_multiturn_controllability": {
        "layer": "controllability",
        "notes": [
            "Three-arm controllability comparison within a matched model case.",
        ],
    },
    "study_c_controllability": {
        "layer": "controllability",
        "notes": [
            "Three-arm controllability comparison within a matched model case.",
        ],
    },
    "invariance": {
        "layer": "invariance",
        "notes": [
            "Matched benchmark base-vs-perturbed invariance cache pairs.",
        ],
    },
    "invariance_under_control": {
        "layer": "invariance",
        "notes": [
            "Matched explicit-control base-vs-perturbed cache pairs for invariance-under-control analysis.",
        ],
    },
    "control_under_invariance": {
        "layer": "invariance",
        "notes": [
            "Matched explicit-control base-vs-perturbed cache pairs for control-under-invariance analysis.",
        ],
    },
}


def runtime_root_from(path_like: str | Path) -> Path:
    path = Path(path_like).resolve()
    if path.is_dir():
        return path
    return path.parent


def build_case_manifest(
    *,
    slice_id: str,
    runtime_root: str | Path,
    output_path: str | Path | None = None,
    include_systems: Optional[List[str]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    runtime_path = runtime_root_from(runtime_root)
    if slice_id not in SLICE_METADATA:
        raise ValueError(f"unknown slice_id: {slice_id}")

    layer = SLICE_METADATA[slice_id]["layer"]
    builder = _BUILDERS.get(slice_id)
    if builder is None:
        manifest = _empty_manifest(
            slice_id=slice_id,
            layer=layer,
            notes=SLICE_METADATA[slice_id]["notes"],
        )
    else:
        manifest = builder(
            runtime_path,
            include_systems=set(include_systems or []),
        )
        manifest["slice_id"] = slice_id
        manifest["layer"] = layer
        manifest["manifest_version"] = _MANIFEST_VERSION
        manifest["built_at"] = datetime.now(timezone.utc).isoformat()
        if include_systems:
            manifest["candidate_system_filter"] = sorted(dict.fromkeys(include_systems))
            manifest["boundary_notes"] = list(manifest.get("boundary_notes", [])) + [
                "Candidate systems were filtered to the requested subset."
            ]

    if strict and manifest["status"] != "ready":
        raise FileNotFoundError(
            f"slice {slice_id} could not be built cleanly: {manifest['status']}"
        )

    destination = (
        Path(output_path)
        if output_path
        else runtime_path
        / "metric-results"
        / "pairwise"
        / "manifests"
        / f"{slice_id}_case_manifest.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _empty_manifest(*, slice_id: str, layer: str, notes: List[str]) -> Dict[str, Any]:
    return {
        "manifest_version": _MANIFEST_VERSION,
        "slice_id": slice_id,
        "layer": layer,
        "status": "missing_inputs",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "systems": [],
        "cases": [],
        "boundary_notes": list(notes),
        "source_paths": [],
    }


def _build_study_a(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    source_root = runtime_root / "processed" / "_archived" / "duplicates" / "study_a_cleaned"
    manifest = _build_from_jsonl_dirs(
        source_root=source_root,
        filename="study_a_generations.jsonl",
        case_key=lambda row: row["id"],
        context_for=lambda row: row.get("prompt") or "",
        text_for=lambda row: row.get("output_text") or "",
        tags_for=lambda _row: [],
        case_extra=lambda _row: {},
        boundary_notes=["Candidate responses are historical runtime panel outputs."],
        include_systems=include_systems,
    )
    return _apply_case_budget(manifest, slice_id="study_a")


def _build_study_a_bias(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    source_root = runtime_root / "results"
    manifest = _build_from_jsonl_dirs(
        source_root=source_root,
        filename="study_a_bias_generations.jsonl",
        case_key=lambda row: row["id"],
        context_for=lambda row: row.get("prompt") or "",
        text_for=lambda row: row.get("output_text") or row.get("response_text") or "",
        tags_for=lambda row: ["bias", f"bias_feature:{row.get('bias_feature', 'unknown')}"],
        case_extra=lambda row: {
            "bias_feature": row.get("bias_feature"),
            "bias_label": row.get("bias_label"),
        },
        boundary_notes=[
            "Bias prompts are taken from the cached study_a_bias generation files.",
            "Candidate responses are historical runtime panel outputs.",
        ],
        include_row=lambda row: str(row.get("status") or "").strip().lower() == "ok",
        include_systems=include_systems,
    )
    return _apply_case_budget(manifest, slice_id="study_a_bias")


def _build_study_b(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    source_root = runtime_root / "processed" / "_archived" / "duplicates" / "study_b_cleaned"
    manifest = _build_from_jsonl_dirs(
        source_root=source_root,
        filename="study_b_generations.jsonl",
        case_key=_study_b_case_key,
        context_for=lambda row: row.get("prompt") or "",
        text_for=lambda row: row.get("response_text") or "",
        tags_for=lambda row: [row.get("variant", "control")] if row.get("variant") else [],
        case_extra=lambda row: {
            "variant": row.get("variant"),
            "gold_answer": row.get("gold_answer"),
        },
        include_row=lambda row: row.get("variant") in {"control", "injected"},
        boundary_notes=["Candidate responses are historical runtime panel outputs."],
        include_systems=include_systems,
    )
    return _apply_case_budget(manifest, slice_id="study_b")


def _build_study_b_multiturn(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    source_root = runtime_root / "results"
    manifest = _build_from_jsonl_dirs(
        source_root=source_root,
        filename="study_b_multi_turn_generations.jsonl",
        case_key=_study_b_multiturn_case_key,
        context_for=lambda row: row.get("conversation_text") or "",
        text_for=lambda row: row.get("response_text") or "",
        tags_for=lambda _row: ["multi_turn", "method_fit"],
        case_extra=lambda row: {
            "variant": row.get("variant"),
            "turn_num": row.get("turn_num"),
            "gold_answer": row.get("gold_answer"),
        },
        include_row=lambda row: (
            row.get("variant") == "multi_turn"
            and str(row.get("status") or "").strip().lower() == "ok"
        ),
        boundary_notes=["Candidate responses are historical runtime panel outputs."],
        include_systems=include_systems,
    )
    return _apply_case_budget(manifest, slice_id="study_b_multiturn")


def _build_study_c(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    source_root = runtime_root / "processed" / "_archived" / "duplicates" / "study_c_cleaned"
    manifest = _build_from_jsonl_dirs(
        source_root=source_root,
        filename="study_c_generations.jsonl",
        case_key=_study_c_case_key,
        context_for=lambda row: row.get("prompt") or "",
        text_for=lambda row: row.get("response_text") or "",
        tags_for=lambda row: [row.get("variant", "summary")],
        case_extra=lambda row: {
            "variant": row.get("variant"),
            "persona_id": row.get("persona_id"),
            "turn_num": row.get("turn_num"),
        },
        boundary_notes=["Candidate responses are historical runtime panel outputs."],
        include_systems=include_systems,
    )
    return _apply_case_budget(manifest, slice_id="study_c")


def _build_study_a_controllability(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    return _build_controllability_manifest(
        runtime_root=runtime_root,
        study_key="study_a",
        filename="ctrl_study_a_generations.jsonl",
        case_key=lambda row: row["id"],
        context_for=lambda row: row.get("prompt") or "",
        text_for=lambda row: row.get("output_text") or row.get("response_text") or "",
        tags_for=lambda row: [_normalise_control_arm(row)],
        case_extra=lambda row: {
            "mode": row.get("mode"),
            "model_name": row.get("model_name"),
        },
        boundary_notes=SLICE_METADATA["study_a_controllability"]["notes"],
        include_systems=include_systems,
    )


def _build_study_a_bias_controllability(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    return _build_controllability_manifest(
        runtime_root=runtime_root,
        study_key="study_a_bias",
        filename="ctrl_study_a_bias_generations.jsonl",
        case_key=lambda row: row["id"],
        context_for=lambda row: row.get("prompt") or "",
        text_for=lambda row: row.get("output_text") or row.get("response_text") or "",
        tags_for=lambda row: [
            _normalise_control_arm(row),
            "bias",
            f"bias_feature:{row.get('bias_feature', 'unknown')}",
        ],
        case_extra=lambda row: {
            "bias_feature": row.get("bias_feature"),
            "bias_label": row.get("bias_label"),
            "pair_group_id": row.get("pair_group_id"),
        },
        boundary_notes=SLICE_METADATA["study_a_bias_controllability"]["notes"],
        include_systems=include_systems,
    )


def _build_study_b_controllability(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    return _build_controllability_manifest(
        runtime_root=runtime_root,
        study_key="study_b",
        filename="ctrl_study_b_generations.jsonl",
        case_key=_study_b_case_key,
        context_for=lambda row: row.get("prompt") or "",
        text_for=lambda row: row.get("response_text") or row.get("output_text") or "",
        tags_for=lambda row: [_normalise_control_arm(row), row.get("variant", "control")],
        case_extra=lambda row: {
            "variant": row.get("variant"),
            "gold_answer": row.get("gold_answer"),
            "incorrect_opinion": row.get("incorrect_opinion"),
        },
        boundary_notes=SLICE_METADATA["study_b_controllability"]["notes"],
        include_systems=include_systems,
    )


def _build_study_b_multiturn_controllability(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    return _build_controllability_manifest(
        runtime_root=runtime_root,
        study_key="study_b_multiturn",
        filename="ctrl_study_b_multi_turn_generations.jsonl",
        case_key=_study_b_multiturn_case_key,
        context_for=lambda row: row.get("conversation_text") or row.get("conversation_history") or "",
        text_for=lambda row: row.get("response_text") or row.get("output_text") or "",
        tags_for=lambda row: [_normalise_control_arm(row), "multi_turn"],
        case_extra=lambda row: {
            "variant": row.get("variant"),
            "turn_num": row.get("turn_num"),
            "pressure_level": (row.get("meta") or {}).get("pressure_level"),
        },
        boundary_notes=SLICE_METADATA["study_b_multiturn_controllability"]["notes"],
        include_systems=include_systems,
    )


def _build_study_c_controllability(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    return _build_controllability_manifest(
        runtime_root=runtime_root,
        study_key="study_c",
        filename="ctrl_study_c_generations.jsonl",
        case_key=_study_c_case_key,
        context_for=lambda row: row.get("prompt") or row.get("conversation_text") or "",
        text_for=lambda row: row.get("response_text") or row.get("output_text") or "",
        tags_for=lambda row: [_normalise_control_arm(row), row.get("variant", "summary")],
        case_extra=lambda row: {
            "variant": row.get("variant"),
            "turn_num": row.get("turn_num"),
            "persona_id": row.get("persona_id"),
        },
        boundary_notes=SLICE_METADATA["study_c_controllability"]["notes"],
        include_systems=include_systems,
    )


def _build_invariance(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    return _build_matched_manifest(
        runtime_root=runtime_root,
        slice_id="invariance",
        system_base="benchmark_base",
        system_variant="invariance_variant",
        specs=_benchmark_invariance_specs(runtime_root),
        boundary_notes=SLICE_METADATA["invariance"]["notes"],
        include_systems=include_systems,
    )


def _build_invariance_under_control(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    return _build_matched_manifest(
        runtime_root=runtime_root,
        slice_id="invariance_under_control",
        system_base="explicit_control_base",
        system_variant="ctrl_invariance_variant",
        specs=_control_invariance_specs(runtime_root),
        boundary_notes=SLICE_METADATA["invariance_under_control"]["notes"],
        include_systems=include_systems,
    )


def _build_control_under_invariance(runtime_root: Path, include_systems: set[str]) -> Dict[str, Any]:
    return _build_matched_manifest(
        runtime_root=runtime_root,
        slice_id="control_under_invariance",
        system_base="explicit_control_base",
        system_variant="control_under_invariance_variant",
        specs=_control_invariance_specs(runtime_root),
        boundary_notes=SLICE_METADATA["control_under_invariance"]["notes"],
        include_systems=include_systems,
    )


def _build_from_jsonl_dirs(
    *,
    source_root: Path,
    filename: str,
    case_key: Callable[[Dict[str, Any]], str],
    context_for: Callable[[Dict[str, Any]], str],
    text_for: Callable[[Dict[str, Any]], str],
    tags_for: Callable[[Dict[str, Any]], List[str]],
    case_extra: Callable[[Dict[str, Any]], Dict[str, Any]],
    boundary_notes: List[str],
    include_row: Optional[Callable[[Dict[str, Any]], bool]] = None,
    include_systems: Optional[set[str]] = None,
) -> Dict[str, Any]:
    if not source_root.exists():
        return {
            "status": "missing_inputs",
            "systems": [],
            "cases": [],
            "boundary_notes": boundary_notes + [f"Missing source root: {source_root}"],
            "source_paths": [str(source_root)],
        }

    cases: Dict[str, Dict[str, Any]] = {}
    systems: List[str] = []
    source_paths: List[str] = []

    for model_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if include_systems and model_dir.name not in include_systems:
            continue
        source_path = model_dir / filename
        if not source_path.exists():
            continue
        source_paths.append(str(source_path))
        system_id = model_dir.name
        systems.append(system_id)

        for row in _read_jsonl(source_path):
            if include_row and not include_row(row):
                continue
            text = str(text_for(row)).strip()
            if not text:
                continue
            key = str(case_key(row))
            if key not in cases:
                cases[key] = {
                    "case_id": key,
                    "context": str(context_for(row)).strip(),
                    "tags": list(tags_for(row)),
                    "responses": [],
                    "case_meta": case_extra(row),
                }

            cases[key]["responses"].append(
                {
                    "system_id": system_id,
                    "response_id": f"{key}::{system_id}",
                    "text": text,
                    "token_count": _token_count_from_row(row),
                    "word_count": _word_count(text),
                    "source_path": str(source_path),
                }
            )

    return _finalise_grouped_cases(
        cases=cases,
        systems=sorted(set(systems)),
        boundary_notes=boundary_notes,
        source_paths=source_paths,
    )


def _build_controllability_manifest(
    *,
    runtime_root: Path,
    study_key: str,
    filename: str,
    case_key: Callable[[Dict[str, Any]], str],
    context_for: Callable[[Dict[str, Any]], str],
    text_for: Callable[[Dict[str, Any]], str],
    tags_for: Callable[[Dict[str, Any]], List[str]],
    case_extra: Callable[[Dict[str, Any]], Dict[str, Any]],
    boundary_notes: List[str],
    include_systems: Optional[set[str]] = None,
) -> Dict[str, Any]:
    source_root = runtime_root / "results"
    if not source_root.exists():
        return _missing_source_manifest(boundary_notes, source_root)

    cases: Dict[str, Dict[str, Any]] = {}
    systems: List[str] = []
    source_paths: List[str] = []

    for model_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if include_systems and model_dir.name not in include_systems:
            continue
        source_path = model_dir / filename
        if not source_path.exists():
            continue
        source_paths.append(str(source_path))
        for row in _read_jsonl(source_path):
            text = str(text_for(row)).strip()
            if not text:
                continue
            control_system = _normalise_control_arm(row)
            systems.append(control_system)
            raw_case_key = str(case_key(row))
            manifest_case_id = f"{study_key}::{raw_case_key}::{model_dir.name}"
            if manifest_case_id not in cases:
                cases[manifest_case_id] = {
                    "case_id": manifest_case_id,
                    "context": str(context_for(row)).strip(),
                    "tags": list(dict.fromkeys(["study:" + study_key, *tags_for(row)])),
                    "responses": [],
                    "case_meta": {
                        "study": study_key,
                        "model_id": model_dir.name,
                        "raw_case_id": raw_case_key,
                        **case_extra(row),
                    },
                }
            cases[manifest_case_id]["responses"].append(
                {
                    "system_id": control_system,
                    "response_id": f"{manifest_case_id}::{control_system}",
                    "text": text,
                    "token_count": _token_count_from_row(row),
                    "word_count": _word_count(text),
                    "source_path": str(source_path),
                }
            )

    return _finalise_grouped_cases(
        cases=cases,
        systems=sorted(set(systems)),
        boundary_notes=boundary_notes,
        source_paths=source_paths,
    )


def _build_matched_manifest(
    *,
    runtime_root: Path,
    slice_id: str,
    system_base: str,
    system_variant: str,
    specs: List[Dict[str, Any]],
    boundary_notes: List[str],
    include_systems: Optional[set[str]] = None,
) -> Dict[str, Any]:
    cases: Dict[str, Dict[str, Any]] = {}
    source_paths: List[str] = []
    available_specs = 0

    for spec in specs:
        study = spec["study"]
        base_root = spec["base_root"]
        variant_root = spec["variant_root"]
        base_filename = spec["base_filename"]
        variant_filename = spec["variant_filename"]
        include_base = spec.get("include_base")
        base_context_for = spec["base_context_for"]
        base_text_for = spec["base_text_for"]
        variant_text_for = spec["variant_text_for"]
        case_key = spec["case_key"]
        tags_for = spec["tags_for"]
        case_meta = spec["case_meta"]

        if not base_root.exists():
            boundary_notes = boundary_notes + [f"Missing base root for {study}: {base_root}"]
            continue
        if not variant_root.exists():
            boundary_notes = boundary_notes + [f"Missing variant root for {study}: {variant_root}"]
            continue

        for variant_model_dir in sorted(path for path in variant_root.iterdir() if path.is_dir()):
            if include_systems and variant_model_dir.name not in include_systems:
                continue
            variant_path = variant_model_dir / variant_filename
            if not variant_path.exists():
                continue
            base_path = base_root / variant_model_dir.name / base_filename
            if not base_path.exists():
                continue

            base_rows: Dict[str, Dict[str, Any]] = {}
            for row in _read_jsonl(base_path):
                if include_base and not include_base(row):
                    continue
                key = str(case_key(row))
                base_rows[key] = row

            matched_rows = 0
            for variant_row in _read_jsonl(variant_path):
                key = str(case_key(variant_row))
                base_row = base_rows.get(key)
                if base_row is None:
                    continue
                base_text = str(base_text_for(base_row)).strip()
                variant_text = str(variant_text_for(variant_row)).strip()
                if not base_text or not variant_text:
                    continue

                manifest_case_id = f"{slice_id}::{study}::{variant_model_dir.name}::{key}"
                tags = list(
                    dict.fromkeys(
                        [
                            f"study:{study}",
                            f"model:{variant_model_dir.name}",
                            *tags_for(base_row),
                        ]
                    )
                )
                cases[manifest_case_id] = {
                    "case_id": manifest_case_id,
                    "context": str(base_context_for(base_row)).strip(),
                    "tags": tags,
                    "responses": [
                        {
                            "system_id": system_base,
                            "response_id": f"{manifest_case_id}::{system_base}",
                            "text": base_text,
                            "token_count": _token_count_from_row(base_row),
                            "word_count": _word_count(base_text),
                            "source_path": str(base_path),
                        },
                        {
                            "system_id": system_variant,
                            "response_id": f"{manifest_case_id}::{system_variant}",
                            "text": variant_text,
                            "token_count": _token_count_from_row(variant_row),
                            "word_count": _word_count(variant_text),
                            "source_path": str(variant_path),
                        },
                    ],
                    "case_meta": {
                        "study": study,
                        "model_id": variant_model_dir.name,
                        "raw_case_id": key,
                        **case_meta(base_row),
                    },
                    "pairings": [{"system_a": system_base, "system_b": system_variant}],
                }
                matched_rows += 1

            if matched_rows:
                available_specs += 1
                source_paths.extend([str(base_path), str(variant_path)])

    status = "ready" if cases else "missing_inputs"
    notes = list(boundary_notes)
    if not cases:
        notes.append("No matched base/variant cache pairs were found.")
    elif available_specs < len(specs):
        notes.append(
            "Some study/model combinations were unavailable; the slice uses only matched cache pairs present in this branch."
        )

    return {
        "status": status,
        "systems": [system_base, system_variant],
        "cases": [cases[key] for key in sorted(cases)],
        "boundary_notes": notes,
        "source_paths": sorted(dict.fromkeys(source_paths)),
    }


def _benchmark_invariance_specs(runtime_root: Path) -> List[Dict[str, Any]]:
    return [
        {
            "study": "study_a",
            "base_root": runtime_root / "processed" / "_archived" / "duplicates" / "study_a_cleaned",
            "variant_root": runtime_root / "results_invariance",
            "base_filename": "study_a_generations.jsonl",
            "variant_filename": "study_a_invariance_generations.jsonl",
            "case_key": lambda row: row["id"],
            "base_context_for": lambda row: row.get("prompt") or "",
            "base_text_for": lambda row: row.get("output_text") or "",
            "variant_text_for": lambda row: row.get("output_text") or row.get("response_text") or "",
            "tags_for": lambda _row: ["invariance"],
            "case_meta": lambda _row: {},
        },
        {
            "study": "study_a_bias",
            "base_root": runtime_root / "processed" / "study_a_bias_pipeline",
            "variant_root": runtime_root / "results_invariance",
            "base_filename": "study_a_bias_processed.jsonl",
            "variant_filename": "study_a_bias_invariance_generations.jsonl",
            "case_key": lambda row: row["id"],
            "base_context_for": lambda row: row.get("prompt") or "",
            "base_text_for": lambda row: row.get("output_text") or "",
            "variant_text_for": lambda row: row.get("output_text") or row.get("response_text") or "",
            "tags_for": lambda row: [
                "invariance",
                "bias",
                f"bias_feature:{row.get('bias_feature', 'unknown')}",
            ],
            "case_meta": lambda row: {
                "bias_feature": row.get("bias_feature"),
                "bias_label": row.get("bias_label"),
            },
        },
        {
            "study": "study_b",
            "base_root": runtime_root / "processed" / "_archived" / "duplicates" / "study_b_cleaned",
            "variant_root": runtime_root / "results_invariance",
            "base_filename": "study_b_generations.jsonl",
            "variant_filename": "study_b_invariance_generations.jsonl",
            "case_key": _study_b_case_key,
            "base_context_for": lambda row: row.get("prompt") or "",
            "base_text_for": lambda row: row.get("response_text") or "",
            "variant_text_for": lambda row: row.get("response_text") or row.get("output_text") or "",
            "tags_for": lambda row: ["invariance", row.get("variant", "control")],
            "case_meta": lambda row: {
                "variant": row.get("variant"),
                "gold_answer": row.get("gold_answer"),
            },
            "include_base": lambda row: row.get("variant") in {"control", "injected"},
        },
        {
            "study": "study_b_multiturn",
            "base_root": runtime_root / "processed" / "_archived" / "duplicates" / "study_b_cleaned",
            "variant_root": runtime_root / "results_invariance",
            "base_filename": "study_b_generations.jsonl",
            "variant_filename": "study_b_multi_turn_invariance_generations.jsonl",
            "case_key": _study_b_multiturn_case_key,
            "base_context_for": lambda row: row.get("conversation_text") or "",
            "base_text_for": lambda row: row.get("response_text") or "",
            "variant_text_for": lambda row: row.get("response_text") or row.get("output_text") or "",
            "tags_for": lambda _row: ["invariance", "multi_turn"],
            "case_meta": lambda row: {
                "variant": row.get("variant"),
                "turn_num": row.get("turn_num"),
            },
            "include_base": lambda row: row.get("variant") == "multi_turn",
        },
        {
            "study": "study_c",
            "base_root": runtime_root / "processed" / "_archived" / "duplicates" / "study_c_cleaned",
            "variant_root": runtime_root / "results_invariance",
            "base_filename": "study_c_generations.jsonl",
            "variant_filename": "study_c_invariance_generations.jsonl",
            "case_key": _study_c_case_key,
            "base_context_for": lambda row: row.get("prompt") or "",
            "base_text_for": lambda row: row.get("response_text") or "",
            "variant_text_for": lambda row: row.get("response_text") or row.get("output_text") or "",
            "tags_for": lambda row: ["invariance", row.get("variant", "summary")],
            "case_meta": lambda row: {
                "variant": row.get("variant"),
                "turn_num": row.get("turn_num"),
            },
        },
    ]


def _control_invariance_specs(runtime_root: Path) -> List[Dict[str, Any]]:
    return [
        {
            "study": "study_a",
            "base_root": runtime_root / "results",
            "variant_root": runtime_root / "results_ctrl_invariance",
            "base_filename": "ctrl_study_a_generations.jsonl",
            "variant_filename": "study_a_invariance_generations.jsonl",
            "case_key": lambda row: row["id"],
            "base_context_for": lambda row: row.get("prompt") or "",
            "base_text_for": lambda row: row.get("output_text") or row.get("response_text") or "",
            "variant_text_for": lambda row: row.get("output_text") or row.get("response_text") or "",
            "tags_for": lambda _row: ["control", "explicit_control"],
            "case_meta": lambda row: {"mode": row.get("mode")},
            "include_base": _is_explicit_control_row,
        },
        {
            "study": "study_a_bias",
            "base_root": runtime_root / "results",
            "variant_root": runtime_root / "results_ctrl_invariance",
            "base_filename": "ctrl_study_a_bias_generations.jsonl",
            "variant_filename": "study_a_bias_invariance_generations.jsonl",
            "case_key": lambda row: row["id"],
            "base_context_for": lambda row: row.get("prompt") or "",
            "base_text_for": lambda row: row.get("output_text") or row.get("response_text") or "",
            "variant_text_for": lambda row: row.get("output_text") or row.get("response_text") or "",
            "tags_for": lambda row: [
                "control",
                "explicit_control",
                "bias",
                f"bias_feature:{row.get('bias_feature', 'unknown')}",
            ],
            "case_meta": lambda row: {
                "bias_feature": row.get("bias_feature"),
                "bias_label": row.get("bias_label"),
            },
            "include_base": _is_explicit_control_row,
        },
        {
            "study": "study_b",
            "base_root": runtime_root / "results",
            "variant_root": runtime_root / "results_ctrl_invariance",
            "base_filename": "ctrl_study_b_generations.jsonl",
            "variant_filename": "study_b_invariance_generations.jsonl",
            "case_key": _study_b_case_key,
            "base_context_for": lambda row: row.get("prompt") or "",
            "base_text_for": lambda row: row.get("response_text") or row.get("output_text") or "",
            "variant_text_for": lambda row: row.get("response_text") or row.get("output_text") or "",
            "tags_for": lambda row: ["control", "explicit_control", row.get("variant", "control")],
            "case_meta": lambda row: {
                "variant": row.get("variant"),
                "gold_answer": row.get("gold_answer"),
            },
            "include_base": _is_explicit_control_row,
        },
        {
            "study": "study_b_multiturn",
            "base_root": runtime_root / "results",
            "variant_root": runtime_root / "results_ctrl_invariance",
            "base_filename": "ctrl_study_b_multi_turn_generations.jsonl",
            "variant_filename": "study_b_multi_turn_invariance_generations.jsonl",
            "case_key": _study_b_multiturn_case_key,
            "base_context_for": lambda row: row.get("conversation_text") or row.get("conversation_history") or "",
            "base_text_for": lambda row: row.get("response_text") or row.get("output_text") or "",
            "variant_text_for": lambda row: row.get("response_text") or row.get("output_text") or "",
            "tags_for": lambda _row: ["control", "explicit_control", "multi_turn"],
            "case_meta": lambda row: {"turn_num": row.get("turn_num")},
            "include_base": _is_explicit_control_row,
        },
        {
            "study": "study_c",
            "base_root": runtime_root / "results",
            "variant_root": runtime_root / "results_ctrl_invariance",
            "base_filename": "ctrl_study_c_generations.jsonl",
            "variant_filename": "study_c_invariance_generations.jsonl",
            "case_key": _study_c_case_key,
            "base_context_for": lambda row: row.get("prompt") or row.get("conversation_text") or "",
            "base_text_for": lambda row: row.get("response_text") or row.get("output_text") or "",
            "variant_text_for": lambda row: row.get("response_text") or row.get("output_text") or "",
            "tags_for": lambda row: ["control", "explicit_control", row.get("variant", "summary")],
            "case_meta": lambda row: {
                "variant": row.get("variant"),
                "turn_num": row.get("turn_num"),
            },
            "include_base": _is_explicit_control_row,
        },
    ]


def _finalise_grouped_cases(
    *,
    cases: Dict[str, Dict[str, Any]],
    systems: List[str],
    boundary_notes: List[str],
    source_paths: List[str],
) -> Dict[str, Any]:
    manifest_cases: List[Dict[str, Any]] = []
    for key in sorted(cases):
        case = cases[key]
        response_by_system: Dict[str, Dict[str, Any]] = {}
        for response in case["responses"]:
            response_by_system.setdefault(response["system_id"], response)
        responses = sorted(response_by_system.values(), key=lambda item: item["system_id"])
        if len(responses) < 2:
            continue
        case["responses"] = responses
        case["pairings"] = [
            {"system_a": left["system_id"], "system_b": right["system_id"]}
            for left, right in combinations(responses, 2)
        ]
        manifest_cases.append(case)

    notes = list(boundary_notes)
    status = "ready" if manifest_cases else "missing_inputs"
    if not manifest_cases:
        notes.append("No cases with at least two candidate systems were found.")

    return {
        "status": status,
        "systems": sorted(set(systems)),
        "cases": manifest_cases,
        "boundary_notes": notes,
        "source_paths": sorted(dict.fromkeys(source_paths)),
    }


def _apply_case_budget(manifest: Dict[str, Any], *, slice_id: str) -> Dict[str, Any]:
    budget = CORE_MAIN_LANE_CASE_BUDGETS.get(slice_id)
    if not budget or manifest.get("status") != "ready":
        return manifest

    cases = list(manifest.get("cases", []))
    if len(cases) <= budget:
        return manifest

    expected_response_count = len(manifest.get("systems", []))
    complete_cases = [
        case
        for case in cases
        if len(case.get("responses", [])) == expected_response_count
    ]
    if len(complete_cases) >= budget:
        cases = complete_cases

    manifest = dict(manifest)
    manifest["cases"] = cases[:budget]
    manifest["boundary_notes"] = list(manifest.get("boundary_notes", [])) + [
        f"Main-lane case budget applied: first {budget} deterministic complete case_ids retained."
    ]
    return manifest


def _missing_source_manifest(boundary_notes: List[str], source_root: Path) -> Dict[str, Any]:
    return {
        "status": "missing_inputs",
        "systems": [],
        "cases": [],
        "boundary_notes": list(boundary_notes) + [f"Missing source root: {source_root}"],
        "source_paths": [str(source_root)],
    }


def _normalise_control_arm(row: Dict[str, Any]) -> str:
    arm = str(row.get("arm") or "").strip()
    prompt_id = str(row.get("control_prompt_id") or "").strip()
    if arm:
        return arm
    if prompt_id == "generic":
        return "generic_control"
    if prompt_id == "explicit":
        return "explicit_control"
    return "spontaneous"


def _is_explicit_control_row(row: Dict[str, Any]) -> bool:
    return _normalise_control_arm(row) == "explicit_control"


def _study_b_case_key(row: Dict[str, Any]) -> str:
    return f"{row['id']}::{row.get('variant', 'control')}"


def _study_b_multiturn_case_key(row: Dict[str, Any]) -> str:
    return f"{row['case_id']}::turn_{row.get('turn_num', 1)}"


def _study_c_case_key(row: Dict[str, Any]) -> str:
    return (
        f"{row['case_id']}::{row.get('variant', 'summary')}::turn_{row.get('turn_num', 1)}"
    )


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1252")

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("version https://git-lfs.github.com/spec/v1"):
            continue
        yield json.loads(line)


def _load_bias_legacy_catalogue(runtime_root: Path) -> Dict[str, str]:
    catalogue_path = (
        runtime_root
        / "data"
        / "releases"
        / "clinician_readiness_v4_2026-02-22"
        / "adversarial_bias"
        / "biased_vignettes_legacy_2016.json"
    )
    if not catalogue_path.exists():
        return {}
    payload = json.loads(catalogue_path.read_text(encoding="utf-8"))
    prompts: Dict[str, str] = {}
    for row in payload.get("cases", []):
        row_id = str(row.get("id", ""))
        if row_id.startswith("bias_"):
            prompts[row_id.replace("bias_", "bias_0")] = str(row.get("prompt", ""))
            prompts[row_id] = str(row.get("prompt", ""))
    return prompts


def _token_count_from_row(row: Dict[str, Any]) -> Optional[int]:
    meta = row.get("meta") or {}
    response_tokens = meta.get("response_tokens")
    if isinstance(response_tokens, int):
        return response_tokens
    return None


def _word_count(text: str) -> int:
    return len([token for token in text.split() if token])


_BUILDERS = {
    "study_a": _build_study_a,
    "study_a_bias": _build_study_a_bias,
    "study_b": _build_study_b,
    "study_b_multiturn": _build_study_b_multiturn,
    "study_c": _build_study_c,
    "study_a_controllability": _build_study_a_controllability,
    "study_a_bias_controllability": _build_study_a_bias_controllability,
    "study_b_controllability": _build_study_b_controllability,
    "study_b_multiturn_controllability": _build_study_b_multiturn_controllability,
    "study_c_controllability": _build_study_c_controllability,
    "invariance": _build_invariance,
    "invariance_under_control": _build_invariance_under_control,
    "control_under_invariance": _build_control_under_invariance,
}
