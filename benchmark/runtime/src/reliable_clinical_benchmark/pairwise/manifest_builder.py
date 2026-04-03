"""
Frozen case-manifest builders for pairwise evaluation.

The pairwise layer reads cached candidate generations only. It does not
regenerate candidate responses and it fails closed for slices whose cached
inputs are absent in the current branch.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_MANIFEST_VERSION = "pairwise.cases.v1"

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
        "notes": ["Awaiting controllability caches in this branch."],
    },
    "study_a_bias_controllability": {
        "layer": "controllability",
        "notes": ["Awaiting controllability caches in this branch."],
    },
    "study_b_controllability": {
        "layer": "controllability",
        "notes": ["Awaiting controllability caches in this branch."],
    },
    "study_b_multiturn_controllability": {
        "layer": "controllability",
        "notes": ["Awaiting controllability caches in this branch."],
    },
    "study_c_controllability": {
        "layer": "controllability",
        "notes": ["Awaiting controllability caches in this branch."],
    },
    "invariance": {
        "layer": "invariance",
        "notes": ["Awaiting invariance caches in this branch."],
    },
    "invariance_under_control": {
        "layer": "invariance",
        "notes": ["Awaiting invariance-under-control caches in this branch."],
    },
    "control_under_invariance": {
        "layer": "invariance",
        "notes": ["Awaiting control-under-invariance caches in this branch."],
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
        manifest = builder(runtime_path)
        manifest["slice_id"] = slice_id
        manifest["layer"] = layer
        manifest["manifest_version"] = _MANIFEST_VERSION
        manifest["built_at"] = datetime.now(timezone.utc).isoformat()

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


def _build_study_a(runtime_root: Path) -> Dict[str, Any]:
    source_root = runtime_root / "processed" / "_archived" / "duplicates" / "study_a_cleaned"
    return _build_from_jsonl_dirs(
        source_root=source_root,
        filename="study_a_generations.jsonl",
        case_key=lambda row: row["id"],
        context_for=lambda row: row.get("prompt") or "",
        text_for=lambda row: row.get("output_text") or "",
        tags_for=lambda _row: [],
        case_extra=lambda _row: {},
        boundary_notes=["Candidate responses are historical runtime panel outputs."],
    )


def _build_study_a_bias(runtime_root: Path) -> Dict[str, Any]:
    source_root = runtime_root / "processed" / "study_a_bias_pipeline"
    prompt_catalogue = _load_bias_legacy_catalogue(runtime_root)

    def context_for(row: Dict[str, Any]) -> str:
        prompt = prompt_catalogue.get(row["id"])
        if prompt:
            return prompt
        return (
            "Original study_a_bias prompt unavailable in this branch. "
            f"Bias metadata: feature={row.get('bias_feature')}, label={row.get('bias_label')}."
        )

    return _build_from_jsonl_dirs(
        source_root=source_root,
        filename="study_a_bias_processed.jsonl",
        case_key=lambda row: row["id"],
        context_for=context_for,
        text_for=lambda row: row.get("output_text") or "",
        tags_for=lambda row: ["bias", f"bias_feature:{row.get('bias_feature', 'unknown')}"],
        case_extra=lambda row: {
            "bias_feature": row.get("bias_feature"),
            "bias_label": row.get("bias_label"),
        },
        boundary_notes=[
            "Bias prompts were recovered from the legacy 2016 adversarial-bias catalogue.",
            "Candidate responses are historical runtime panel outputs.",
        ],
    )


def _build_study_b(runtime_root: Path) -> Dict[str, Any]:
    source_root = runtime_root / "processed" / "_archived" / "duplicates" / "study_b_cleaned"

    def include_row(row: Dict[str, Any]) -> bool:
        return row.get("variant") in {"control", "injected"}

    return _build_from_jsonl_dirs(
        source_root=source_root,
        filename="study_b_generations.jsonl",
        case_key=lambda row: f"{row['id']}::{row.get('variant', 'control')}",
        context_for=lambda row: row.get("prompt") or "",
        text_for=lambda row: row.get("response_text") or "",
        tags_for=lambda row: [row.get("variant", "control")] if row.get("variant") else [],
        case_extra=lambda row: {
            "variant": row.get("variant"),
            "gold_answer": row.get("gold_answer"),
        },
        include_row=include_row,
        boundary_notes=["Candidate responses are historical runtime panel outputs."],
    )


def _build_study_b_multiturn(runtime_root: Path) -> Dict[str, Any]:
    source_root = runtime_root / "processed" / "_archived" / "duplicates" / "study_b_cleaned"

    return _build_from_jsonl_dirs(
        source_root=source_root,
        filename="study_b_generations.jsonl",
        case_key=lambda row: row["case_id"],
        context_for=lambda row: row.get("conversation_text") or "",
        text_for=lambda row: row.get("response_text") or "",
        tags_for=lambda _row: ["multi_turn", "method_fit"],
        case_extra=lambda row: {
            "variant": row.get("variant"),
            "turn_num": row.get("turn_num"),
            "gold_answer": row.get("gold_answer"),
        },
        include_row=lambda row: row.get("variant") == "multi_turn",
        boundary_notes=["Candidate responses are historical runtime panel outputs."],
    )


def _build_study_c(runtime_root: Path) -> Dict[str, Any]:
    source_root = runtime_root / "processed" / "_archived" / "duplicates" / "study_c_cleaned"

    return _build_from_jsonl_dirs(
        source_root=source_root,
        filename="study_c_generations.jsonl",
        case_key=lambda row: f"{row['case_id']}::{row.get('variant', 'summary')}::turn_{row.get('turn_num', 1)}",
        context_for=lambda row: row.get("prompt") or "",
        text_for=lambda row: row.get("response_text") or "",
        tags_for=lambda row: [row.get("variant", "summary")],
        case_extra=lambda row: {
            "variant": row.get("variant"),
            "persona_id": row.get("persona_id"),
            "turn_num": row.get("turn_num"),
        },
        boundary_notes=["Candidate responses are historical runtime panel outputs."],
    )


def _build_from_jsonl_dirs(
    *,
    source_root: Path,
    filename: str,
    case_key,
    context_for,
    text_for,
    tags_for,
    case_extra,
    boundary_notes: List[str],
    include_row=None,
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

    manifest_cases: List[Dict[str, Any]] = []
    for key in sorted(cases):
        case = cases[key]
        responses = sorted(case["responses"], key=lambda item: item["system_id"])
        if len(responses) < 2:
            continue
        case["responses"] = responses
        case["pairings"] = [
            {"system_a": left["system_id"], "system_b": right["system_id"]}
            for left, right in combinations(responses, 2)
        ]
        manifest_cases.append(case)

    status = "ready" if manifest_cases else "missing_inputs"
    note = boundary_notes.copy()
    if not manifest_cases:
        note.append("No cases with at least two candidate systems were found.")

    return {
        "status": status,
        "systems": sorted(set(systems)),
        "cases": manifest_cases,
        "boundary_notes": note,
        "source_paths": source_paths,
    }


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
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
    prompt_tokens = meta.get("response_tokens")
    if isinstance(prompt_tokens, int):
        return prompt_tokens
    return None


def _word_count(text: str) -> int:
    return len([token for token in text.split() if token])


_BUILDERS = {
    "study_a": _build_study_a,
    "study_a_bias": _build_study_a_bias,
    "study_b": _build_study_b,
    "study_b_multiturn": _build_study_b_multiturn,
    "study_c": _build_study_c,
}
