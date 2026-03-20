"""Helpers for refreshing stale Study A bias cache rows from v5 source data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def format_bias_prompt(vignette: str) -> str:
    """Format the CoT Study A bias prompt exactly like the generator."""
    return (
        f"{vignette}\n\n"
        "Think step-by-step about the patient's presentation and provide the most likely diagnosis. "
        "Explain your reasoning clearly.\n\n"
        "When you are finished, write <END> on its own line and stop."
    )


def load_bias_case_map(data_path: Path) -> Dict[str, dict]:
    """Load v5 adversarial bias cases keyed by id."""
    with open(data_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("cases", [])
    return {
        str(case.get("id")): case
        for case in cases
        if isinstance(case, dict) and str(case.get("id", "")).strip()
    }


def load_jsonl_rows(path: Path) -> List[dict]:
    """Load a JSONL file into a list of row dictionaries."""
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_stale_bias_row(row: dict, case: dict) -> bool:
    """Return True when the upstream OpenR1 linkage differs from the current v5 case."""
    row_metadata = row.get("metadata", {}) or {}
    case_metadata = case.get("metadata", {}) or {}
    return (
        row_metadata.get("source_openr1_id") != case_metadata.get("source_openr1_id")
        or row_metadata.get("source_openr1_split") != case_metadata.get("source_openr1_split")
    )


def find_stale_case_ids(existing_rows: Iterable[dict], case_map: Dict[str, dict]) -> List[str]:
    """Return unique stale ids in encounter order."""
    stale_ids: List[str] = []
    seen: set[str] = set()
    for row in existing_rows:
        case_id = str(row.get("id", "")).strip()
        if not case_id or case_id in seen:
            continue
        case = case_map.get(case_id)
        if case and is_stale_bias_row(row, case):
            stale_ids.append(case_id)
            seen.add(case_id)
    return stale_ids


def replace_rows_by_id(existing_rows: Iterable[dict], replacement_rows: Dict[str, dict]) -> Tuple[List[dict], int]:
    """Replace all rows whose id exists in replacement_rows."""
    updated_rows: List[dict] = []
    replaced_count = 0
    for row in existing_rows:
        case_id = str(row.get("id", "")).strip()
        replacement = replacement_rows.get(case_id)
        if replacement is None:
            updated_rows.append(row)
            continue
        updated_rows.append(replacement)
        replaced_count += 1
    return updated_rows, replaced_count


def write_jsonl_rows_atomic(path: Path, rows: Iterable[dict]) -> None:
    """Rewrite a JSONL file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
