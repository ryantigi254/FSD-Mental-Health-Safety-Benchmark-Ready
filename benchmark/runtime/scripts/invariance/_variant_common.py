from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_split_root(source_root: Path, output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)


def infer_study_file(study: str) -> str:
    mapping = {
        "study_a": "study_a_test.json",
        "study_b": "study_b_test.json",
        "study_b_multi_turn": "study_b_multi_turn_test.json",
        "study_c": "study_c_test.json",
    }
    return mapping[study]


def read_rows(root: Path, study: str) -> Tuple[Any, List[Dict[str, Any]]]:
    payload = load_json(root / infer_study_file(study))
    if study == "study_a":
        return payload, payload.get("samples", [])
    if study == "study_c":
        return payload, payload.get("cases", [])
    return payload, payload if isinstance(payload, list) else payload.get("multi_turn_cases", [])


def write_rows(root: Path, study: str, payload: Any, rows: List[Dict[str, Any]]) -> None:
    path = root / infer_study_file(study)
    if study == "study_a":
        payload["samples"] = rows
        write_json(path, payload)
        return
    if study == "study_c":
        payload["cases"] = rows
        write_json(path, payload)
        return
    if isinstance(payload, list):
        write_json(path, rows)
        if study == "study_b_multi_turn":
            mirror = root / "study_b_multi_turn.json"
            write_json(mirror, rows)
        return
    if study == "study_b_multi_turn":
        payload["multi_turn_cases"] = rows
        write_json(path, payload)
        mirror = root / "study_b_multi_turn.json"
        write_json(mirror, rows)
        return
    payload["samples"] = rows
    write_json(path, payload)


def build_variant_metadata(
    *,
    study: str,
    variant_tag: str,
    source_root: Path,
    output_root: Path,
    changed_ids: Iterable[str],
    seed: int,
    notes: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    changed_ids = list(changed_ids)
    return {
        "study": study,
        "variant_tag": variant_tag,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": seed,
        "changed_ids": changed_ids,
        "n_changed": len(changed_ids),
        "notes": notes or {},
    }
