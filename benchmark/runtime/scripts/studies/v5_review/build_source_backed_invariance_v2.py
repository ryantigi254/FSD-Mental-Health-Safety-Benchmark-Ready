#!/usr/bin/env python3
"""Build source-backed invariance `_v2` roots from corrected parent datasets."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = RUNTIME_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from reliable_clinical_benchmark.invariance import (  # noqa: E402
    build_invariance_manifest,
    default_invariance_sample_sizes,
    materialize_invariance_split_root,
)


V5_SAMPLE_ROOT = RUNTIME_ROOT / "data" / "invariance" / "misc" / "v5_invariance_samples"
V5_SAMPLE_ROOT_V2 = RUNTIME_ROOT / "data" / "invariance" / "misc" / "v5_invariance_samples_v2"
V6_PARENT_ROOT = RUNTIME_ROOT / "data" / "frozen_splits" / "v6"

CTRL_BASE_ROOT = RUNTIME_ROOT / "data" / "invariance" / "misc" / "invariance_variants" / "variant_family" / "base"
CTRL_BASE_ROOT_V2 = RUNTIME_ROOT / "data" / "invariance" / "misc" / "invariance_variants" / "variant_family" / "base_v2"
CTRL_PARENT_ROOT_V2 = RUNTIME_ROOT / "data" / "controllability" / "misc" / "controllability_splits_large_resolved_v2"
CTRL_SAMPLE_ROOT_V2 = (
    RUNTIME_ROOT / "data" / "invariance" / "misc" / "controllability_splits_large_resolved_invariance_samples_v2"
)
VARIANT_FAMILY_ROOT_V2 = RUNTIME_ROOT / "data" / "invariance" / "misc" / "invariance_variants" / "variant_family_v2"

AFFECTED_STUDIES = ("study_b", "study_b_multi_turn", "study_c")
UNCHANGED_STUDIES = ("study_a", "study_a_bias")
SEED = 42
MIN_HIGH_RISK = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _build_manifests(
    *,
    study_names: Iterable[str],
    source_root: Path,
    output_root: Path,
    profile: str,
) -> None:
    sample_sizes = default_invariance_sample_sizes(profile)
    for study in study_names:
        manifest = build_invariance_manifest(
            study,
            root=source_root,
            sample_size=sample_sizes[study],
            seed=SEED,
            min_high_risk=MIN_HIGH_RISK,
        )
        _write_json(output_root / f"{study}_manifest.json", manifest)


def _merged_summary(
    *,
    output_root: Path,
    inherited_root: Path,
    regenerated_summary: Dict[str, Any],
    parent_root: Path,
    profile: str,
) -> Dict[str, Any]:
    inherited_manifest = _read_json(inherited_root / "manifest.json")
    inherited_studies = dict(inherited_manifest.get("studies", {}))
    inherited_studies.update(regenerated_summary.get("studies", {}))
    inherited_studies.update(_materialized_studies(output_root))
    return {
        "created_at_utc": _now_iso(),
        "source_root": str(parent_root),
        "source_profile": profile,
        "manifest_dir": str(output_root),
        "layout": "frozen_snapshot",
        "inherits_unchanged_studies_from": str(inherited_root),
        "regeneration_policy": {
            "copied_through": list(UNCHANGED_STUDIES),
            "regenerated": list(AFFECTED_STUDIES),
        },
        "studies": inherited_studies,
    }


def _materialized_studies(root: Path) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}

    study_a_path = root / "study_a_test.json"
    if study_a_path.exists():
        payload = _read_json(study_a_path)
        rows = payload.get("samples", payload) if isinstance(payload, dict) else payload
        summary["study_a"] = {"n_rows": len(rows), "path": "study_a_test.json"}

    study_a_bias_path = root / "adversarial_bias" / "biased_vignettes.json"
    if study_a_bias_path.exists():
        payload = _read_json(study_a_bias_path)
        rows = payload.get("cases", payload) if isinstance(payload, dict) else payload
        summary["study_a_bias"] = {
            "n_rows": len(rows),
            "path": "adversarial_bias/biased_vignettes.json",
        }

    study_b_path = root / "study_b_test.json"
    if study_b_path.exists():
        rows = _read_json(study_b_path)
        summary["study_b"] = {"n_rows": len(rows), "path": "study_b_test.json"}

    study_b_multi_path = root / "study_b_multi_turn_test.json"
    if study_b_multi_path.exists():
        rows = _read_json(study_b_multi_path)
        summary["study_b_multi_turn"] = {
            "n_rows": len(rows),
            "path": "study_b_multi_turn_test.json",
        }

    study_c_path = root / "study_c_test.json"
    if study_c_path.exists():
        payload = _read_json(study_c_path)
        rows = payload.get("cases", payload) if isinstance(payload, dict) else payload
        summary["study_c"] = {"n_rows": len(rows), "path": "study_c_test.json"}

    return summary


def _build_partial_v2_root(
    *,
    inherited_root: Path,
    output_root: Path,
    parent_root: Path,
    profile: str,
) -> Dict[str, Any]:
    _replace_tree(inherited_root, output_root)
    _build_manifests(
        study_names=AFFECTED_STUDIES,
        source_root=parent_root,
        output_root=output_root,
        profile=profile,
    )
    regenerated_summary = materialize_invariance_split_root(
        source_root=parent_root,
        output_root=output_root,
        manifest_dir=output_root,
        studies=AFFECTED_STUDIES,
    )
    summary = _merged_summary(
        output_root=output_root,
        inherited_root=inherited_root,
        regenerated_summary=regenerated_summary,
        parent_root=parent_root,
        profile=profile,
    )
    _write_json(output_root / "manifest.json", summary)
    return summary


def _build_variant_family_matrix(*, base_root: Path, output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    script_path = RUNTIME_ROOT / "scripts" / "invariance" / "build_variant_family_matrix.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--base-root",
            str(base_root),
            "--output-root",
            str(output_root),
            "--seed",
            str(SEED),
        ],
        check=True,
        cwd=str(script_path.parent),
    )


def main() -> int:
    v5_summary = _build_partial_v2_root(
        inherited_root=V5_SAMPLE_ROOT,
        output_root=V5_SAMPLE_ROOT_V2,
        parent_root=V6_PARENT_ROOT,
        profile="v5",
    )

    ctrl_summary = _build_partial_v2_root(
        inherited_root=CTRL_BASE_ROOT,
        output_root=CTRL_BASE_ROOT_V2,
        parent_root=CTRL_PARENT_ROOT_V2,
        profile="controllability",
    )

    _replace_tree(CTRL_BASE_ROOT_V2, CTRL_SAMPLE_ROOT_V2)
    _write_json(
        CTRL_SAMPLE_ROOT_V2 / "manifest.json",
        {
            **ctrl_summary,
            "manifest_dir": str(CTRL_SAMPLE_ROOT_V2),
            "mirrors_base_root": str(CTRL_BASE_ROOT_V2),
        },
    )

    _build_variant_family_matrix(
        base_root=CTRL_BASE_ROOT_V2,
        output_root=VARIANT_FAMILY_ROOT_V2,
    )

    payload = {
        "v5_invariance_samples_v2": {
            "root": str(V5_SAMPLE_ROOT_V2),
            "studies": v5_summary["studies"],
        },
        "controllability_invariance_samples_v2": {
            "root": str(CTRL_SAMPLE_ROOT_V2),
            "studies": ctrl_summary["studies"],
        },
        "variant_family_v2": str(VARIANT_FAMILY_ROOT_V2),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
