#!/usr/bin/env python3
"""Build source-backed invariance roots from strict parent datasets.

Supports two ctrl lineages:
  - v2_1: sampled from controllability_splits_v2_1
  - v3:   sampled from controllability_splits_v3  (scaled pool)
"""

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


# v2_1 uses v6_1 strict parents instead of v6
# Reorganised layout: data/invariance/ and data/controllability/
V5_SAMPLE_ROOT = RUNTIME_ROOT / "data" / "invariance" / "misc" / "v5_invariance_samples"
V5_SAMPLE_ROOT_V2_1 = RUNTIME_ROOT / "data" / "invariance" / "v5" / "base" / "v2_1"
V6_1_PARENT_ROOT = RUNTIME_ROOT / "data" / "frozen_splits" / "v6_1"

CTRL_BASE_ROOT = RUNTIME_ROOT / "data" / "invariance" / "misc" / "invariance_variants" / "variant_family" / "base"
CTRL_BASE_ROOT_V2_1 = RUNTIME_ROOT / "data" / "invariance" / "misc" / "ctrl_variants_base_v2_1"
CTRL_PARENT_ROOT_V2_1 = RUNTIME_ROOT / "data" / "controllability" / "controllability_splits_v2_1"
CTRL_SAMPLE_ROOT_V2_1 = RUNTIME_ROOT / "data" / "invariance" / "ctrl" / "base" / "v2_1"
VARIANT_FAMILY_ROOT_V2_1 = RUNTIME_ROOT / "data" / "invariance" / "ctrl" / "variants" / "v2_1"

# v3: sampled from the scaled controllability_splits_v3 pool
CTRL_BASE_ROOT_V3 = RUNTIME_ROOT / "data" / "invariance" / "misc" / "ctrl_variants_base_v3"
CTRL_PARENT_ROOT_V3 = RUNTIME_ROOT / "data" / "controllability" / "controllability_splits_v3"
CTRL_SAMPLE_ROOT_V3 = RUNTIME_ROOT / "data" / "invariance" / "ctrl" / "base" / "v3"
VARIANT_FAMILY_ROOT_V3 = RUNTIME_ROOT / "data" / "invariance" / "ctrl" / "variants" / "v3"

AFFECTED_STUDIES = ("study_a", "study_a_bias", "study_b", "study_b_multi_turn", "study_c")
UNCHANGED_STUDIES: tuple[str, ...] = ()
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
        summary["study_a_bias"] = {"n_rows": len(rows), "path": "adversarial_bias/biased_vignettes.json"}

    study_b_path = root / "study_b_test.json"
    if study_b_path.exists():
        rows = _read_json(study_b_path)
        summary["study_b"] = {"n_rows": len(rows), "path": "study_b_test.json"}

    study_b_multi_path = root / "study_b_multi_turn_test.json"
    if study_b_multi_path.exists():
        rows = _read_json(study_b_multi_path)
        summary["study_b_multi_turn"] = {"n_rows": len(rows), "path": "study_b_multi_turn_test.json"}

    study_c_path = root / "study_c_test.json"
    if study_c_path.exists():
        payload = _read_json(study_c_path)
        rows = payload.get("cases", payload) if isinstance(payload, dict) else payload
        summary["study_c"] = {"n_rows": len(rows), "path": "study_c_test.json"}

    return summary


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
        "parent_version": "v6.1_strict_no_generation",
        "regeneration_policy": {
            "copied_through": list(UNCHANGED_STUDIES),
            "regenerated": list(AFFECTED_STUDIES),
        },
        "studies": inherited_studies,
    }


def _build_partial_v2_1_root(
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


def _read_existing_summary(root: Path) -> Dict[str, Any]:
    """Load an already-built manifest as the summary dict."""
    manifest = _read_json(root / "manifest.json")
    return manifest


def main() -> int:
    # ── v2_1 builds (skip if already materialised or inherited root missing) ──
    v5_summary: Dict[str, Any] | None = None
    if V5_SAMPLE_ROOT_V2_1.exists() and (V5_SAMPLE_ROOT_V2_1 / "manifest.json").exists():
        print(f"[skip] v5 v2_1 already exists at {V5_SAMPLE_ROOT_V2_1}")
        v5_summary = _read_existing_summary(V5_SAMPLE_ROOT_V2_1)
    elif V5_SAMPLE_ROOT.exists():
        v5_summary = _build_partial_v2_1_root(
            inherited_root=V5_SAMPLE_ROOT,
            output_root=V5_SAMPLE_ROOT_V2_1,
            parent_root=V6_1_PARENT_ROOT,
            profile="v5",
        )
    else:
        print(f"[skip] v5 inherited root missing ({V5_SAMPLE_ROOT}), skipping v5 build")

    if CTRL_SAMPLE_ROOT_V2_1.exists() and (CTRL_SAMPLE_ROOT_V2_1 / "manifest.json").exists():
        print(f"[skip] ctrl v2_1 already exists at {CTRL_SAMPLE_ROOT_V2_1}")
        ctrl_summary = _read_existing_summary(CTRL_SAMPLE_ROOT_V2_1)
    else:
        ctrl_summary = _build_partial_v2_1_root(
            inherited_root=CTRL_BASE_ROOT,
            output_root=CTRL_BASE_ROOT_V2_1,
            parent_root=CTRL_PARENT_ROOT_V2_1,
            profile="controllability",
        )
        _replace_tree(CTRL_BASE_ROOT_V2_1, CTRL_SAMPLE_ROOT_V2_1)
        _write_json(
            CTRL_SAMPLE_ROOT_V2_1 / "manifest.json",
            {
                **ctrl_summary,
                "manifest_dir": str(CTRL_SAMPLE_ROOT_V2_1),
                "mirrors_base_root": str(CTRL_BASE_ROOT_V2_1),
            },
        )
        _build_variant_family_matrix(
            base_root=CTRL_BASE_ROOT_V2_1,
            output_root=VARIANT_FAMILY_ROOT_V2_1,
        )

    # ── v3 ctrl invariance (from scaled controllability_splits_v3) ──
    ctrl_v3_inherited = CTRL_SAMPLE_ROOT_V2_1  # use existing v2_1 as skeleton
    ctrl_v3_summary = _build_partial_v2_1_root(
        inherited_root=ctrl_v3_inherited,
        output_root=CTRL_BASE_ROOT_V3,
        parent_root=CTRL_PARENT_ROOT_V3,
        profile="controllability",
    )

    _replace_tree(CTRL_BASE_ROOT_V3, CTRL_SAMPLE_ROOT_V3)
    _write_json(
        CTRL_SAMPLE_ROOT_V3 / "manifest.json",
        {
            **ctrl_v3_summary,
            "manifest_dir": str(CTRL_SAMPLE_ROOT_V3),
            "mirrors_base_root": str(CTRL_BASE_ROOT_V3),
            "parent_version": "v3_scaled_full_pool",
        },
    )

    _build_variant_family_matrix(
        base_root=CTRL_BASE_ROOT_V3,
        output_root=VARIANT_FAMILY_ROOT_V3,
    )

    payload: Dict[str, Any] = {}
    if v5_summary is not None:
        payload["v5_invariance_samples_v2_1"] = {
            "root": str(V5_SAMPLE_ROOT_V2_1),
            "studies": v5_summary["studies"],
        }
    payload.update({
        "controllability_invariance_samples_v2_1": {
            "root": str(CTRL_SAMPLE_ROOT_V2_1),
            "studies": ctrl_summary["studies"],
        },
        "variant_family_v2_1": str(VARIANT_FAMILY_ROOT_V2_1),
        "controllability_invariance_samples_v3": {
            "root": str(CTRL_SAMPLE_ROOT_V3),
            "studies": ctrl_v3_summary["studies"],
        },
        "variant_family_v3": str(VARIANT_FAMILY_ROOT_V3),
    })
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
