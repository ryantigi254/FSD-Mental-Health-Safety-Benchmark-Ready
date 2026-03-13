"""Unit tests for release-aware metric data root resolution."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from reliable_clinical_benchmark.data.release_data_resolver import (
    resolve_metric_data_roots,
)

BASE_DIR = Path(__file__).resolve().parents[3]


def _expected_latest_release_name(runtime_root: Path) -> str:
    latest_path = runtime_root / "data" / "releases" / "LATEST.md"
    text = latest_path.read_text(encoding="utf-8")
    match = re.search(r"Current canonical release:\s*`([^`]+)`", text)
    assert match, f"Unable to parse release name from {latest_path}"
    return match.group(1)


def _build_minimal_metric_root(root: Path) -> None:
    (root / "openr1_psy_splits").mkdir(parents=True, exist_ok=True)
    (root / "study_a_gold").mkdir(parents=True, exist_ok=True)
    (root / "study_c_gold").mkdir(parents=True, exist_ok=True)

    for rel_path in (
        "openr1_psy_splits/study_a_test.json",
        "openr1_psy_splits/study_b_test.json",
        "openr1_psy_splits/study_b_multi_turn_test.json",
        "openr1_psy_splits/study_c_test.json",
        "study_a_gold/gold_diagnosis_labels.json",
    ):
        path = root / rel_path
        path.write_text("{}", encoding="utf-8")


def _build_minimal_frozen_metric_root(root: Path) -> None:
    (root / "study_a").mkdir(parents=True, exist_ok=True)
    (root / "study_c").mkdir(parents=True, exist_ok=True)
    for rel_path in (
        "study_a_test.json",
        "study_b_test.json",
        "study_b_multi_turn_test.json",
        "study_c_test.json",
        "study_a/gold_diagnosis_labels.json",
        "study_c/study_c_target_plans.json",
    ):
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


@pytest.mark.unit
def test_resolve_metric_data_roots_latest_release_happy_path():
    expected_release = _expected_latest_release_name(BASE_DIR)

    resolved = resolve_metric_data_roots(BASE_DIR, data_source="latest_release")

    assert resolved.release_name == expected_release
    assert resolved.root == (BASE_DIR / "data" / "releases" / expected_release).resolve()
    assert resolved.openr1_splits_dir.is_dir()
    assert resolved.study_a_gold_dir.is_dir()
    assert resolved.study_c_gold_dir.is_dir()


@pytest.mark.unit
def test_resolve_metric_data_roots_fails_closed_on_malformed_latest(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    releases_dir = runtime_root / "data" / "releases"
    releases_dir.mkdir(parents=True)
    (releases_dir / "LATEST.md").write_text("invalid latest pointer", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed release pointer"):
        resolve_metric_data_roots(runtime_root, data_source="latest_release")


@pytest.mark.unit
def test_resolve_metric_data_roots_fails_closed_on_missing_release_layout(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    releases_dir = runtime_root / "data" / "releases"
    releases_dir.mkdir(parents=True)
    (releases_dir / "LATEST.md").write_text(
        "Current canonical release: `clinician_readiness_vx`\n",
        encoding="utf-8",
    )
    (releases_dir / "clinician_readiness_vx").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Expected either a release layout"):
        resolve_metric_data_roots(runtime_root, data_source="latest_release")


@pytest.mark.unit
def test_resolve_metric_data_roots_uses_explicit_override(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    explicit_root = runtime_root / "custom_data_root"
    _build_minimal_metric_root(explicit_root)

    releases_dir = runtime_root / "data" / "releases"
    releases_dir.mkdir(parents=True)
    (releases_dir / "LATEST.md").write_text("invalid latest pointer", encoding="utf-8")

    resolved = resolve_metric_data_roots(
        runtime_root,
        data_source="latest_release",
        data_root=Path("custom_data_root"),
    )

    assert resolved.root == explicit_root.resolve()
    assert resolved.source.startswith("explicit_root:")
    assert resolved.release_name is None


@pytest.mark.unit
def test_resolve_metric_data_roots_accepts_frozen_snapshot_override(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    explicit_root = runtime_root / "frozen_v5"
    _build_minimal_frozen_metric_root(explicit_root)

    releases_dir = runtime_root / "data" / "releases"
    releases_dir.mkdir(parents=True)
    (releases_dir / "LATEST.md").write_text("invalid latest pointer", encoding="utf-8")

    resolved = resolve_metric_data_roots(
        runtime_root,
        data_source="latest_release",
        data_root=Path("frozen_v5"),
    )

    assert resolved.root == explicit_root.resolve()
    assert resolved.openr1_splits_dir == explicit_root.resolve()
    assert resolved.study_a_gold_dir == (explicit_root / "study_a").resolve()
    assert resolved.study_c_gold_dir == (explicit_root / "study_c").resolve()
    assert resolved.layout == "frozen_snapshot"


@pytest.mark.unit
def test_resolve_metric_data_roots_rejects_frozen_snapshot_without_study_c_target_plans(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    explicit_root = runtime_root / "frozen_v5"
    _build_minimal_frozen_metric_root(explicit_root)
    (explicit_root / "study_c" / "study_c_target_plans.json").unlink()

    releases_dir = runtime_root / "data" / "releases"
    releases_dir.mkdir(parents=True)
    (releases_dir / "LATEST.md").write_text("invalid latest pointer", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Expected either a release layout"):
        resolve_metric_data_roots(
            runtime_root,
            data_source="latest_release",
            data_root=Path("frozen_v5"),
        )
