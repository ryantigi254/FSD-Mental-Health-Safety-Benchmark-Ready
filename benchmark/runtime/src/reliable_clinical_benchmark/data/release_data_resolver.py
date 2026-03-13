"""Release-aware data source resolution for metrics pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Optional

DATA_SOURCE_CHOICES = ("latest_release", "working_data")

_LATEST_RELEASE_PATTERN = re.compile(r"Current canonical release:\s*`([^`]+)`")
_PRIMARY_PATH_PATTERN = re.compile(r"Primary path:\s*-\s*`([^`]+)`", re.MULTILINE)


@dataclass(frozen=True)
class MetricDataRoots:
    """Resolved data roots used by study metric calculators."""

    root: Path
    openr1_splits_dir: Path
    study_a_gold_dir: Path
    study_c_gold_dir: Path
    source: str
    release_name: Optional[str] = None
    layout: str = "release"


def _resolve_explicit_root(runtime_root: Path, explicit_root: Path) -> Path:
    root = explicit_root.expanduser()
    if not root.is_absolute():
        root = runtime_root / root
    return root.resolve()


def _resolve_latest_release_root(data_dir: Path) -> tuple[Path, str]:
    latest_path = data_dir / "releases" / "LATEST.md"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"Missing release pointer file: {latest_path}. "
            "Cannot resolve latest_release data source."
        )

    text = latest_path.read_text(encoding="utf-8")
    match = _LATEST_RELEASE_PATTERN.search(text)
    if not match:
        raise ValueError(
            f"Malformed release pointer file: {latest_path}. "
            "Expected 'Current canonical release: `...`'."
        )

    release_name = match.group(1).strip()
    if not release_name:
        raise ValueError(
            f"Malformed release pointer file: {latest_path}. "
            "Release name is empty."
        )

    local_release_root = (data_dir / "releases" / release_name).resolve()
    if local_release_root.exists():
        return local_release_root, release_name

    # Fallback to absolute primary path if present in LATEST.md.
    path_match = _PRIMARY_PATH_PATTERN.search(text)
    if path_match:
        primary_path = Path(path_match.group(1).strip()).expanduser()
        if primary_path.exists():
            return primary_path.resolve(), release_name

    raise FileNotFoundError(
        f"Release '{release_name}' from {latest_path} does not exist under "
        f"{data_dir / 'releases'} and no valid primary path was found."
    )


def _resolve_layout_dirs(root: Path) -> tuple[Path, Path, Path, str] | None:
    release_layout = (
        root / "openr1_psy_splits",
        root / "study_a_gold",
        root / "study_c_gold",
        "release",
    )
    if all(path.is_dir() for path in release_layout[:3]):
        return release_layout

    frozen_study_a_dir = root / "study_a"
    frozen_study_c_dir = root / "study_c"
    if frozen_study_a_dir.is_dir() and frozen_study_c_dir.is_dir():
        required_root_files = [
            root / "study_a_test.json",
            root / "study_b_test.json",
            root / "study_b_multi_turn_test.json",
            root / "study_c_test.json",
            frozen_study_a_dir / "gold_diagnosis_labels.json",
        ]
        if all(path.is_file() for path in required_root_files):
            return root, frozen_study_a_dir, frozen_study_c_dir, "frozen_snapshot"

    return None


def _validate_root_layout(root: Path, source: str) -> tuple[Path, Path, Path, str]:
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(
            f"Resolved data root does not exist for {source}: {root}"
        )

    layout_dirs = _resolve_layout_dirs(root)
    if layout_dirs is None:
        raise FileNotFoundError(
            f"Invalid data root for {source}. Expected either a release layout with "
            f"openr1_psy_splits/study_a_gold/study_c_gold or a frozen snapshot layout "
            f"with top-level study_* files plus study_a/ and study_c/ subdirectories: {root}"
        )

    openr1_splits_dir, study_a_gold_dir, study_c_gold_dir, layout = layout_dirs
    required_files = [
        openr1_splits_dir / "study_a_test.json",
        openr1_splits_dir / "study_b_test.json",
        openr1_splits_dir / "study_b_multi_turn_test.json",
        openr1_splits_dir / "study_c_test.json",
        study_a_gold_dir / "gold_diagnosis_labels.json",
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Invalid data root for {source}. Missing required files: "
            f"{', '.join(missing_files)}"
        )
    return openr1_splits_dir, study_a_gold_dir, study_c_gold_dir, layout


def resolve_metric_data_roots(
    runtime_root: Path,
    *,
    data_source: str = "latest_release",
    data_root: Optional[Path] = None,
) -> MetricDataRoots:
    """
    Resolve the root used by metrics scripts.

    Args:
        runtime_root: Benchmark runtime root (`benchmark/runtime`).
        data_source: One of `latest_release` or `working_data`.
        data_root: Optional explicit root override. Accepts absolute or runtime-relative.

    Returns:
        MetricDataRoots with resolved paths.

    Raises:
        ValueError: for malformed release pointers or invalid option values.
        FileNotFoundError: when required roots/files are missing.
    """

    runtime_root = Path(runtime_root).resolve()
    data_dir = runtime_root / "data"

    if data_root is not None:
        resolved_root = _resolve_explicit_root(runtime_root, data_root)
        source = f"explicit_root:{resolved_root}"
        release_name = None
    elif data_source == "working_data":
        resolved_root = data_dir.resolve()
        source = "working_data"
        release_name = None
    elif data_source == "latest_release":
        resolved_root, release_name = _resolve_latest_release_root(data_dir)
        source = f"latest_release:{release_name}"
    else:
        raise ValueError(
            f"Unsupported data_source '{data_source}'. "
            f"Expected one of: {', '.join(DATA_SOURCE_CHOICES)}"
        )

    openr1_splits_dir, study_a_gold_dir, study_c_gold_dir, layout = _validate_root_layout(
        resolved_root, source
    )

    return MetricDataRoots(
        root=resolved_root,
        openr1_splits_dir=openr1_splits_dir,
        study_a_gold_dir=study_a_gold_dir,
        study_c_gold_dir=study_c_gold_dir,
        source=source,
        release_name=release_name,
        layout=layout,
    )

