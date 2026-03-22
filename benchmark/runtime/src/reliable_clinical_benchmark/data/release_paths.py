"""Pinned clinician-readiness release layout under data/releases/.

Working-tree copies under data/adversarial_bias, data/study_a_gold, and
data/study_c_gold may be absent; loaders fall back to the bundled release.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

CLINICIAN_READINESS_RELEASE_ID = "clinician_readiness_v0.3_2026-02-16"

DEFAULT_ADVERSARIAL_VIGNETTES_RELPATH = (
    f"data/releases/{CLINICIAN_READINESS_RELEASE_ID}/adversarial_bias/biased_vignettes.json"
)


def release_root_under_data(data_root: Path) -> Path:
    return data_root / "releases" / CLINICIAN_READINESS_RELEASE_ID


def resolve_study_a_gold_dir(splits_file: Path) -> Optional[Path]:
    """Prefer data/study_a_gold; else the matching subtree in the release bundle."""
    data_root = splits_file.parent.parent
    legacy = data_root / "study_a_gold"
    if legacy.is_dir():
        return legacy
    bundled = release_root_under_data(data_root) / "study_a_gold"
    if bundled.is_dir():
        return bundled
    return None


def resolve_study_c_gold_dir(splits_file: Path) -> Optional[Path]:
    data_root = splits_file.parent.parent
    legacy = data_root / "study_c_gold"
    if legacy.is_dir():
        return legacy
    bundled = release_root_under_data(data_root) / "study_c_gold"
    if bundled.is_dir():
        return bundled
    return None


def resolve_adversarial_bias_vignettes_path(data_root: Path) -> Path:
    """Prefer data/adversarial_bias; else release bundle."""
    legacy = data_root / "adversarial_bias" / "biased_vignettes.json"
    if legacy.exists():
        return legacy
    return release_root_under_data(data_root) / "adversarial_bias" / "biased_vignettes.json"
