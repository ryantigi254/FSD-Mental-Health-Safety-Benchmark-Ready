"""Utilities for Study A diagnosis metadata resolution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


DEFAULT_STUDY_A_METADATA = {
    "safety_flag": "none",
    "review_status": "not_flagged",
    "certainty": "unknown",
}


def load_study_a_metadata_map(
    metadata_path: Optional[Union[str, Path]],
) -> Dict[str, Dict]:
    """Load sparse Study A metadata sidecar keyed by sample ID."""
    if metadata_path is None:
        return {}

    path = Path(metadata_path)
    if not path.exists():
        logger.warning(f"Study A metadata file not found: {path}")
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Failed to parse Study A metadata file {path}: {exc}")
        return {}

    if not isinstance(payload, dict):
        logger.warning(f"Study A metadata payload must be an object: {path}")
        return {}

    normalised: Dict[str, Dict] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            normalised[str(key)] = dict(value)
    return normalised


def _to_string(value, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def resolve_study_a_metadata(sample_id: str, metadata_map: Dict[str, Dict]) -> Dict[str, str]:
    """Resolve explicit metadata or defaults for a Study A sample ID."""
    sid = str(sample_id)
    explicit = metadata_map.get(sid)

    if isinstance(explicit, dict):
        resolved = dict(explicit)
        resolved["id"] = _to_string(resolved.get("id"), sid)
        resolved["safety_flag"] = _to_string(
            resolved.get("safety_flag"), DEFAULT_STUDY_A_METADATA["safety_flag"]
        )
        resolved["review_status"] = _to_string(
            resolved.get("review_status"), DEFAULT_STUDY_A_METADATA["review_status"]
        )
        resolved["certainty"] = _to_string(
            resolved.get("certainty"), DEFAULT_STUDY_A_METADATA["certainty"]
        )
        resolved["metadata_source"] = "explicit"
        return resolved

    return {
        "id": sid,
        "safety_flag": DEFAULT_STUDY_A_METADATA["safety_flag"],
        "review_status": DEFAULT_STUDY_A_METADATA["review_status"],
        "certainty": DEFAULT_STUDY_A_METADATA["certainty"],
        "metadata_source": "default",
    }
