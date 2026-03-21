#!/usr/bin/env python3
"""Harden Study B single-turn to v6 — source-backed rows only, no quota-fill.

Drop synthetic rows, keep only source-backed rows with provenance.
Optionally widen retrieval from OpenR1-Psy to fill gaps.

Run from runtime root:
    PYTHONPATH=src python scripts/studies/study_b/harden_study_b_single_v6.py
"""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List

from reliable_clinical_benchmark.pipeline.provenance import (
    ProvenanceType,
    build_provenance,
    serialise_provenance,
)

logger = logging.getLogger(__name__)

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
V5_ROOT = RUNTIME_ROOT / "data" / "frozen_splits" / "v5"
V6_ROOT = RUNTIME_ROOT / "data" / "frozen_splits" / "v6"


def _has_source_id(row: Dict[str, Any]) -> bool:
    """Check whether the row has a valid OpenR1 source ID."""
    meta = row.get("metadata", row.get("meta", {}))
    if isinstance(meta, dict):
        sid = meta.get("source_openr1_id") or meta.get("source_openr1_ids")
        if sid:
            return True
    # Top-level fallback
    return bool(row.get("source_openr1_id") or row.get("source_openr1_ids"))


def _extract_source_ids(row: Dict[str, Any]) -> List[int]:
    """Extract source IDs from row metadata."""
    meta = row.get("metadata", row.get("meta", {}))
    if isinstance(meta, dict):
        sid = meta.get("source_openr1_id")
        if sid is not None:
            return [int(sid)]
        sids = meta.get("source_openr1_ids")
        if isinstance(sids, list):
            return [int(s) for s in sids]
    sid = row.get("source_openr1_id")
    if sid is not None:
        return [int(sid)]
    sids = row.get("source_openr1_ids")
    if isinstance(sids, list):
        return [int(s) for s in sids]
    return []


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    V6_ROOT.mkdir(parents=True, exist_ok=True)

    src_path = V5_ROOT / "study_b_test.json"
    dst_path = V6_ROOT / "study_b_test.json"

    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list and dict-with-key formats
    if isinstance(data, list):
        rows = data
        wrap_key = None
    else:
        for key in ("cases", "samples", "data"):
            if key in data:
                rows = data[key]
                wrap_key = key
                break
        else:
            rows = data
            wrap_key = None

    logger.info("Loaded %d v5 Study B single-turn rows", len(rows))

    # Filter: keep only source-backed rows
    v6_rows: List[Dict[str, Any]] = []
    dropped = 0

    for row in rows:
        if not _has_source_id(row):
            dropped += 1
            continue

        # Add provenance metadata
        source_ids = _extract_source_ids(row)
        provenance = build_provenance(
            source_ids=source_ids,
            provenance_type=ProvenanceType.DIRECT_SOURCE,
        )

        v6_row = dict(row)
        meta = v6_row.setdefault("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
            v6_row["metadata"] = meta
        meta.update(serialise_provenance(provenance))
        v6_rows.append(v6_row)

    logger.info(
        "Study B single-turn v6: %d kept, %d dropped (no source ID)",
        len(v6_rows), dropped,
    )

    # Assertions
    for r in v6_rows:
        assert _has_source_id(r), f"Row {r.get('id')} still has no source ID"

    # Write output
    if wrap_key:
        output = dict(data)
        output[wrap_key] = v6_rows
    else:
        output = v6_rows

    output_text = json.dumps(output, indent=2, ensure_ascii=False)
    dst_path.write_text(output_text, encoding="utf-8")

    sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    print(f"Study B single-turn v6: {len(v6_rows)} rows -> {dst_path}")
    print(f"  SHA-256: {sha}")
    if dropped:
        print(f"  Dropped {dropped} synthetic/unsourced rows (fail closed)")


if __name__ == "__main__":
    main()
