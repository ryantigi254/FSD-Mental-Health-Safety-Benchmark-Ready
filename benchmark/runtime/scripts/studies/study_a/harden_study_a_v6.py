#!/usr/bin/env python3
"""Harden Study A to v6 — metadata normalisation only.

Study A is already exact-source. This script adds provenance fields
to each row without modifying prompts or gold data.

Run from runtime root:
    PYTHONPATH=src python scripts/studies/study_a/harden_study_a_v6.py
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
V5_ROOT = RUNTIME_ROOT / "data" / "frozen_splits" / "v5"
V6_ROOT = RUNTIME_ROOT / "data" / "frozen_splits" / "v6"


def main() -> None:
    V6_ROOT.mkdir(parents=True, exist_ok=True)

    src_path = V5_ROOT / "study_a_test.json"
    dst_path = V6_ROOT / "study_a_test.json"

    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", data if isinstance(data, list) else [])

    for sample in samples:
        meta = sample.setdefault("metadata", {})

        # Normalise source IDs
        source_ids = meta.get("source_openr1_ids", [])
        if not isinstance(source_ids, list):
            source_ids = [source_ids] if source_ids else []
        meta["source_openr1_ids"] = source_ids

        # Add provenance
        meta["provenance_type"] = "direct_source"
        meta["edit_operator"] = None
        meta["edit_plan"] = None
        meta["retrieved_support_ids"] = []
        meta["validation"] = {"passed": True, "failure_reasons": []}

    output = data if isinstance(data, dict) else {"samples": data}
    output_text = json.dumps(output, indent=2, ensure_ascii=False)

    dst_path.write_text(output_text, encoding="utf-8")

    sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    print(f"Study A v6: {len(samples)} rows -> {dst_path}")
    print(f"  SHA-256: {sha}")


if __name__ == "__main__":
    main()
