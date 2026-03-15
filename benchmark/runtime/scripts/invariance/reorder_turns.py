#!/usr/bin/env python3
"""Create deterministic turn-order variants for Study C invariance runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from _variant_common import (
    build_variant_metadata,
    copy_split_root,
    read_rows,
    write_json,
    write_rows,
)


def _is_anchor_turn(message: str, critical_entities: list[str]) -> bool:
    text = str(message or "").lower()
    for entity in critical_entities:
        if str(entity).lower() in text:
            return True
    return any(token in text for token in ("medication", "allergy", "plan", "dose", "suicid", "self-harm"))


def _reorder_noncritical_turns(turns: list[dict], critical_entities: list[str]) -> list[dict]:
    reordered = [dict(turn) for turn in turns]
    movable = [idx for idx, turn in enumerate(reordered) if not _is_anchor_turn(turn.get("message", ""), critical_entities)]
    if len(movable) >= 2:
        first, second = movable[0], movable[1]
        reordered[first], reordered[second] = reordered[second], reordered[first]
    for idx, turn in enumerate(reordered, start=1):
        turn["turn"] = idx
    return reordered


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic turn reorder variants for Study C.")
    parser.add_argument("--study", required=True, choices=["study_c"])
    parser.add_argument("--variant", default="noncritical_reorder", choices=["noncritical_reorder"])
    parser.add_argument("--base-root", type=Path, default=Path("benchmark/runtime/data/frozen_splits/v5_invariance_samples"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    copy_split_root(args.base_root, args.output_root)
    payload, rows = read_rows(args.output_root, args.study)
    changed_ids = []

    for row in rows:
        critical_entities = [str(value) for value in row.get("critical_entities", [])]
        row["turns"] = _reorder_noncritical_turns(row.get("turns", []), critical_entities)
        row.setdefault("metadata", {})["control_variant"] = args.variant
        changed_ids.append(str(row.get("id", "")))

    write_rows(args.output_root, args.study, payload, rows)
    metadata = build_variant_metadata(
        study=args.study,
        variant_tag=args.variant,
        source_root=args.base_root.resolve(),
        output_root=args.output_root.resolve(),
        changed_ids=changed_ids,
        seed=args.seed,
        notes={"variant_family": "reorder_turns"},
    )
    write_json(args.output_root / "variant_metadata.json", metadata)
    print(f"Wrote variant root {args.output_root}")
    print(f"Changed ids: {len(changed_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
