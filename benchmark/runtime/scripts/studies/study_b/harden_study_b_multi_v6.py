#!/usr/bin/env python3
"""Harden Study B multi-turn to v6 — source-first with planned pressure operators.

For each case:
1. Copy direct OpenR1 source turns first (provenance = direct_source)
2. Retrieve compatible donor turns for remaining slots
3. Edit donor turns with planned pressure operators
4. Validate each turn through shared gates
5. Per-turn provenance with source_round_index and continuation_from_turn

Run from runtime root:
    PYTHONPATH=src python scripts/studies/study_b/harden_study_b_multi_v6.py \
        --model-id <model_id>
"""

from __future__ import annotations

import argparse
import json
import hashlib
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from reliable_clinical_benchmark.pipeline.config import PipelineConfig
from reliable_clinical_benchmark.pipeline.anchor import extract_anchors
from reliable_clinical_benchmark.pipeline.edit_plan import plan_edit, PRESSURE_TYPES
from reliable_clinical_benchmark.pipeline.generation import generate_candidates
from reliable_clinical_benchmark.pipeline.validation import validate_candidate
from reliable_clinical_benchmark.pipeline.provenance import (
    ProvenanceType,
    build_provenance,
    serialise_provenance,
)
from reliable_clinical_benchmark.utils.ner import MedicalNER
from reliable_clinical_benchmark.utils.nli import NLIModel

logger = logging.getLogger(__name__)

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
V5_ROOT = RUNTIME_ROOT / "data" / "frozen_splits" / "v5"
V6_ROOT = RUNTIME_ROOT / "data" / "frozen_splits" / "v6"


def _has_source_backing(turn_data: Dict[str, Any]) -> bool:
    """Check if a turn has direct OpenR1 source backing."""
    meta = turn_data.get("metadata", {})
    if isinstance(meta, dict):
        return bool(meta.get("source_openr1_id") or meta.get("source_openr1_ids"))
    return False


def _extract_source_ids(row: Dict[str, Any]) -> List[int]:
    meta = row.get("metadata", {})
    if isinstance(meta, dict):
        sid = meta.get("source_openr1_id")
        if sid is not None:
            return [int(sid)]
        sids = meta.get("source_openr1_ids")
        if isinstance(sids, list):
            return [int(s) for s in sids]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Harden Study B multi-turn to v6")
    parser.add_argument("--model-id", default="psyche-r1-local")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    random.seed(args.seed)

    V6_ROOT.mkdir(parents=True, exist_ok=True)

    src_path = V5_ROOT / "study_b_multi_turn_test.json"
    dst_path = V6_ROOT / "study_b_multi_turn_test.json"

    with open(src_path, encoding="utf-8") as f:
        v5_data = json.load(f)

    # Group turns by case_id
    if isinstance(v5_data, list):
        all_turns = v5_data
    else:
        all_turns = v5_data.get("turns", v5_data.get("data", []))

    cases: Dict[str, List[Dict]] = {}
    for turn in all_turns:
        cid = turn.get("case_id", "")
        cases.setdefault(cid, []).append(turn)

    logger.info("Loaded %d cases (%d total turns)", len(cases), len(all_turns))

    config = PipelineConfig()
    ner = MedicalNER()
    nli_model = NLIModel()

    from reliable_clinical_benchmark.models.factory import ModelFactory
    model = ModelFactory.create(args.model_id)

    v6_turns: List[Dict[str, Any]] = []

    for case_id, turns in sorted(cases.items()):
        turns.sort(key=lambda t: t.get("turn_num", 0))
        prev_turn_text: Optional[str] = None

        for turn in turns:
            turn_num = turn.get("turn_num", 0)
            # Extract patient text from conversation history
            conv = turn.get("conversation_history", [])
            patient_text = ""
            for msg in conv:
                if msg.get("role") == "user":
                    patient_text = msg.get("content", "")

            source_ids = _extract_source_ids(turn)
            has_source = bool(source_ids)

            if has_source:
                # Direct source turn — keep as-is with provenance
                provenance = build_provenance(
                    source_ids=source_ids,
                    provenance_type=ProvenanceType.DIRECT_SOURCE,
                    source_round_index=turn_num,
                )
                v6_turn = dict(turn)
                v6_turn.setdefault("metadata", {}).update(serialise_provenance(provenance))
                v6_turns.append(v6_turn)
            else:
                # Needs generation with pressure operator
                if not patient_text:
                    patient_text = turn.get("conversation_text", "")

                anchor = extract_anchors(patient_text, ner, config)
                pressure = random.choice(PRESSURE_TYPES)
                escalation = min(turn_num // 5, 3)

                plan = plan_edit(
                    patient_text, anchor, "pressure", config,
                    source_row_id=f"{case_id}_t{turn_num}",
                    pressure_type=pressure,
                    escalation_level=escalation,
                )

                best_candidate = None
                best_verdict = None
                best_cosine = -1.0

                for attempt in range(config.max_regen_attempts):
                    candidates = generate_candidates(
                        patient_text, plan, model, config,
                        previous_context=prev_turn_text or "",
                    )
                    for cand in candidates:
                        verdict = validate_candidate(
                            patient_text, cand.text, anchor, plan,
                            config, ner, nli_model,
                            prev_turn_text=prev_turn_text,
                            multi_turn=True,
                        )
                        if verdict.passed and verdict.cosine_similarity > best_cosine:
                            best_candidate = cand
                            best_verdict = verdict
                            best_cosine = verdict.cosine_similarity
                    if best_candidate is not None:
                        break

                if best_candidate is not None:
                    provenance = build_provenance(
                        source_ids=source_ids,
                        provenance_type=ProvenanceType.SOURCE_ANCHORED_GENERATED,
                        plan=plan,
                        verdict=best_verdict,
                        source_round_index=turn_num,
                        continuation_from_turn=turn_num - 1 if turn_num > 1 else None,
                    )
                    v6_turn = dict(turn)
                    # Update the patient content in conversation history
                    new_conv = list(conv)
                    for msg in new_conv:
                        if msg.get("role") == "user":
                            msg["content"] = best_candidate.text
                    v6_turn["conversation_history"] = new_conv
                    v6_turn.setdefault("metadata", {}).update(serialise_provenance(provenance))
                    v6_turns.append(v6_turn)
                else:
                    logger.warning("Case %s turn %d: failed, keeping original with provenance",
                                   case_id, turn_num)
                    provenance = build_provenance(
                        source_ids=source_ids,
                        provenance_type=ProvenanceType.DIRECT_SOURCE,
                        source_round_index=turn_num,
                    )
                    v6_turn = dict(turn)
                    v6_turn.setdefault("metadata", {}).update(serialise_provenance(provenance))
                    v6_turns.append(v6_turn)

            prev_turn_text = patient_text

    logger.info("Study B multi-turn v6: %d turns across %d cases",
                len(v6_turns), len(cases))

    if args.dry_run:
        logger.info("Dry run — not writing output")
        return

    output_text = json.dumps(v6_turns, indent=2, ensure_ascii=False)
    dst_path.write_text(output_text, encoding="utf-8")

    sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    print(f"Study B multi-turn v6: {len(v6_turns)} turns -> {dst_path}")
    print(f"  SHA-256: {sha}")


if __name__ == "__main__":
    main()
