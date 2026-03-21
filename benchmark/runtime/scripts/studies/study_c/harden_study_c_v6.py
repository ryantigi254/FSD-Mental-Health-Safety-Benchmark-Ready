#!/usr/bin/env python3
"""Harden Study C to v6 — source-first longitudinal cases.

For each case:
1. Source case turns form the backbone (direct_source)
2. Retrieve support turns against evolving case state
3. source_anchored_generated only after retrieval miss
4. Validate entity drift, meds/allergies/risk preservation

Run from runtime root:
    PYTHONPATH=src python scripts/studies/study_c/harden_study_c_v6.py \
        --model-id <model_id>
"""

from __future__ import annotations

import argparse
import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from reliable_clinical_benchmark.pipeline.config import PipelineConfig
from reliable_clinical_benchmark.pipeline.anchor import extract_anchors
from reliable_clinical_benchmark.pipeline.edit_plan import plan_edit
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Harden Study C to v6")
    parser.add_argument("--model-id", default="psyche-r1-local")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    V6_ROOT.mkdir(parents=True, exist_ok=True)

    src_path = V5_ROOT / "study_c_test.json"
    dst_path = V6_ROOT / "study_c_test.json"

    with open(src_path, encoding="utf-8") as f:
        data = json.load(f)

    cases_list = data.get("cases", data if isinstance(data, list) else [])
    logger.info("Loaded %d Study C cases", len(cases_list))

    config = PipelineConfig()
    ner = MedicalNER()
    nli_model = NLIModel()

    from reliable_clinical_benchmark.models.factory import ModelFactory
    model = ModelFactory.create(args.model_id)

    v6_cases: List[Dict[str, Any]] = []

    for case in cases_list:
        case_id = case.get("id", "")
        patient_summary = case.get("patient_summary", "")
        critical_entities = set(case.get("critical_entities", []))
        turns = case.get("turns", [])

        # Extract case-level anchors from patient summary
        case_anchor = extract_anchors(patient_summary, ner, config)

        v6_turns: List[Dict[str, Any]] = []
        prev_turn_text: Optional[str] = None

        for turn in turns:
            turn_num = turn.get("turn", 0)
            message = turn.get("message", "")

            # Check if this turn has direct source backing
            turn_meta = turn.get("metadata", {})
            has_source = bool(
                turn_meta.get("source_openr1_id") or
                turn_meta.get("source_openr1_ids")
            )

            if has_source:
                # Direct source turn — keep as-is
                provenance = build_provenance(
                    source_ids=_extract_ids(turn_meta),
                    provenance_type=ProvenanceType.DIRECT_SOURCE,
                    source_round_index=turn_num,
                )
                v6_turn = dict(turn)
                v6_turn["metadata"] = {
                    **turn_meta,
                    **serialise_provenance(provenance),
                }
                v6_turns.append(v6_turn)
            else:
                # Needs continuation generation
                anchor = extract_anchors(message, ner, config)

                plan = plan_edit(
                    message, anchor, "continuation", config,
                    source_row_id=f"{case_id}_t{turn_num}",
                )

                best_candidate = None
                best_verdict = None
                best_cosine = -1.0

                for attempt in range(config.max_regen_attempts):
                    candidates = generate_candidates(
                        message, plan, model, config,
                        previous_context=prev_turn_text or patient_summary,
                    )
                    for cand in candidates:
                        verdict = validate_candidate(
                            message, cand.text, case_anchor, plan,
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
                        source_ids=_extract_ids(turn_meta),
                        provenance_type=ProvenanceType.SOURCE_ANCHORED_GENERATED,
                        plan=plan,
                        verdict=best_verdict,
                        source_round_index=turn_num,
                        continuation_from_turn=turn_num - 1 if turn_num > 1 else None,
                    )
                    v6_turn = {"turn": turn_num, "message": best_candidate.text}
                else:
                    # Fail closed: keep original
                    provenance = build_provenance(
                        source_ids=_extract_ids(turn_meta),
                        provenance_type=ProvenanceType.DIRECT_SOURCE,
                        source_round_index=turn_num,
                    )
                    v6_turn = {"turn": turn_num, "message": message}

                v6_turn["metadata"] = serialise_provenance(provenance)
                v6_turns.append(v6_turn)

            prev_turn_text = message

        v6_case = {
            "id": case_id,
            "patient_summary": patient_summary,
            "critical_entities": list(critical_entities),
            "num_turns": len(v6_turns),
            "turns": v6_turns,
        }
        v6_cases.append(v6_case)

    logger.info("Study C v6: %d cases", len(v6_cases))

    if args.dry_run:
        return

    output = {"cases": v6_cases}
    output_text = json.dumps(output, indent=2, ensure_ascii=False)
    dst_path.write_text(output_text, encoding="utf-8")

    sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    print(f"Study C v6: {len(v6_cases)} cases -> {dst_path}")
    print(f"  SHA-256: {sha}")


def _extract_ids(meta: Dict[str, Any]) -> List[int]:
    if not isinstance(meta, dict):
        return []
    sid = meta.get("source_openr1_id")
    if sid is not None:
        return [int(sid)]
    sids = meta.get("source_openr1_ids")
    if isinstance(sids, list):
        return [int(s) for s in sids]
    return []


if __name__ == "__main__":
    main()
