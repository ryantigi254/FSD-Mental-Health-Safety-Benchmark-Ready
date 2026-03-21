#!/usr/bin/env python3
"""Harden Study A Bias to v6 — full rebuild with embedded demographic cues.

Replaces visible 'Bias probe profile:' suffix probes with naturally
embedded counterfactual rewrites, validated through the shared pipeline.

Run from runtime root:
    PYTHONPATH=src python scripts/studies/study_a/harden_study_a_bias_v6.py \
        --model-id <model_id>
"""

from __future__ import annotations

import argparse
import json
import hashlib
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

from reliable_clinical_benchmark.pipeline.config import PipelineConfig
from reliable_clinical_benchmark.pipeline.anchor import extract_anchors
from reliable_clinical_benchmark.pipeline.edit_plan import plan_edit, BIAS_INSERTION_ANGLES
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
BIAS_DIR = "adversarial_bias"


def _strip_probe_suffix(prompt: str) -> str:
    """Remove legacy 'Bias probe profile:...' suffix from prompt text."""
    import re
    # Remove the bias probe line and any trailing instruction
    cleaned = re.sub(
        r"\n?Bias probe profile:.*$",
        "",
        prompt,
        flags=re.DOTALL,
    ).strip()
    # Also remove 'Assess this case clinically...' trailing instruction
    cleaned = re.sub(
        r"\n?Assess this case clinically.*$",
        "",
        cleaned,
        flags=re.DOTALL,
    ).strip()
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Harden Study A Bias to v6")
    parser.add_argument("--model-id", default="psyche-r1-local",
                        help="Model ID for candidate generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only, do not write output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    random.seed(args.seed)

    # Load v5 bias data
    src_path = V5_ROOT / BIAS_DIR / "biased_vignettes.json"
    with open(src_path, encoding="utf-8") as f:
        v5_data = json.load(f)
    cases: List[Dict[str, Any]] = v5_data.get("cases", [])
    logger.info("Loaded %d v5 bias cases from %s", len(cases), src_path)

    # Initialise pipeline components
    config = PipelineConfig()
    ner = MedicalNER()
    nli_model = NLIModel()

    # Load model for generation
    from reliable_clinical_benchmark.models.factory import ModelFactory
    model = ModelFactory.create(args.model_id)

    v6_cases: List[Dict[str, Any]] = []
    failed_ids: List[str] = []

    for i, case in enumerate(cases):
        case_id = case.get("id", f"abias_{i:04d}")
        bias_feature = case.get("bias_feature", "")
        bias_label = case.get("bias_label", "")

        # Strip legacy probe suffix to get clean source text
        source_text = _strip_probe_suffix(case.get("prompt", ""))
        if not source_text:
            logger.warning("Empty source text for %s, skipping", case_id)
            failed_ids.append(case_id)
            continue

        # Extract anchors from clean source
        anchor = extract_anchors(source_text, ner, config)

        # Choose insertion angle
        angle = random.choice(BIAS_INSERTION_ANGLES)

        # Plan the edit
        plan = plan_edit(
            source_text,
            anchor,
            "bias",
            config,
            source_row_id=case_id,
            insertion_angle=angle,
            bias_feature=bias_feature,
            bias_label=bias_label,
        )

        # Generate and validate candidates
        best_candidate = None
        best_verdict = None
        best_cosine = -1.0

        for attempt in range(config.max_regen_attempts):
            candidates = generate_candidates(source_text, plan, model, config)
            for cand in candidates:
                verdict = validate_candidate(
                    source_text, cand.text, anchor, plan,
                    config, ner, nli_model,
                )
                if verdict.passed and verdict.cosine_similarity > best_cosine:
                    best_candidate = cand
                    best_verdict = verdict
                    best_cosine = verdict.cosine_similarity

            if best_candidate is not None:
                break

        if best_candidate is None:
            logger.warning("Case %s: no passing candidate after %d attempts",
                           case_id, config.max_regen_attempts)
            failed_ids.append(case_id)
            continue

        # Build v6 case
        provenance = build_provenance(
            source_ids=[int(case.get("metadata", {}).get("source_openr1_id", 0))],
            provenance_type=ProvenanceType.SOURCE_ANCHORED_GENERATED,
            plan=plan,
            verdict=best_verdict,
        )

        v6_case = {
            "id": case_id,
            "prompt": best_candidate.text,
            "bias_feature": bias_feature,
            "bias_label": bias_label,
            "metadata": {
                **case.get("metadata", {}),
                **serialise_provenance(provenance),
            },
            "pair_group_id": case.get("pair_group_id", ""),
            "template_signature": "",  # rebuilt — old signature no longer applies
            "structure_version": "v6.0",
            "source_variant_count": case.get("source_variant_count", 1),
        }
        v6_cases.append(v6_case)

        if (i + 1) % 100 == 0:
            logger.info("Processed %d/%d cases (%d failed so far)",
                        i + 1, len(cases), len(failed_ids))

    logger.info("Rebuild complete: %d passed, %d failed",
                len(v6_cases), len(failed_ids))

    if args.dry_run:
        logger.info("Dry run — not writing output")
        return

    # Write v6 output
    out_dir = V6_ROOT / BIAS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_path = out_dir / "biased_vignettes.json"

    output = {"cases": v6_cases}
    output_text = json.dumps(output, indent=2, ensure_ascii=False)
    dst_path.write_text(output_text, encoding="utf-8")

    sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    print(f"Study A Bias v6: {len(v6_cases)} rows -> {dst_path}")
    print(f"  SHA-256: {sha}")
    if failed_ids:
        print(f"  Failed IDs ({len(failed_ids)}): {failed_ids[:10]}...")


if __name__ == "__main__":
    main()
