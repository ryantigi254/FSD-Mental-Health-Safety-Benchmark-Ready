"""Data loading for adversarial bias cases."""

import json
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def load_adversarial_bias_cases(data_path: str) -> List[Dict]:
    """
    Load adversarial bias test cases.

    Args:
        data_path: Path to biased_vignettes.json

    Returns:
        List of adversarial case dictionaries with 'prompt', 'bias_feature', 'bias_label'
    """
    path = Path(data_path)
    if not path.exists():
        logger.warning(f"Adversarial bias data file not found: {data_path}")
        return []

    with open(path, "r") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    logger.info(f"Loaded {len(cases)} adversarial bias cases from {data_path}")

    # Optional v3.1 structure-quality fields are non-breaking and informative only.
    if cases:
        structured_rows = 0
        personas = set()
        revisions = set()
        for case in cases:
            if (
                isinstance(case.get("pair_group_id"), str)
                and case["pair_group_id"].strip()
                and isinstance(case.get("template_signature"), str)
                and case["template_signature"].strip()
            ):
                structured_rows += 1
            metadata = case.get("metadata", {}) if isinstance(case, dict) else {}
            if isinstance(metadata, dict):
                persona_id = metadata.get("persona_id")
                revision = metadata.get("openr1_revision")
                if isinstance(persona_id, str) and persona_id.strip():
                    personas.add(persona_id.strip())
                if isinstance(revision, str) and revision.strip():
                    revisions.add(revision.strip())
        if structured_rows:
            logger.info(
                "Detected adversarial structure fields in %s/%s rows",
                structured_rows,
                len(cases),
            )
        if personas:
            logger.info("Adversarial persona coverage: %s personas", len(personas))
        if revisions:
            logger.info("Adversarial OpenR1 revisions present: %s", sorted(revisions))
    return cases
