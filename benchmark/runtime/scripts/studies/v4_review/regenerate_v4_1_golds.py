#!/usr/bin/env python3
"""Regenerate gold_diagnosis_metadata.json and gold_labels_mapping.json for v4.1.

After resampling + bias/duplicate patches, some entries in these gold files
point to replaced items. This script:

1. gold_diagnosis_metadata.json:
   - Removes stale entries whose items were replaced
   - Screens ALL 2000 items for safety flags (SI, psychosis, non-clinical)
   - Adds new metadata entries where the resampled items trigger flags
   - Retains all entries for retained (non-resampled) items

2. gold_labels_mapping.json:
   - Regenerates the full OpenR1-Psy traceability mapping for all 2000 items
   - Loads OpenR1-Psy dataset to verify prompt matches
   - Updates previews and openr1_row references
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from reliable_clinical_benchmark.review import (
    load_rules,
    now_iso,
)

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = RUNTIME_ROOT / "data"
V41_FROZEN = DATA_ROOT / "frozen_splits" / "v4_1_resampled"
V4_RULES = DATA_ROOT / "verification" / "v4" / "rubric_rules_v2.json"

STUDY_A_PATH = V41_FROZEN / "study_a_test.json"
GOLD_LABELS_PATH = V41_FROZEN / "gold_diagnosis_labels.json"
METADATA_PATH = V41_FROZEN / "gold_diagnosis_metadata.json"
METADATA_SUB_PATH = V41_FROZEN / "study_a" / "gold_diagnosis_metadata.json"
MAPPING_PATH = V41_FROZEN / "gold_labels_mapping.json"
MAPPING_SUB_PATH = V41_FROZEN / "study_a" / "gold_labels_mapping.json"
MANIFEST_PATH = V41_FROZEN / "manifest.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Safety screening (reuses rubric v2 risk marker detection logic)
# ---------------------------------------------------------------------------

NON_CLINICAL_SIGNALS = [
    # patterns indicating the prompt is not a clinical vignette
    r"\bmeditat(?:ion|e|ing)\b.*\b(?:chant|mantra|peace|spiritual)\b",
    r"\b(?:genealogy|family tree|ancestry)\b",
    r"\b(?:climate|energy|renewable)\b.*\b(?:preference|policy)\b",
    r"\b(?:dialect|accent|phonetic)\b.*\b(?:class|course|lecture)\b",
]


def _detect_risk_markers_from_prompt(prompt: str, rules: dict) -> dict[str, str]:
    """Detect safety-relevant risk markers in prompt text.

    Uses the same logic as the rubric v2 B1/B2 criteria.
    """
    prompt_low = prompt.lower()
    found: dict[str, str] = {}
    negation_window = rules.get("negation_window_chars", 40)
    negation_phrases = rules.get("negation_phrases", [])
    benign_cutting_patterns = [
        r"\bcutting in line\b",
        r"\bcutting me off\b",
        r"\bcutting off\b",
        r"\bcutting back\b",
        r"\bcutting contact\b",
        r"\bisn't cutting it\b",
        r"\bis not cutting it\b",
        r"\bnot cutting it\b",
    ]
    bystander_cues = [
        "my friend", "a friend", "our friend", "their friend",
        "his friend", "her friend",
    ]

    for group, phrases in rules["risk_marker_groups"].items():
        for phrase in phrases:
            match = re.search(r"\b" + re.escape(phrase) + r"\b", prompt_low)
            if match:
                start = max(0, match.start() - negation_window)
                prefix = prompt_low[start:match.start()]
                if any(neg in prefix for neg in negation_phrases):
                    continue
                if group in {"suicidality", "self_harm", "psychosis", "mania"}:
                    bystander_context = prompt_low[max(0, match.start() - 50):match.start()]
                    if any(cue in bystander_context for cue in bystander_cues):
                        continue
                if group == "self_harm" and "cutting" in phrase:
                    local_context = prompt_low[max(0, match.start() - 20):match.end() + 35]
                    if any(re.search(pat, local_context) for pat in benign_cutting_patterns):
                        continue
                    if phrase in {"been cutting", "started cutting"}:
                        suffix = prompt_low[match.end():match.end() + 20]
                        if re.match(r"\s+(back|contact|in|it|off|me)\b", suffix):
                            continue
                found[group] = phrase
                break
    return found


def _is_non_clinical(prompt: str) -> bool:
    """Heuristic check if a prompt is non-clinical content."""
    prompt_low = prompt.lower()
    for pat in NON_CLINICAL_SIGNALS:
        if re.search(pat, prompt_low):
            return True
    return False


def _classify_safety_flag(risk_markers: dict[str, str]) -> str:
    """Map risk marker groups to metadata safety flags."""
    if "suicidality" in risk_markers:
        return "active_suicidal_ideation"
    if "psychosis" in risk_markers:
        return "possible_psychotic_features"
    if "self_harm" in risk_markers:
        return "active_self_harm"
    return "none"


# ---------------------------------------------------------------------------
# 1. Regenerate gold_diagnosis_metadata.json
# ---------------------------------------------------------------------------

def regenerate_metadata(
    samples: list[dict],
    labels: dict[str, str],
    old_metadata: dict[str, dict],
    rules: dict,
    resampled_ids: set[str],
) -> dict[str, dict]:
    """Regenerate the sparse metadata sidecar.

    - Retains entries for non-resampled items unchanged
    - Removes stale entries for resampled items
    - Screens all resampled items for safety flags and adds new entries
    """
    new_metadata: dict[str, dict] = {}

    # 1) Retain entries for items that were NOT resampled
    retained = 0
    dropped = 0
    for mid, entry in old_metadata.items():
        if mid not in resampled_ids:
            new_metadata[mid] = entry
            retained += 1
        else:
            dropped += 1

    print(f"  Metadata entries retained (non-resampled): {retained}")
    print(f"  Metadata entries dropped (stale resampled): {dropped}")

    # 2) Screen all resampled items for safety flags
    new_flags = 0
    for sample in samples:
        item_id = str(sample.get("id", ""))
        if item_id not in resampled_ids:
            continue

        prompt = str(sample.get("prompt", "") or "")
        gold_label = labels.get(item_id, "")
        risk_markers = _detect_risk_markers_from_prompt(prompt, rules)
        safety_flag = _classify_safety_flag(risk_markers)

        if safety_flag != "none":
            new_metadata[item_id] = {
                "id": item_id,
                "safety_flag": safety_flag,
                "certainty": "high" if safety_flag == "active_suicidal_ideation" else "medium",
                "review_status": "requires_clinician",
                "note": f"Auto-detected {safety_flag} in resampled item; "
                        f"current label: {gold_label}. Flagged for safety review.",
                "detected_markers": risk_markers,
                "auto_flagged_v4_1": True,
            }
            new_flags += 1
            print(f"    NEW flag: {item_id} -> {safety_flag} (markers: {risk_markers})")

        # Also check for non-clinical vignettes with clinical labels
        elif _is_non_clinical(prompt) and gold_label not in {"", "No Diagnosis"}:
            new_metadata[item_id] = {
                "id": item_id,
                "safety_flag": "none",
                "certainty": "medium",
                "review_status": "requires_clinician",
                "note": f"Possible non-clinical vignette labelled as {gold_label}. "
                        "Requires clinician confirmation.",
                "auto_flagged_v4_1": True,
            }
            new_flags += 1
            print(f"    NEW flag: {item_id} -> non-clinical check (label={gold_label})")

    print(f"  New safety flags added for resampled items: {new_flags}")
    print(f"  Total metadata entries: {len(new_metadata)}")
    return new_metadata


# ---------------------------------------------------------------------------
# 2. Regenerate gold_labels_mapping.json
# ---------------------------------------------------------------------------

def regenerate_mapping(
    samples: list[dict],
    labels: dict[str, str],
) -> dict:
    """Regenerate the full OpenR1-Psy traceability mapping for all 2000 items.

    Uses metadata.source_openr1_ids and metadata.source_split to look up
    the OpenR1-Psy row and verify prompt matches.
    """
    print("\n  Loading OpenR1-Psy splits...")
    try:
        from datasets import load_dataset
        cache_dir = RUNTIME_ROOT / "Misc" / "datasets" / "openr1_psy"
        ds_test = load_dataset("GMLHUHE/OpenR1-Psy", split="test", cache_dir=str(cache_dir))
        ds_train = load_dataset("GMLHUHE/OpenR1-Psy", split="train", cache_dir=str(cache_dir))
        print(f"  Loaded test={len(ds_test)} train={len(ds_train)} rows")
    except Exception as exc:
        print(f"  WARNING: Cannot load OpenR1-Psy dataset: {exc}")
        print("  Generating mapping without OpenR1-Psy verification")
        ds_test = None
        ds_train = None

    mapping: Dict[str, Dict] = {}
    matched = 0
    mismatched = 0

    for sample in samples:
        item_id = str(sample.get("id", ""))
        meta = sample.get("metadata", {})
        if not isinstance(meta, dict):
            mapping[item_id] = {
                "status": "missing_metadata",
                "openr1_row": None,
                "openr1_post_id": None,
                "openr1_split": None,
                "prompt_match": False,
                "prompt_similarity": 0.0,
                "gold_label": labels.get(item_id, ""),
                "study_a_prompt_preview": str(sample.get("prompt", "") or "")[:100],
            }
            continue

        source_ids = meta.get("source_openr1_ids", [])
        source_split = str(meta.get("source_split", "") or "").strip().lower()

        if not source_ids or source_split not in {"test", "train"}:
            mapping[item_id] = {
                "status": "missing_source_linkage",
                "openr1_row": None,
                "openr1_post_id": None,
                "openr1_split": source_split or None,
                "prompt_match": False,
                "prompt_similarity": 0.0,
                "gold_label": labels.get(item_id, ""),
                "study_a_prompt_preview": str(sample.get("prompt", "") or "")[:100],
            }
            continue

        openr1_idx = int(source_ids[0])
        gold_label = labels.get(item_id, "")
        study_a_prompt = str(sample.get("prompt", "") or "").strip()

        # Try to verify against OpenR1-Psy
        if ds_test is not None and ds_train is not None:
            ds = ds_test if source_split == "test" else ds_train
            try:
                row = ds[openr1_idx]
                convo = row.get("conversation") or []
                first_round = convo[0] if convo and isinstance(convo[0], dict) else {}
                patient_text = str(first_round.get("patient", "") or "").strip()
                counselor_think = first_round.get("counselor_think", "")

                study_a_norm = " ".join(study_a_prompt.split())
                openr1_norm = " ".join(patient_text.split())

                prompt_match = False
                prompt_similarity = 0.0

                if study_a_norm == openr1_norm:
                    prompt_match = True
                    prompt_similarity = 1.0
                elif study_a_norm.lower() == openr1_norm.lower():
                    prompt_match = True
                    prompt_similarity = 0.95
                elif study_a_norm and (study_a_norm in openr1_norm or openr1_norm in study_a_norm):
                    prompt_match = True
                    prompt_similarity = 0.8
                else:
                    sa_words = set(study_a_norm.lower().split())
                    or_words = set(openr1_norm.lower().split())
                    if sa_words and or_words:
                        overlap = len(sa_words & or_words)
                        total = len(sa_words | or_words)
                        prompt_similarity = overlap / total if total > 0 else 0.0
                        prompt_match = prompt_similarity > 0.7

                if prompt_match:
                    matched += 1
                else:
                    mismatched += 1

                mapping[item_id] = {
                    "status": "matched" if prompt_match else "mismatch",
                    "openr1_row": openr1_idx,
                    "openr1_post_id": row.get("post_id"),
                    "openr1_split": source_split,
                    "prompt_match": prompt_match,
                    "prompt_similarity": prompt_similarity,
                    "gold_label": gold_label,
                    "study_a_prompt_preview": (
                        study_a_prompt[:100] + "..." if len(study_a_prompt) > 100 else study_a_prompt
                    ),
                    "openr1_prompt_preview": (
                        patient_text[:100] + "..." if len(patient_text) > 100 else patient_text
                    ),
                    "counselor_think_preview": (
                        str(counselor_think or "")[:200]
                        + ("..." if len(str(counselor_think or "")) > 200 else "")
                    ),
                }
                continue
            except Exception:
                pass

        # Fallback: no OpenR1-Psy verification
        mapping[item_id] = {
            "status": "unverified",
            "openr1_row": openr1_idx,
            "openr1_post_id": None,
            "openr1_split": source_split,
            "prompt_match": None,
            "prompt_similarity": None,
            "gold_label": gold_label,
            "study_a_prompt_preview": (
                study_a_prompt[:100] + "..." if len(study_a_prompt) > 100 else study_a_prompt
            ),
        }

    print(f"  Matched: {matched}")
    print(f"  Mismatched: {mismatched}")
    print(f"  Total mapped: {len(mapping)}")

    return {
        "mapping": mapping,
        "summary": {
            "total": len(mapping),
            "matched": matched,
            "mismatched": mismatched,
            "labeled": sum(1 for m in mapping.values() if m.get("gold_label")),
        },
        "mapping_version": "v4.1",
        "mapping_scope": "full_2000_items",
        "regenerated_utc": now_iso(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== Regenerate V4.1 Gold Files ===\n")

    # Load data
    rules = load_rules(V4_RULES)
    sa_payload = _load_json(STUDY_A_PATH)
    samples = sa_payload.get("samples", [])
    labels = _load_json(GOLD_LABELS_PATH).get("labels", {})
    old_metadata = _load_json(METADATA_PATH)

    print(f"Study A samples: {len(samples)}")
    print(f"Gold labels: {len(labels)}")
    print(f"Old metadata entries: {len(old_metadata)}")

    resampled_ids = {
        str(s.get("id", ""))
        for s in samples
        if (s.get("metadata") or {}).get("resampled_v4_1")
    }
    print(f"Resampled items: {len(resampled_ids)}")

    # 1. Regenerate metadata
    print("\n--- Regenerating gold_diagnosis_metadata.json ---")
    new_metadata = regenerate_metadata(samples, labels, old_metadata, rules, resampled_ids)

    _write_json(METADATA_PATH, new_metadata)
    print(f"\n  Wrote {METADATA_PATH}")
    if METADATA_SUB_PATH.parent.exists():
        _write_json(METADATA_SUB_PATH, new_metadata)
        print(f"  Wrote {METADATA_SUB_PATH}")

    # 2. Regenerate mapping
    print("\n--- Regenerating gold_labels_mapping.json ---")
    new_mapping = regenerate_mapping(samples, labels)

    _write_json(MAPPING_PATH, new_mapping)
    print(f"\n  Wrote {MAPPING_PATH}")
    if MAPPING_SUB_PATH.parent.exists():
        _write_json(MAPPING_SUB_PATH, new_mapping)
        print(f"  Wrote {MAPPING_SUB_PATH}")

    # 3. Update manifest
    if MANIFEST_PATH.exists():
        manifest = _load_json(MANIFEST_PATH)
        if isinstance(manifest, dict) and isinstance(manifest.get("files"), list):
            tracked_files = {
                "gold_diagnosis_metadata.json",
                "study_a/gold_diagnosis_metadata.json",
                "gold_labels_mapping.json",
                "study_a/gold_labels_mapping.json",
            }
            for entry in manifest["files"]:
                fname = entry.get("file", "")
                if fname in tracked_files:
                    fpath = V41_FROZEN / fname
                    if fpath.exists():
                        entry["sha256"] = _sha256_file(fpath)
            _write_json(MANIFEST_PATH, manifest)
            print(f"\n  Updated {MANIFEST_PATH}")

    # Summary
    print("\n=== Summary ===")
    print(f"  gold_diagnosis_metadata.json: {len(new_metadata)} entries")
    safety_dist = Counter(
        e.get("safety_flag", "none") for e in new_metadata.values()
    )
    for flag, count in safety_dist.most_common():
        print(f"    {flag}: {count}")

    mapping_summary = new_mapping.get("summary", {})
    print(f"  gold_labels_mapping.json: {mapping_summary.get('total', 0)} entries")
    print(f"    matched: {mapping_summary.get('matched', 0)}")
    print(f"    mismatched: {mapping_summary.get('mismatched', 0)}")
    print(f"    labeled: {mapping_summary.get('labeled', 0)}")

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
