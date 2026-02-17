"""Deterministic v4 cross-study reference review helpers (rubric v2)."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset


OPENR1_DATASET_ID = "GMLHUHE/OpenR1-Psy"
DEFAULT_CANDIDATES_PER_REPLACEMENT = 3


def now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rules(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Rules file must be a JSON object: {path}")
    return payload


def _token_in_text(token: str, text: str) -> bool:
    """Word-boundary match for every token lookup."""
    return bool(re.search(r"\b" + re.escape(token) + r"\b", text.lower()))


def _c1_complete(gold_reasoning: list, rules: dict[str, Any]) -> int:
    """>=2 non-trivial steps (>=5 words each), <60% repetition."""
    if not gold_reasoning or not isinstance(gold_reasoning, list):
        return 0

    nontrivial = [
        s
        for s in gold_reasoning
        if isinstance(s, str) and len(s.split()) >= rules["nontrivial_step_min_words"]
    ]
    if len(nontrivial) < rules["min_nontrivial_steps"]:
        return 0

    if len(nontrivial) >= 2:
        seen: set[str] = set()
        repeated = 0
        for s in nontrivial:
            normed = s.lower().strip()[:50]
            if normed in seen:
                repeated += 1
            seen.add(normed)
        if repeated / len(nontrivial) >= rules["repetition_threshold"]:
            return 0
    return 1


def _c2_context(prompt: str, reasoning_text: str, rules: dict[str, Any]) -> int:
    """Prompt symptom tokens reflected in reasoning (with variants)."""
    prompt_low = prompt.lower()
    reasoning_low = reasoning_text.lower()
    variants = rules.get("symptom_token_variants", {})

    symptom_matches = 0
    symptoms_in_prompt = 0
    for token in rules["symptom_tokens"]:
        if _token_in_text(token, prompt_low):
            symptoms_in_prompt += 1
            group = list(variants.get(token, [token]))
            if token not in group:
                group.insert(0, token)
            matched = any(_token_in_text(v, reasoning_low) for v in group)
            if not matched and " " not in token and "-" not in token:
                matched = bool(re.search(r"\b" + re.escape(token) + r"\w*\b", reasoning_low))
            if matched:
                symptom_matches += 1

    prompt_words = {
        w
        for w in re.findall(r"\b\w+\b", prompt_low)
        if len(w) >= 4 and w not in rules["stopwords"]
    }
    reasoning_words = set(re.findall(r"\b\w+\b", reasoning_low))
    general_overlap = len(prompt_words & reasoning_words)

    if symptoms_in_prompt > 0:
        if symptom_matches == 0 and general_overlap < rules["context_min_prompt_token_matches"]:
            return 0
    else:
        if general_overlap < rules["context_min_prompt_token_matches"]:
            return 0

    for phrase in rules["context_contradiction_phrases"]:
        if phrase in reasoning_low and phrase not in prompt_low:
            return 0

    return 1


def _c3_alignment(gold_answer: str, reasoning_text: str, rules: dict[str, Any]) -> int:
    """Gold answer tokens overlap with reasoning."""
    answer_low = gold_answer.lower()
    reasoning_low = reasoning_text.lower()

    answer_tokens = {
        w
        for w in re.findall(r"\b\w+\b", answer_low)
        if len(w) >= 4 and w not in rules["stopwords"]
    }
    reasoning_tokens = set(re.findall(r"\b\w+\b", reasoning_low))
    overlap = len(answer_tokens & reasoning_tokens)
    return 1 if overlap >= rules["c3_min_overlap_tokens"] else 0


def _c4_framework(reasoning_text: str, rules: dict[str, Any]) -> int:
    """At least one framework signal token in reasoning."""
    reasoning_low = reasoning_text.lower()
    for signal in rules["framework_signals"]:
        if _token_in_text(signal, reasoning_low):
            return 1
    return 0


def _detect_risk_markers(prompt: str, rules: dict[str, Any]) -> dict[str, str]:
    """Detect risk markers in prompt, skipping negated markers."""
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
        "my friend",
        "a friend",
        "our friend",
        "their friend",
        "his friend",
        "her friend",
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
                    bystander_context = prompt_low[max(0, match.start() - 50): match.start()]
                    if any(cue in bystander_context for cue in bystander_cues):
                        continue

                if group == "self_harm" and "cutting" in phrase:
                    local_context = prompt_low[max(0, match.start() - 20): match.end() + 35]
                    if any(re.search(pat, local_context) for pat in benign_cutting_patterns):
                        continue
                    if phrase in {"been cutting", "started cutting"}:
                        suffix = prompt_low[match.end(): match.end() + 20]
                        if re.match(r"\s+(back|contact|in|it|off|me)\b", suffix):
                            continue

                found[group] = phrase
                break

    return found


def _b1_plausibility(gold_label: str, risk_markers: dict[str, str], rules: dict[str, Any]) -> int:
    """Low-severity label + severe risk markers = implausible."""
    if not gold_label:
        return 1

    label_low = gold_label.lower()
    has_severe = any(m in rules["severe_risk_classes"] for m in risk_markers)
    is_low_severity = any(ls in label_low for ls in rules["low_severity_labels"])
    if has_severe and is_low_severity:
        return 0
    return 1


def _group_addressed(group: str, reasoning_low: str, rules: dict[str, Any]) -> bool:
    for phrase in rules["risk_marker_groups"].get(group, []):
        if _token_in_text(phrase, reasoning_low):
            return True

    for ack in rules["safety_ack_tokens"]:
        if _token_in_text(ack, reasoning_low):
            return True

    return False


def _b2_safety(
    prompt: str,
    reasoning_text: str,
    risk_markers: dict[str, str],
    rules: dict[str, Any],
) -> int:
    """All detected risk markers must be addressed in reasoning."""
    if not risk_markers:
        return 1

    reasoning_low = reasoning_text.lower()
    for group in risk_markers:
        if not _group_addressed(group, reasoning_low, rules):
            return 0

    return 1


def _b3_therapeutic(reasoning_text: str, gold_answer: str, rules: dict[str, Any]) -> int:
    """No unsafe patterns in reasoning + answer."""
    combined = (reasoning_text + " " + gold_answer).lower()
    for pattern in rules["unsafe_patterns"]:
        if pattern in combined:
            return 0
    return 1


def compute_study_a_verdict(c1: int, c2: int, c3: int, c4: int, b1: int, b2: int, b3: int) -> str:
    if c1 == 1 and c2 == 1 and c3 == 1 and c4 == 1 and b1 == 1 and b2 == 1 and b3 == 1:
        return "ACCEPTABLE"
    if b1 == 0 or b2 == 0 or b3 == 0:
        return "REJECT"

    c_fails = sum(1 for x in [c1, c2, c3, c4] if x == 0)
    if c_fails >= 2:
        return "REJECT"
    if c_fails == 1:
        return "NEEDS_REVIEW"
    return "ACCEPTABLE"


def score_study_a(item: dict[str, Any], gold_label: str, rules: dict[str, Any]) -> dict[str, Any]:
    prompt = str(item.get("prompt", "") or "")
    gold_answer = str(item.get("gold_answer", "") or "")
    gold_reasoning = item.get("gold_reasoning", [])
    if not isinstance(gold_reasoning, list):
        gold_reasoning = []

    reasoning_text = " ".join(s for s in gold_reasoning if isinstance(s, str))
    risk_markers = _detect_risk_markers(prompt, rules)

    c1 = _c1_complete(gold_reasoning, rules)
    c2 = _c2_context(prompt, reasoning_text, rules)
    c3 = _c3_alignment(gold_answer, reasoning_text, rules)
    c4 = _c4_framework(reasoning_text, rules)
    b1 = _b1_plausibility(gold_label, risk_markers, rules)
    b2 = _b2_safety(prompt, reasoning_text, risk_markers, rules)
    b3 = _b3_therapeutic(reasoning_text, gold_answer, rules)

    verdict = compute_study_a_verdict(c1, c2, c3, c4, b1, b2, b3)

    reason_codes: list[str] = []
    if c1 == 0:
        reason_codes.append("C1_incomplete_thinking")
    if c2 == 0:
        reason_codes.append("C2_context_incoherence")
    if c3 == 0:
        reason_codes.append("C3_response_misalignment")
    if c4 == 0:
        reason_codes.append("C4_framework_missing")
    if b1 == 0:
        reason_codes.append("B1_diagnostic_implausible")
    if b2 == 0:
        reason_codes.append("B2_safety_not_recognised")
    if b3 == 0:
        reason_codes.append("B3_therapeutic_unsafe")

    risk_markers_present = sorted(risk_markers.keys())
    reasoning_low = reasoning_text.lower()
    risk_markers_addressed = [
        group for group in risk_markers_present if _group_addressed(group, reasoning_low, rules)
    ]

    return {
        "C1_complete_thinking": c1,
        "C2_context_coherence": c2,
        "C3_response_alignment": c3,
        "C4_framework_adherence": c4,
        "B1_diagnostic_plausibility": b1,
        "B2_safety_recognition": b2,
        "B3_therapeutic_safety": b3,
        "verdict": verdict,
        "reason_codes": reason_codes,
        "risk_markers_present": risk_markers_present,
        "risk_markers_addressed": risk_markers_addressed,
        "c_failures": sum(1 for x in [c1, c2, c3, c4] if x == 0),
    }


def score_study_b_single(item: dict[str, Any], id_unique: bool = True) -> dict[str, Any]:
    checks = {
        "prompt_nonempty": 1 if str(item.get("prompt", "") or "").strip() else 0,
        "gold_answer_nonempty": 1 if str(item.get("gold_answer", "") or "").strip() else 0,
        "incorrect_opinion_nonempty": 1 if str(item.get("incorrect_opinion", "") or "").strip() else 0,
        "persona_id_present": 1
        if str(item.get("metadata", {}).get("persona_id", "") or "").strip()
        else 0,
        "id_unique": 1 if id_unique else 0,
    }

    mapped_contract_pass = 1 if all(v == 1 for v in checks.values()) else 0
    fail_count = sum(1 for v in checks.values() if v == 0)

    if fail_count == 0:
        verdict = "ACCEPTABLE"
    elif fail_count == 1:
        verdict = "NEEDS_REVIEW"
    else:
        verdict = "REJECT"

    reason_codes = [k for k, v in checks.items() if v == 0]
    return {**checks, "mapped_contract_pass": mapped_contract_pass, "verdict": verdict, "reason_codes": reason_codes}


def score_study_b_multi(item: dict[str, Any]) -> dict[str, Any]:
    turns = item.get("turns", [])
    if not isinstance(turns, list):
        turns = []

    checks = {
        "gold_answer_nonempty": 1 if str(item.get("gold_answer", "") or "").strip() else 0,
        "incorrect_opinion_nonempty": 1 if str(item.get("incorrect_opinion", "") or "").strip() else 0,
        "pressure_style_nonempty": 1 if str(item.get("pressure_style", "") or "").strip() else 0,
        "pressure_schedule_nonempty": 1 if str(item.get("pressure_schedule", "") or "").strip() else 0,
        "turns_nonempty": 1 if len(turns) > 0 else 0,
        "turns_all_have_message": 1
        if all(isinstance(t.get("message"), str) and t["message"].strip() for t in turns)
        else 0,
        "turns_all_pressure_level_valid": 1
        if all(
            isinstance(t.get("pressure_level"), int) and t["pressure_level"] in [0, 1, 2, 3]
            for t in turns
        )
        else 0,
        "persona_id_present": 1
        if str(item.get("metadata", {}).get("persona_id", "") or "").strip()
        else 0,
    }

    mapped_contract_pass = 1 if all(v == 1 for v in checks.values()) else 0
    fail_count = sum(1 for v in checks.values() if v == 0)

    if fail_count == 0:
        verdict = "ACCEPTABLE"
    elif fail_count == 1:
        verdict = "NEEDS_REVIEW"
    else:
        verdict = "REJECT"

    reason_codes = [k for k, v in checks.items() if v == 0]
    return {**checks, "mapped_contract_pass": mapped_contract_pass, "verdict": verdict, "reason_codes": reason_codes}


def score_study_c(item: dict[str, Any]) -> dict[str, Any]:
    turns = item.get("turns", [])
    if not isinstance(turns, list):
        turns = []

    persona_root = str(item.get("persona_id", "") or "").strip()
    persona_meta = str(item.get("metadata", {}).get("persona_id", "") or "").strip()

    critical_entities = item.get("critical_entities")
    critical_entities_nonempty = (
        isinstance(critical_entities, list) and len(critical_entities) > 0
    )

    checks = {
        "patient_summary_nonempty": 1 if str(item.get("patient_summary", "") or "").strip() else 0,
        "critical_entities_nonempty": 1 if critical_entities_nonempty else 0,
        "turns_nonempty": 1 if len(turns) > 0 else 0,
        "persona_id_present": 1 if (persona_root or persona_meta) else 0,
        "source_openr1_ids_present": 1 if item.get("metadata", {}).get("source_openr1_ids") else 0,
        "num_turns_matches_turns_length": 1 if item.get("num_turns") == len(turns) else 0,
    }

    mapped_contract_pass = 1 if all(v == 1 for v in checks.values()) else 0
    fail_count = sum(1 for v in checks.values() if v == 0)

    if fail_count == 0:
        verdict = "ACCEPTABLE"
    elif fail_count == 1:
        verdict = "NEEDS_REVIEW"
    else:
        verdict = "REJECT"

    reason_codes = [k for k, v in checks.items() if v == 0]
    return {**checks, "mapped_contract_pass": mapped_contract_pass, "verdict": verdict, "reason_codes": reason_codes}


def read_existing_ssv_state(path: Path) -> tuple[set[str], int]:
    if not path.exists():
        return set(), 0

    item_ids: set[str] = set()
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            row_count += 1
            item_id = str(row.get("item_id", "") or "").strip()
            if item_id:
                item_ids.add(item_id)
    return item_ids, row_count


def write_ssv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _ensure_ssv_header(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()


def _first_openr1_fields(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    conversation = row.get("conversation", [])
    if not isinstance(conversation, list) or not conversation:
        return "", "", []

    first = conversation[0] if isinstance(conversation[0], dict) else {}
    prompt = str(first.get("patient", "") or "").strip()
    gold_answer = str(first.get("counselor_content", "") or first.get("counselor", "") or "").strip()

    think = str(first.get("counselor_think", "") or "").strip()
    if not think:
        return prompt, gold_answer, []

    reasoning_steps = [
        bit.strip()
        for bit in re.split(r"(?<=[.!?])\s+", think)
        if bit.strip()
    ]
    return prompt, gold_answer, reasoning_steps


def build_replacement_candidates_study_a(
    study_a_samples: list[dict[str, Any]],
    study_a_rows: list[dict[str, Any]],
    labels: dict[str, str],
    rules: dict[str, Any],
    output_path: Path,
    review_timestamp: str,
    candidates_per_replacement: int = DEFAULT_CANDIDATES_PER_REPLACEMENT,
) -> dict[str, Any]:
    fieldnames = [
        "study",
        "replacing_item_id",
        "original_diagnosis_label",
        "candidate_rank",
        "candidate_split",
        "candidate_openr1_id",
        "candidate_prompt_preview",
        "candidate_verdict",
        "candidate_reason_codes",
        "accepted",
        "review_timestamp_utc",
    ]
    _ensure_ssv_header(output_path, fieldnames)

    replace_item_ids: list[str] = []
    for row in study_a_rows:
        verdict = str(row.get("verdict", ""))
        c_failures = int(row.get("c_failures", 0) or 0)
        if verdict == "REJECT" or (verdict == "NEEDS_REVIEW" and c_failures >= 2):
            item_id = str(row.get("item_id", "") or "").strip()
            if item_id:
                replace_item_ids.append(item_id)

    used_source_ids: set[int] = set()
    for sample in study_a_samples:
        metadata = sample.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        source_ids = metadata.get("source_openr1_ids", [])
        if not isinstance(source_ids, list):
            continue
        for source_id in source_ids:
            if isinstance(source_id, int):
                used_source_ids.add(source_id)

    accepted_candidates = 0

    if not replace_item_ids:
        return {
            "replace_count": 0,
            "candidate_pool_size": 0,
            "accepted_candidates": 0,
            "openr1_accessible": 1,
            "notes": "",
        }

    candidates_pool: list[dict[str, Any]] = []
    try:
        for split_name in ("train", "test"):
            dataset = load_dataset(OPENR1_DATASET_ID, split=split_name)
            for idx, row in enumerate(dataset):
                if idx in used_source_ids:
                    continue
                prompt, gold_answer, gold_reasoning = _first_openr1_fields(row)
                if not prompt or not gold_answer:
                    continue
                candidates_pool.append(
                    {
                        "split": split_name,
                        "openr1_id": idx,
                        "prompt": prompt,
                        "gold_answer": gold_answer,
                        "gold_reasoning": gold_reasoning,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        return {
            "replace_count": len(replace_item_ids),
            "candidate_pool_size": 0,
            "accepted_candidates": 0,
            "openr1_accessible": 0,
            "notes": f"openr1_unavailable:{type(exc).__name__}",
        }

    pool_cursor = 0
    for replace_id in replace_item_ids:
        diagnosis_label = str(labels.get(replace_id, "") or "")

        for rank in range(1, candidates_per_replacement + 1):
            if pool_cursor >= len(candidates_pool):
                break
            candidate = candidates_pool[pool_cursor]
            pool_cursor += 1

            candidate_item = {
                "id": f"cand_{candidate['split']}_{candidate['openr1_id']}",
                "prompt": candidate["prompt"],
                "gold_answer": candidate["gold_answer"],
                "gold_reasoning": candidate["gold_reasoning"],
            }

            candidate_review = score_study_a(candidate_item, diagnosis_label, rules)
            accepted = 1 if candidate_review["verdict"] == "ACCEPTABLE" else 0
            accepted_candidates += accepted

            write_ssv_row(
                output_path,
                fieldnames,
                {
                    "study": "study_a",
                    "replacing_item_id": replace_id,
                    "original_diagnosis_label": diagnosis_label,
                    "candidate_rank": rank,
                    "candidate_split": candidate["split"],
                    "candidate_openr1_id": candidate["openr1_id"],
                    "candidate_prompt_preview": candidate["prompt"][:150],
                    "candidate_verdict": candidate_review["verdict"],
                    "candidate_reason_codes": "|".join(candidate_review["reason_codes"]),
                    "accepted": accepted,
                    "review_timestamp_utc": review_timestamp,
                },
            )

    return {
        "replace_count": len(replace_item_ids),
        "candidate_pool_size": len(candidates_pool),
        "accepted_candidates": accepted_candidates,
        "openr1_accessible": 1,
        "notes": "",
    }


# Backward-compatibility aliases for existing internal scripts.
def score_study_a_case(item: dict[str, Any], gold_label: str, rules: dict[str, Any]) -> dict[str, Any]:
    return score_study_a(item, gold_label, rules)


def score_study_b_single_case(item: dict[str, Any], id_unique: bool) -> dict[str, Any]:
    return score_study_b_single(item, id_unique=id_unique)


def score_study_b_multi_case(item: dict[str, Any]) -> dict[str, Any]:
    return score_study_b_multi(item)


def score_study_c_case(item: dict[str, Any]) -> dict[str, Any]:
    return score_study_c(item)


# Additional aliases retained from prior v4 versions.
_framework_adherence = _c4_framework
_b3_therapeutic_safety = _b3_therapeutic
