"""Deterministic v4 cross-study reference review helpers."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset


OPENR1_DATASET_ID = "GMLHUHE/OpenR1-Psy"
OPENR1_REVISION = "56fc0ef2fa5926df86713ed9b35f8689a6f85425"
DEFAULT_CANDIDATES_PER_REPLACEMENT = 3

WORD_RE = re.compile(r"[a-z0-9_+-]+")
WHITESPACE_RE = re.compile(r"\s+")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text or "").strip().lower())


def _tokenise(text: str) -> list[str]:
    return WORD_RE.findall(_norm(text))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rules(path: Path) -> dict[str, Any]:
    rules = _load_json(path)
    if not isinstance(rules, dict):
        raise ValueError(f"Invalid rules payload: {path}")
    return rules


def read_existing_ssv_state(path: Path) -> tuple[set[str], int]:
    if not path.exists():
        return set(), 0

    ids: set[str] = set()
    row_count = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            row_count += 1
            item_id = str(row.get("item_id", "") or "").strip()
            if item_id:
                ids.add(item_id)
    return ids, row_count


def write_ssv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _risk_classes_in_text(text: str, rules: dict[str, Any]) -> set[str]:
    text_n = _norm(text)
    out: set[str] = set()
    for cls, markers in (rules.get("risk_marker_groups") or {}).items():
        if any(_norm(marker) in text_n for marker in markers):
            out.add(cls)
    return out


def _reasoning_has_class(text: str, cls: str, rules: dict[str, Any]) -> bool:
    text_n = _norm(text)
    markers = (rules.get("risk_marker_groups") or {}).get(cls, [])
    if any(_norm(marker) in text_n for marker in markers):
        return True
    return any(_norm(tok) in text_n for tok in (rules.get("safety_ack_tokens") or []))


def _framework_adherence(reasoning_text: str, rules: dict[str, Any]) -> int:
    rt = _norm(reasoning_text)
    signals = rules.get("framework_signals") or []
    return 1 if any(_norm(s) in rt for s in signals) else 0


def _c1_complete(reasoning_steps: list[str], rules: dict[str, Any]) -> int:
    min_steps = int(rules.get("min_nontrivial_steps", 2))
    min_words = int(rules.get("nontrivial_step_min_words", 5))
    repetition_threshold = float(rules.get("repetition_threshold", 0.6))

    if not isinstance(reasoning_steps, list) or not reasoning_steps:
        return 0

    cleaned = []
    for s in reasoning_steps:
        if not isinstance(s, str):
            continue
        tokens = _tokenise(s)
        if len(tokens) >= min_words:
            cleaned.append(" ".join(tokens))

    if len(cleaned) < min_steps:
        return 0

    unique_ratio = len(set(cleaned)) / max(len(cleaned), 1)
    repetition = 1.0 - unique_ratio
    if repetition > repetition_threshold:
        return 0

    return 1


def _c2_context(prompt: str, reasoning_text: str, rules: dict[str, Any]) -> int:
    prompt_n = _norm(prompt)
    reasoning_n = _norm(reasoning_text)

    symptom_tokens = [tok for tok in (rules.get("symptom_tokens") or []) if _norm(tok) in prompt_n]
    min_matches = int(rules.get("context_min_prompt_token_matches", 1))

    if symptom_tokens:
        matched = sum(1 for tok in symptom_tokens if _norm(tok) in reasoning_n)
        if matched < min_matches:
            return 0

    contradiction_phrases = rules.get("context_contradiction_phrases") or []
    if any(_norm(p) in reasoning_n for p in contradiction_phrases) and symptom_tokens:
        return 0

    return 1


def _c3_alignment(gold_answer: str, reasoning_text: str, rules: dict[str, Any]) -> int:
    if not isinstance(gold_answer, str) or not gold_answer.strip():
        return 0
    rt = _norm(reasoning_text)

    stopwords = set(_norm(x) for x in (rules.get("stopwords") or []))
    answer_tokens = [
        t for t in _tokenise(gold_answer)
        if len(t) >= 4 and t not in stopwords
    ]

    if answer_tokens:
        overlap = sum(1 for t in set(answer_tokens) if t in rt)
        min_overlap = int(rules.get("c3_min_overlap_tokens", 1))
        return 1 if overlap >= min_overlap else 0

    return 1 if rt else 0


def _b1_plausibility(label: str, prompt: str, reasoning_text: str, rules: dict[str, Any]) -> int:
    if not isinstance(label, str) or not label.strip():
        return 0

    label_n = _norm(label)
    prompt_risks = _risk_classes_in_text(prompt, rules)

    lower_severity = set(_norm(x) for x in (rules.get("low_severity_labels") or []))
    severe_risks = set(_norm(x) for x in (rules.get("severe_risk_classes") or []))

    if label_n in lower_severity and any(r in severe_risks for r in prompt_risks):
        return 0

    contradiction_phrases = rules.get("diagnostic_contradiction_phrases") or []
    reasoning_n = _norm(reasoning_text)
    if any(_norm(p) in reasoning_n for p in contradiction_phrases) and label_n:
        return 0

    return 1


def _b2_safety(prompt: str, reasoning_text: str, rules: dict[str, Any]) -> tuple[int, list[str], list[str]]:
    present = sorted(_risk_classes_in_text(prompt, rules))
    if not present:
        return 1, [], []

    addressed = sorted([cls for cls in present if _reasoning_has_class(reasoning_text, cls, rules)])
    return (1 if len(addressed) == len(present) else 0), present, addressed


def _b3_therapeutic_safety(reasoning_text: str, gold_answer: str, rules: dict[str, Any]) -> int:
    text = _norm(f"{reasoning_text} {gold_answer}")
    unsafe_patterns = rules.get("unsafe_patterns") or []
    return 0 if any(_norm(p) in text for p in unsafe_patterns) else 1


def score_study_a_case(
    item: dict[str, Any],
    gold_label: str,
    rules: dict[str, Any],
) -> dict[str, Any]:
    required = ["id", "prompt", "gold_answer", "gold_reasoning"]
    missing = [k for k in required if k not in item]
    if missing or not isinstance(item.get("gold_reasoning"), list) or not isinstance(gold_label, str) or not gold_label.strip():
        return {
            "C1_complete_thinking": "",
            "C2_context_coherence": "",
            "C3_response_alignment": "",
            "C4_framework_adherence": "",
            "B1_diagnostic_plausibility": "",
            "B2_safety_recognition": "",
            "B3_therapeutic_safety": "",
            "verdict": "REJECT",
            "reason_codes": ["schema_missing_required_field"],
            "risk_markers_present": [],
            "risk_markers_addressed": [],
        }

    prompt = str(item.get("prompt", ""))
    gold_answer = str(item.get("gold_answer", ""))
    reasoning_steps = item.get("gold_reasoning", [])
    reasoning_text = " ".join(str(s) for s in reasoning_steps if isinstance(s, str))

    c1 = _c1_complete(reasoning_steps, rules)
    c2 = _c2_context(prompt, reasoning_text, rules)
    c3 = _c3_alignment(gold_answer, reasoning_text, rules)
    c4 = _framework_adherence(reasoning_text, rules)
    b1 = _b1_plausibility(gold_label, prompt, reasoning_text, rules)
    b2, risk_present, risk_addressed = _b2_safety(prompt, reasoning_text, rules)
    b3 = _b3_therapeutic_safety(reasoning_text, gold_answer, rules)

    reason_codes = []
    if c1 == 0:
        reason_codes.append("C1_incomplete_thinking")
    if c2 == 0:
        reason_codes.append("C2_context_incoherence")
    if c3 == 0:
        reason_codes.append("C3_response_mismatch")
    if c4 == 0:
        reason_codes.append("C4_framework_missing")
    if b1 == 0:
        reason_codes.append("B1_diagnostic_implausible")
    if b2 == 0:
        reason_codes.append("B2_safety_not_recognised")
    if b3 == 0:
        reason_codes.append("B3_therapeutic_unsafe")

    c_failures = sum(1 for v in (c1, c2, c3, c4) if v == 0)
    if all(v == 1 for v in (c1, c2, c3, c4, b1, b2, b3)):
        verdict = "ACCEPTABLE"
    elif c_failures == 1 and all(v == 1 for v in (b1, b2, b3)):
        verdict = "NEEDS_REVIEW"
    else:
        verdict = "REJECT"

    return {
        "C1_complete_thinking": str(c1),
        "C2_context_coherence": str(c2),
        "C3_response_alignment": str(c3),
        "C4_framework_adherence": str(c4),
        "B1_diagnostic_plausibility": str(b1),
        "B2_safety_recognition": str(b2),
        "B3_therapeutic_safety": str(b3),
        "verdict": verdict,
        "reason_codes": sorted(set(reason_codes)),
        "risk_markers_present": risk_present,
        "risk_markers_addressed": risk_addressed,
        "c_failures": c_failures,
    }


def _mapped_verdict(structural_errors: list[str], soft_failures: list[str]) -> str:
    if structural_errors:
        return "REJECT"
    if not soft_failures:
        return "ACCEPTABLE"
    if len(soft_failures) == 1:
        return "NEEDS_REVIEW"
    return "REJECT"


def score_study_b_single_case(item: dict[str, Any], id_unique: bool) -> dict[str, Any]:
    structural_errors: list[str] = []
    soft_failures: list[str] = []

    for key in ("id", "prompt", "gold_answer", "incorrect_opinion", "metadata"):
        if key not in item:
            structural_errors.append(f"missing_{key}")

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else None
    if metadata is None:
        structural_errors.append("metadata_not_object")
        metadata = {}

    prompt_nonempty = int(bool(isinstance(item.get("prompt"), str) and item.get("prompt", "").strip()))
    gold_answer_nonempty = int(bool(isinstance(item.get("gold_answer"), str) and item.get("gold_answer", "").strip()))
    incorrect_opinion_nonempty = int(bool(isinstance(item.get("incorrect_opinion"), str) and item.get("incorrect_opinion", "").strip()))
    persona_id_present = int(bool(isinstance(metadata.get("persona_id"), str) and metadata.get("persona_id", "").strip()))
    id_unique_int = int(id_unique)

    if not prompt_nonempty:
        soft_failures.append("prompt_empty")
    if not gold_answer_nonempty:
        soft_failures.append("gold_answer_empty")
    if not incorrect_opinion_nonempty:
        soft_failures.append("incorrect_opinion_empty")
    if not persona_id_present:
        soft_failures.append("persona_id_missing")
    if not id_unique_int:
        soft_failures.append("duplicate_id")

    verdict = _mapped_verdict(structural_errors, soft_failures)
    mapped_contract_pass = int(not structural_errors and not soft_failures)

    return {
        "prompt_nonempty": str(prompt_nonempty),
        "gold_answer_nonempty": str(gold_answer_nonempty),
        "incorrect_opinion_nonempty": str(incorrect_opinion_nonempty),
        "persona_id_present": str(persona_id_present),
        "id_unique": str(id_unique_int),
        "mapped_contract_pass": str(mapped_contract_pass),
        "verdict": verdict,
        "reason_codes": sorted(set(structural_errors + soft_failures)),
    }


def score_study_b_multi_case(item: dict[str, Any]) -> dict[str, Any]:
    structural_errors: list[str] = []
    soft_failures: list[str] = []

    for key in (
        "id",
        "gold_answer",
        "incorrect_opinion",
        "pressure_style",
        "pressure_schedule",
        "turns",
        "metadata",
    ):
        if key not in item:
            structural_errors.append(f"missing_{key}")

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else None
    if metadata is None:
        structural_errors.append("metadata_not_object")
        metadata = {}

    turns = item.get("turns")
    if not isinstance(turns, list):
        structural_errors.append("turns_not_list")
        turns = []
    elif not turns:
        structural_errors.append("turns_empty")

    gold_answer_nonempty = int(bool(isinstance(item.get("gold_answer"), str) and item.get("gold_answer", "").strip()))
    incorrect_opinion_nonempty = int(bool(isinstance(item.get("incorrect_opinion"), str) and item.get("incorrect_opinion", "").strip()))
    pressure_style_nonempty = int(bool(isinstance(item.get("pressure_style"), str) and item.get("pressure_style", "").strip()))
    pressure_schedule_nonempty = int(bool(isinstance(item.get("pressure_schedule"), str) and item.get("pressure_schedule", "").strip()))
    turns_nonempty = int(bool(turns))
    turns_all_have_message = int(
        bool(turns) and all(isinstance(t, dict) and isinstance(t.get("message"), str) and t.get("message", "").strip() for t in turns)
    )
    turns_all_pressure_level_valid = int(
        bool(turns)
        and all(
            isinstance(t, dict)
            and isinstance(t.get("pressure_level"), int)
            and 0 <= int(t.get("pressure_level")) <= 3
            for t in turns
        )
    )
    persona_id_present = int(bool(isinstance(metadata.get("persona_id"), str) and metadata.get("persona_id", "").strip()))

    if not gold_answer_nonempty:
        soft_failures.append("gold_answer_empty")
    if not incorrect_opinion_nonempty:
        soft_failures.append("incorrect_opinion_empty")
    if not pressure_style_nonempty:
        soft_failures.append("pressure_style_empty")
    if not pressure_schedule_nonempty:
        soft_failures.append("pressure_schedule_empty")
    if not turns_nonempty:
        soft_failures.append("turns_empty")
    if turns and not turns_all_have_message:
        structural_errors.append("turn_message_invalid")
    if turns and not turns_all_pressure_level_valid:
        structural_errors.append("turn_pressure_level_invalid")
    if not persona_id_present:
        soft_failures.append("persona_id_missing")

    verdict = _mapped_verdict(structural_errors, soft_failures)
    mapped_contract_pass = int(not structural_errors and not soft_failures)

    return {
        "gold_answer_nonempty": str(gold_answer_nonempty),
        "incorrect_opinion_nonempty": str(incorrect_opinion_nonempty),
        "pressure_style_nonempty": str(pressure_style_nonempty),
        "pressure_schedule_nonempty": str(pressure_schedule_nonempty),
        "turns_nonempty": str(turns_nonempty),
        "turns_all_have_message": str(turns_all_have_message),
        "turns_all_pressure_level_valid": str(turns_all_pressure_level_valid),
        "persona_id_present": str(persona_id_present),
        "mapped_contract_pass": str(mapped_contract_pass),
        "verdict": verdict,
        "reason_codes": sorted(set(structural_errors + soft_failures)),
    }


def score_study_c_case(item: dict[str, Any]) -> dict[str, Any]:
    structural_errors: list[str] = []
    soft_failures: list[str] = []

    for key in ("id", "patient_summary", "critical_entities", "turns", "metadata"):
        if key not in item:
            structural_errors.append(f"missing_{key}")

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else None
    if metadata is None:
        structural_errors.append("metadata_not_object")
        metadata = {}

    turns = item.get("turns")
    if not isinstance(turns, list):
        structural_errors.append("turns_not_list")
        turns = []

    entities = item.get("critical_entities")
    if not isinstance(entities, list):
        structural_errors.append("critical_entities_not_list")
        entities = []

    patient_summary_nonempty = int(bool(isinstance(item.get("patient_summary"), str) and item.get("patient_summary", "").strip()))
    critical_entities_nonempty = int(bool(entities) and all(isinstance(e, str) and e.strip() for e in entities))
    turns_nonempty = int(bool(turns))
    persona_id_present = int(bool(isinstance(metadata.get("persona_id"), str) and metadata.get("persona_id", "").strip()))
    source_openr1_ids_present = int(isinstance(metadata.get("source_openr1_ids"), list) and len(metadata.get("source_openr1_ids")) > 0)

    num_turns = item.get("num_turns")
    num_turns_matches_turns_length = int(isinstance(num_turns, int) and num_turns == len(turns))

    if not patient_summary_nonempty:
        soft_failures.append("patient_summary_empty")
    if not critical_entities_nonempty:
        soft_failures.append("critical_entities_empty")
    if not turns_nonempty:
        soft_failures.append("turns_empty")
    if not persona_id_present:
        structural_errors.append("persona_id_missing")
    if not source_openr1_ids_present:
        soft_failures.append("source_openr1_ids_missing")
    if not num_turns_matches_turns_length:
        soft_failures.append("num_turns_mismatch")

    verdict = _mapped_verdict(structural_errors, soft_failures)
    mapped_contract_pass = int(not structural_errors and not soft_failures)

    return {
        "patient_summary_nonempty": str(patient_summary_nonempty),
        "critical_entities_nonempty": str(critical_entities_nonempty),
        "turns_nonempty": str(turns_nonempty),
        "persona_id_present": str(persona_id_present),
        "source_openr1_ids_present": str(source_openr1_ids_present),
        "num_turns_matches_turns_length": str(num_turns_matches_turns_length),
        "mapped_contract_pass": str(mapped_contract_pass),
        "verdict": verdict,
        "reason_codes": sorted(set(structural_errors + soft_failures)),
    }


def _first_openr1_fields(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    conversation = row.get("conversation", [])
    if not isinstance(conversation, list) or not conversation:
        return "", "", []
    first = conversation[0] if isinstance(conversation[0], dict) else {}
    prompt = str(first.get("patient", "") or "").strip()
    gold_answer = str(first.get("counselor_content", "") or first.get("counselor", "") or "").strip()

    ct = str(first.get("counselor_think", "") or "").strip()
    if not ct:
        return prompt, gold_answer, []

    bits = [b.strip() for b in re.split(r"(?<=[.!?])\s+", ct) if b.strip()]
    return prompt, gold_answer, bits[:10]


def build_replacement_candidates_study_a(
    study_a_samples: list[dict[str, Any]],
    study_a_rows: list[dict[str, Any]],
    labels: dict[str, str],
    rules: dict[str, Any],
    output_path: Path,
    review_timestamp: str,
    candidates_per_replacement: int = DEFAULT_CANDIDATES_PER_REPLACEMENT,
) -> dict[str, int]:
    replace_ids = []
    for row in study_a_rows:
        verdict = row.get("verdict")
        c_failures = int(row.get("c_failures", 0))
        if verdict == "REJECT" or (verdict == "NEEDS_REVIEW" and c_failures >= 2):
            replace_ids.append(row["item_id"])

    sample_by_id = {str(s.get("id", "")): s for s in study_a_samples}

    used = set()
    for s in study_a_samples:
        metadata = s.get("metadata", {}) if isinstance(s.get("metadata", {}), dict) else {}
        for idx in metadata.get("source_openr1_ids", []) or []:
            if isinstance(idx, int):
                used.add(idx)

    candidates_pool: list[dict[str, Any]] = []
    for split_name in ("train", "test"):
        ds = load_dataset(OPENR1_DATASET_ID, split=split_name, revision=OPENR1_REVISION)
        for idx, row in enumerate(ds):
            if idx in used:
                continue
            prompt, gold_answer, reasoning = _first_openr1_fields(row)
            if not prompt or not gold_answer:
                continue
            candidates_pool.append(
                {
                    "split": split_name,
                    "openr1_id": idx,
                    "prompt": prompt,
                    "gold_answer": gold_answer,
                    "gold_reasoning": reasoning,
                }
            )

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

    if output_path.exists():
        output_path.unlink()

    accepted = 0
    pool_cursor = 0
    for replace_id in sorted(replace_ids):
        label = labels.get(replace_id, "")
        for rank in range(1, candidates_per_replacement + 1):
            if pool_cursor >= len(candidates_pool):
                break
            cand = candidates_pool[pool_cursor]
            pool_cursor += 1

            cand_item = {
                "id": f"cand_{cand['split']}_{cand['openr1_id']}",
                "prompt": cand["prompt"],
                "gold_answer": cand["gold_answer"],
                "gold_reasoning": cand["gold_reasoning"],
            }
            verdict = score_study_a_case(cand_item, label, rules)
            is_accepted = int(verdict["verdict"] == "ACCEPTABLE")
            accepted += is_accepted

            write_ssv_row(
                output_path,
                fieldnames,
                {
                    "study": "study_a_replacement_candidates",
                    "replacing_item_id": replace_id,
                    "original_diagnosis_label": label,
                    "candidate_rank": rank,
                    "candidate_split": cand["split"],
                    "candidate_openr1_id": cand["openr1_id"],
                    "candidate_prompt_preview": cand["prompt"][:200],
                    "candidate_verdict": verdict["verdict"],
                    "candidate_reason_codes": "|".join(verdict["reason_codes"]),
                    "accepted": is_accepted,
                    "review_timestamp_utc": review_timestamp,
                },
            )

    return {
        "replace_count": len(replace_ids),
        "candidate_pool_size": len(candidates_pool),
        "accepted_candidates": accepted,
    }
