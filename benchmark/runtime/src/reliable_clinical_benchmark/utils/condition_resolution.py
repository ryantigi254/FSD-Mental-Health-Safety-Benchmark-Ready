"""Shared condition-resolution helpers for OpenR1-derived studies."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple


CONDITION_ALIASES: Dict[str, str] = {
    "generalised anxiety disorder": "generalized anxiety disorder",
    "gad": "generalized anxiety disorder",
    "g.a.d.": "generalized anxiety disorder",
    "mdd": "major depressive disorder",
    "major depression": "major depressive disorder",
    "clinical depression": "major depressive disorder",
    "major depressive disorder (mdd)": "major depressive disorder",
    "ptsd": "post-traumatic stress disorder",
    "post traumatic stress disorder": "post-traumatic stress disorder",
    "attention deficit hyperactivity disorder": "attention-deficit/hyperactivity disorder",
    "attention deficit disorder": "attention-deficit/hyperactivity disorder",
    "adhd": "attention-deficit/hyperactivity disorder",
    "autism": "autism spectrum disorder",
    "autism spectrum condition": "autism spectrum disorder",
    "autism spectrum condition with sensory overload": "autism spectrum disorder",
    "alcohol use disorder (early recovery)": "alcohol use disorder",
    "aud": "alcohol use disorder",
    "alcoholism": "alcohol use disorder",
    "complicated grief": "prolonged grief disorder",
    "complicated grief disorder": "prolonged grief disorder",
    "complicated grief / prolonged grief": "prolonged grief disorder",
    "insomnia disorder": "insomnia",
    "sleep disorder": "insomnia",
    "emotionally unstable (borderline) personality disorder": "borderline personality disorder",
    "anger dysregulation in emotionally unstable personality structure": "borderline personality disorder",
    "eupd": "borderline personality disorder",
    "bpd": "borderline personality disorder",
    "bipolar ii disorder (depressive episode)": "bipolar disorder",
    "bipolar": "bipolar disorder",
    "ocd": "obsessive-compulsive disorder",
    "obsessive compulsive disorder": "obsessive-compulsive disorder",
    "obsessive-compulsive disorder (contamination subtype)": "obsessive-compulsive disorder",
    "obsessive-compulsive disorder (harm subtype)": "obsessive-compulsive disorder",
    "schizophrenic disorder": "schizophrenia",
    "schizoaffective": "schizophrenia",
    "psychosis (stable, on medication)": "psychosis",
    "somatic symptom": "somatic symptom disorder",
    "somatization disorder with health anxiety": "somatic symptom disorder",
    "ssd": "somatic symptom disorder",
}


EXPLICIT_CONDITION_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"\bmajor depressive disorder\b|\bmdd\b|\bmajor depression\b", "major depressive disorder"),
    (r"\bpersistent depressive disorder\b|\bdysthymi", "persistent depressive disorder"),
    (r"\bgeneralized anxiety disorder\b|\bgeneralised anxiety disorder\b|\bgad\b", "generalized anxiety disorder"),
    (r"\bsocial anxiety disorder\b|\bsocial phobia\b", "social anxiety disorder"),
    (r"\bpanic disorder\b", "panic disorder"),
    (r"\bpost[- ]traumatic stress disorder\b|\bptsd\b", "post-traumatic stress disorder"),
    (r"\bobsessive[- ]compulsive disorder\b|\bocd\b", "obsessive-compulsive disorder"),
    (r"\bbipolar disorder\b|\bbipolar ii\b|\bbipolar i\b", "bipolar disorder"),
    (r"\bschizophrenia\b", "schizophrenia"),
    (r"\bpsychosis\b|\bpsychotic\b", "psychosis"),
    (r"\bborderline personality disorder\b|\bbpd\b", "borderline personality disorder"),
    (r"\balcohol use disorder\b|\balcoholism\b", "alcohol use disorder"),
    (r"\bsubstance use disorder\b", "substance use disorder"),
    (r"\banorexia nervosa\b|\banorexi", "anorexia nervosa"),
    (r"\bbulimia nervosa\b|\bbulimi", "bulimia nervosa"),
    (r"\beating disorder\b", "eating disorder"),
    (r"\bautism spectrum disorder\b|\bautis", "autism spectrum disorder"),
    (r"\battention[- ]deficit/hyperactivity disorder\b|\battention deficit hyperactivity disorder\b|\badhd\b", "attention-deficit/hyperactivity disorder"),
    (r"\binsomnia\b", "insomnia"),
    (r"\bsomatic symptom disorder\b|\bsomat", "somatic symptom disorder"),
    (r"\bprolonged grief disorder\b|\bcomplicated grief\b", "prolonged grief disorder"),
    (r"\bself[- ]harm\b", "self-harm"),
    (r"\bsuicidal crisis\b|\bsuicid", "suicidal crisis"),
    (r"\bagoraphobia\b", "agoraphobia"),
    (r"\bspecific phobia\b", "specific phobia"),
    (r"\bbody dysmorphic disorder\b", "body dysmorphic disorder"),
    (r"\bchronic pain with depression\b", "chronic pain with depression"),
    (r"\badjustment disorder\b", "adjustment disorder"),
)


SYMPTOM_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"\bhand placement\b|\bcompulsion\b|\bintrusive\b|\bobsess(?:ion|ive)\b|\britual", "obsessive-compulsive disorder"),
    (r"\bhallucin|\bvoices?\b|\bparanoi|\bdelusion|\bpsychotic", "psychosis"),
    (r"\bsuicid|\boverdos|\bkill myself|\bend my life", "suicidal crisis"),
    (r"\bself[- ]harm|\bcutting\b|\bburning myself\b", "self-harm"),
    (r"\balcohol\b.*\b(control|problem|depend|recovery)\b|\bdrinking\b.*\btoo much\b", "alcohol use disorder"),
    (r"\bsubstance\b.*\b(use|depend|abuse)\b|\bdrug\b.*\b(use|depend|abuse|problem)\b", "substance use disorder"),
    (r"\binsomnia\b|\bcan'?t sleep\b|\bunable to sleep\b|\bsleep has been worse\b", "insomnia"),
    (r"\bborderline\b|\bfear of abandonment\b|\bunstable relationships\b", "borderline personality disorder"),
    (r"\banorexi|\bbulimi|\bbody image\b|\bafraid of gaining weight\b", "anorexia nervosa"),
    (r"\bautis|\bsensory overload\b", "autism spectrum disorder"),
    (r"\badhd\b|\binattentive\b|\bcan'?t focus\b|\bhyperactiv", "attention-deficit/hyperactivity disorder"),
    (r"\bsomatic\b|\bhealth anxiety\b|\bmedical reassurance\b", "somatic symptom disorder"),
    (r"\bchronic pain\b", "chronic pain with depression"),
    (r"\bgrief\b|\bbereave|\bloss of\b.*\b(parent|partner|spouse|child|friend)\b", "prolonged grief disorder"),
    (r"\bpanic attack\b|\bheart rac(?:es|ing)\b|\bpalpitations\b", "panic disorder"),
    (r"\bflashback|\bnightmare\b|\btrauma\b|\bhypervigil", "post-traumatic stress disorder"),
    (r"\bhopeless\b|\banhedoni|\blow mood\b|\bempty\b|\bpointless\b|\bget out of bed\b", "major depressive disorder"),
    (r"\banxi(?:ety|ous)\b|\bworry\b|\bon edge\b", "generalized anxiety disorder"),
    (r"\bsocial situations\b.*\bavoid\b|\bfear of embarrassment\b", "social anxiety disorder"),
)


NLI_CONDITION_CANDIDATES: Sequence[str] = (
    "major depressive disorder",
    "persistent depressive disorder",
    "generalized anxiety disorder",
    "social anxiety disorder",
    "panic disorder",
    "post-traumatic stress disorder",
    "obsessive-compulsive disorder",
    "bipolar disorder",
    "schizophrenia",
    "psychosis",
    "borderline personality disorder",
    "alcohol use disorder",
    "substance use disorder",
    "anorexia nervosa",
    "bulimia nervosa",
    "eating disorder",
    "autism spectrum disorder",
    "attention-deficit/hyperactivity disorder",
    "insomnia",
    "somatic symptom disorder",
    "prolonged grief disorder",
    "self-harm",
    "suicidal crisis",
    "agoraphobia",
    "specific phobia",
    "body dysmorphic disorder",
    "chronic pain with depression",
    "adjustment disorder",
)


def normalise_condition(label: str) -> str:
    text = str(label or "").strip().lower()
    return CONDITION_ALIASES.get(text, text)


def extract_explicit_condition(text: str) -> Optional[str]:
    haystack = str(text or "")
    if not haystack.strip():
        return None
    for pattern, condition in EXPLICIT_CONDITION_PATTERNS:
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            return normalise_condition(condition)
    return None


def infer_condition_from_symptoms(text: str) -> Optional[str]:
    haystack = str(text or "")
    if not haystack.strip():
        return None
    for pattern, condition in SYMPTOM_PATTERNS:
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            return normalise_condition(condition)
    return None


def score_condition_with_nli(
    *,
    patient_text: str,
    counselor_think: str,
    counselor_content: str,
    nli_model: Any,
    candidates: Sequence[str] = NLI_CONDITION_CANDIDATES,
    threshold: float = 0.5,
) -> Optional[str]:
    if nli_model is None:
        return None

    premise = "\n".join(
        part.strip()
        for part in (patient_text, counselor_think, counselor_content)
        if str(part or "").strip()
    )
    if not premise:
        return None

    best_label: Optional[str] = None
    best_score = 0.0
    for candidate in candidates:
        hypothesis = f"The clinical picture is most consistent with {candidate}."
        if hasattr(nli_model, "predict_with_score"):
            verdict, score = nli_model.predict_with_score(premise, hypothesis)
            effective_score = score if verdict == "entailment" else 0.0
        else:
            verdict = nli_model.predict(premise, hypothesis)
            effective_score = 1.0 if verdict == "entailment" else 0.0
        if effective_score > best_score:
            best_label = candidate
            best_score = effective_score

    if best_score >= threshold:
        return normalise_condition(best_label or "")
    return None


def resolve_case_condition(
    row: Dict[str, Any],
    *,
    nli_model: Any = None,
    nli_threshold: float = 0.5,
) -> Tuple[Optional[str], str]:
    patient_text = str(row.get("patient", "") or "")
    counselor_think = str(row.get("counselor_think", "") or "")
    counselor_content = str(row.get("counselor_content", "") or "")

    explicit_reasoning = extract_explicit_condition("\n".join((counselor_think, counselor_content)))
    if explicit_reasoning:
        return explicit_reasoning, "reasoning_explicit"

    explicit_patient = extract_explicit_condition(patient_text)
    if explicit_patient:
        return explicit_patient, "patient_explicit"

    symptom_match = infer_condition_from_symptoms(
        "\n".join((patient_text, counselor_think, counselor_content))
    )
    if symptom_match:
        return symptom_match, "symptom_heuristic"

    nli_match = score_condition_with_nli(
        patient_text=patient_text,
        counselor_think=counselor_think,
        counselor_content=counselor_content,
        nli_model=nli_model,
        threshold=nli_threshold,
    )
    if nli_match:
        return nli_match, "nli"

    return None, "unresolved"
