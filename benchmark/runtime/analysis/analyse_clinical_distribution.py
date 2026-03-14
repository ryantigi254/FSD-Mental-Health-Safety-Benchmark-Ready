#!/usr/bin/env python3
"""
Clinical Specialty Distribution Analysis
=========================================
Analyses the frozen benchmark dataset (v4, 2026-02-22) and generates
paper-ready distribution figures analogous to OpenR1-Psy Figure 3.

Outputs:
  analysis/distribution_analysis.json   — structured counts
  analysis/figures/fig1_clinical_coverage.pdf
  analysis/figures/fig2_per_study_breakdown.pdf
  analysis/figures/fig3_therapeutic_modalities.pdf
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "frozen_splits" / "v4_1_resampled"
OUT  = Path(__file__).resolve().parent
FIG  = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BIAS_ROOT: Path | None = None
BIAS_SAMPLE_SIZE: int | None = None
BIAS_SAMPLE_SEED = 42

DEFAULT_STUDY_ORDER = ["A", "A_bias", "B", "B_multi", "C"]
DEFAULT_STUDY_LABELS = {
    "A": "Study A",
    "A_bias": "Study A Bias",
    "B": "Study B",
    "B_multi": "Study B multi-turn",
    "C": "Study C",
}
BIAS_CATEGORY = "Bias / Adversarial"

# ── DSM-5 Chapter Category Mapping ──────────────────────────────────────────
CONDITION_TO_CATEGORY = {
    # Mood Disorders
    "major depressive disorder": "Mood Disorders",
    "late-life depression": "Mood Disorders",
    "perinatal depression (antenatal)": "Mood Disorders",
    "depression (undiagnosed, resistant to help)": "Mood Disorders",
    "bipolar disorder": "Mood Disorders",
    "bipolar ii disorder (depressive episode)": "Mood Disorders",
    "persistent depressive disorder": "Mood Disorders",
    "something like bipolar disorder": "Mood Disorders",
    "burnout syndrome with depressive features": "Mood Disorders",

    # Anxiety Disorders
    "generalized anxiety disorder": "Anxiety Disorders",
    "generalised anxiety disorder": "Anxiety Disorders",
    "panic disorder": "Anxiety Disorders",
    "social anxiety disorder": "Anxiety Disorders",
    "agoraphobia": "Anxiety Disorders",
    "agoraphobia with panic disorder": "Anxiety Disorders",
    "specific phobia": "Anxiety Disorders",
    "specific phobia (flying)": "Anxiety Disorders",
    "exam anxiety": "Anxiety Disorders",
    "late-life anxiety with mild cognitive concerns": "Anxiety Disorders",
    "perinatal anxiety (postnatal)": "Anxiety Disorders",
    "gender dysphoria with social anxiety": "Anxiety Disorders",

    # Trauma & Stressor-Related
    "post-traumatic stress disorder": "Trauma & Stressor-Related",
    "complicated grief disorder": "Trauma & Stressor-Related",
    "complicated grief / prolonged grief": "Trauma & Stressor-Related",
    "prolonged grief disorder": "Trauma & Stressor-Related",
    "adjustment disorder": "Trauma & Stressor-Related",
    "adjustment disorder (relationship distress)": "Trauma & Stressor-Related",
    "dissociative episodes (comorbid with ptsd)": "Trauma & Stressor-Related",

    # OCD & Related
    "obsessive-compulsive disorder": "OCD & Related",
    "obsessive-compulsive disorder (contamination subtype)": "OCD & Related",
    "body dysmorphic disorder": "OCD & Related",

    # Personality Disorders
    "emotionally unstable (borderline) personality disorder": "Personality Disorders",
    "borderline personality disorder": "Personality Disorders",
    "anger dysregulation in emotionally unstable personality structure": "Personality Disorders",

    # Neurodevelopmental
    "attention-deficit/hyperactivity disorder": "Neurodevelopmental",
    "attention deficit hyperactivity disorder": "Neurodevelopmental",
    "autism spectrum condition with sensory overload": "Neurodevelopmental",
    "autism spectrum disorder": "Neurodevelopmental",

    # Psychotic Spectrum
    "schizophrenia": "Psychotic Spectrum",
    "schizophrenia (prodromal phase)": "Psychotic Spectrum",
    "psychosis (stable, on medication)": "Psychotic Spectrum",
    "evaluate for psychotic features \u2014 possible psychotic prodrome": "Psychotic Spectrum",
    "evaluate for psychotic features \u2014 auditory hallucination": "Psychotic Spectrum",

    # Eating Disorders
    "eating disorder": "Eating Disorders",
    "anorexia nervosa": "Eating Disorders",
    "anorexia nervosa (in recovery)": "Eating Disorders",

    # Substance Use Disorders
    "alcohol use disorder": "Substance Use Disorders",
    "alcohol use disorder (early recovery)": "Substance Use Disorders",
    "substance use disorder": "Substance Use Disorders",

    # Sleep-Wake Disorders
    "insomnia": "Sleep-Wake Disorders",
    "insomnia disorder": "Sleep-Wake Disorders",
    "sleep disorder": "Sleep-Wake Disorders",

    # Somatic & Health-Related
    "somatization disorder with health anxiety": "Somatic & Health-Related",
    "somatic symptom disorder": "Somatic & Health-Related",
    "health anxiety with tangential thinking": "Somatic & Health-Related",
    "chronic pain with depression": "Somatic & Health-Related",

    # Self-Harm & Suicidality
    "self-harm (active urges)": "Self-Harm & Suicidality",
    "suicidal crisis (active, high risk)": "Self-Harm & Suicidality",

    # Other
    "caregiver burnout": "Other",
    "body image concerns": "Other",
    "work-related stress": "Other",
    "no diagnosis": "Other",
}

# ── Severity Mapping (clinical judgement) ────────────────────────────────────
SEVERITY_MAP = {
    # Critical — immediate risk
    "suicidal crisis (active, high risk)": "Critical",
    "self-harm (active urges)": "Critical",

    # Severe — significant risk or complex presentations
    "schizophrenia": "Severe",
    "schizophrenia (prodromal phase)": "Severe",
    "psychosis (stable, on medication)": "Severe",
    "dissociative episodes (comorbid with ptsd)": "Severe",
    "emotionally unstable (borderline) personality disorder": "Severe",
    "borderline personality disorder": "Severe",
    "anger dysregulation in emotionally unstable personality structure": "Severe",
    "anorexia nervosa": "Severe",
    "anorexia nervosa (in recovery)": "Severe",
    "alcohol use disorder": "Severe",
    "alcohol use disorder (early recovery)": "Severe",
    "substance use disorder": "Severe",
    "bipolar disorder": "Severe",
    "bipolar ii disorder (depressive episode)": "Severe",
    "chronic pain with depression": "Severe",
    "evaluate for psychotic features \u2014 possible psychotic prodrome": "Severe",
    "evaluate for psychotic features \u2014 auditory hallucination": "Severe",

    # Moderate — clear clinical conditions
    "major depressive disorder": "Moderate",
    "post-traumatic stress disorder": "Moderate",
    "panic disorder": "Moderate",
    "agoraphobia with panic disorder": "Moderate",
    "obsessive-compulsive disorder": "Moderate",
    "obsessive-compulsive disorder (contamination subtype)": "Moderate",
    "body dysmorphic disorder": "Moderate",
    "eating disorder": "Moderate",
    "somatization disorder with health anxiety": "Moderate",
    "somatic symptom disorder": "Moderate",
    "health anxiety with tangential thinking": "Moderate",
    "generalized anxiety disorder": "Moderate",
    "generalised anxiety disorder": "Moderate",
    "social anxiety disorder": "Moderate",
    "attention-deficit/hyperactivity disorder": "Moderate",
    "attention deficit hyperactivity disorder": "Moderate",
    "autism spectrum condition with sensory overload": "Moderate",
    "autism spectrum disorder": "Moderate",
    "late-life depression": "Moderate",
    "perinatal depression (antenatal)": "Moderate",
    "depression (undiagnosed, resistant to help)": "Moderate",
    "complicated grief disorder": "Moderate",
    "complicated grief / prolonged grief": "Moderate",
    "prolonged grief disorder": "Moderate",
    "persistent depressive disorder": "Moderate",
    "gender dysphoria with social anxiety": "Moderate",
    "perinatal anxiety (postnatal)": "Moderate",
    "something like bipolar disorder": "Moderate",
    "late-life anxiety with mild cognitive concerns": "Moderate",

    # Mild — subclinical or adjustment-level
    "adjustment disorder": "Mild",
    "adjustment disorder (relationship distress)": "Mild",
    "insomnia": "Mild",
    "insomnia disorder": "Mild",
    "sleep disorder": "Mild",
    "exam anxiety": "Mild",
    "specific phobia": "Mild",
    "specific phobia (flying)": "Mild",
    "agoraphobia": "Mild",
    "body image concerns": "Mild",
    "work-related stress": "Mild",
    "caregiver burnout": "Mild",
    "burnout syndrome with depressive features": "Mild",
    "no diagnosis": "Mild",
}

# ── Therapeutic Modality Keyword Patterns ────────────────────────────────────
MODALITY_PATTERNS = {
    "CBT": [
        r"\bcbt\b", r"cognitive.behav", r"cognitive restructur",
        r"automatic thought", r"thought record", r"behavio(?:u)?ral experiment",
        r"cognitive distortion", r"thought challeng", r"socratic question",
    ],
    "Humanistic / Person-Centred": [
        r"humanistic", r"person.centr", r"unconditional positive regard",
        r"empathic understanding", r"congruence", r"self.actualiz",
        r"rogerian", r"client.centr",
    ],
    "Psychodynamic": [
        r"psychodynamic", r"unconscious", r"transference",
        r"defen[cs]e mechanism", r"psychoanalytic", r"attachment.based",
        r"object relation", r"internal working model",
    ],
    "Motivational Interviewing": [
        r"motivational interview", r"ambivalence", r"readiness to change",
        r"change talk", r"decisional balance",
    ],
    "ACT": [
        r"acceptance and commitment", r"psychological flexibility",
        r"defusion", r"values.based action", r"experiential avoidance",
    ],
    "DBT": [
        r"dialectical behav", r"\bdbt\b", r"distress tolerance",
        r"emotion regulation skill", r"radical acceptance", r"wise mind",
    ],
    "Mindfulness-Based": [
        r"mindfulness", r"mindful awareness", r"grounding exercise",
        r"body scan", r"meditation", r"present.moment",
    ],
    "Solution-Focused": [
        r"solution.focused", r"preferred future", r"miracle question",
        r"scaling question",
    ],
    "Trauma-Informed": [
        r"trauma.informed", r"trauma.focused", r"\bemdr\b",
        r"prolonged exposure therap", r"trauma processing",
    ],
}

# ── Style Configuration ──────────────────────────────────────────────────────
# Paper-style palette (inspired by reference figures)
BLUE = "#3B82B0"
BLUE_DARK = "#2B5E80"
TEAL = "#3C8D93"
GOLD = "#E9C46A"
GOLD_DARK = "#CDA349"
ROSE_LIGHT = "#F6C1C4"
ROSE_MID = "#F07B83"
ROSE_DARK = "#E10600"
TEXT_DARK = "#111827"
TEXT_MID = "#4B5563"
GRID_LIGHT = "#D8DEE9"
BG_LIGHT = "#FFFFFF"
BG_PANEL = "#FFFFFF"
WHITE = "#FFFFFF"

CATEGORY_COLOURS = {
    "Trauma & Stressor-Related": "#1F4E79",
    "Anxiety Disorders": "#2A9D8F",
    "Mood Disorders": "#4C78A8",
    "OCD & Related": "#7A5195",
    "Somatic & Health-Related": "#5F9EA0",
    "Eating Disorders": "#C56C86",
    "Neurodevelopmental": "#577590",
    "Psychotic Spectrum": "#B04A5A",
    "Personality Disorders": "#8E6C8A",
    "Self-Harm & Suicidality": "#8B1E3F",
    "Substance Use Disorders": "#B5651D",
    "Sleep-Wake Disorders": "#6C9A4F",
    "Other": "#6B7280",
    "Unclassified": "#9CA3AF",
}

# Severity donut colours (warm palette like OpenR1-Psy)
SEV_COLOURS = {
    "Critical": "#9B2226",
    "Severe": ROSE_DARK,
    "Moderate": ROSE_MID,
    "Mild": ROSE_LIGHT,
    "Unknown": "#9CA3AF",
}

# Per-study colours
STUDY_COLOURS = {
    "A": "#4B1D9A",
    "A_bias": "#9D4EDD",
    "B": "#F4A261",
    "B_multi": "#C96B27",
    "C": "#2A9D8F",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
    "font.size": 9.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": "#CFD8E3",
    "axes.linewidth": 0.8,
    "axes.facecolor": WHITE,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 8.4,
    "text.color": TEXT_DARK,
    "axes.labelcolor": TEXT_DARK,
    "xtick.color": TEXT_MID,
    "ytick.color": TEXT_MID,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.facecolor": BG_LIGHT,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def format_thousands(x, pos=None):
    return f"{int(x):,}"


def style_axis(ax, *, grid_axis="x"):
    ax.grid(axis=grid_axis, linestyle="-", linewidth=0.65, color=GRID_LIGHT, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))


def configure_paths(
    *,
    data_root: Path | None = None,
    out_dir: Path | None = None,
    bias_root: Path | None = None,
    bias_sample_size: int | None = None,
    bias_sample_seed: int = 42,
):
    global DATA, OUT, FIG, BIAS_ROOT, BIAS_SAMPLE_SIZE, BIAS_SAMPLE_SEED
    if data_root is not None:
        DATA = data_root.resolve()
    if out_dir is not None:
        OUT = out_dir.resolve()
    FIG = OUT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    BIAS_ROOT = bias_root.resolve() if bias_root else None
    BIAS_SAMPLE_SIZE = bias_sample_size
    BIAS_SAMPLE_SEED = bias_sample_seed


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_study_a():
    """Study A: faithfulness evaluation, 2000 single-turn samples."""
    with open(DATA / "study_a_test.json") as f:
        raw = json.load(f)
    samples = raw["samples"]

    with open(DATA / "gold_diagnosis_labels.json") as f:
        gold = json.load(f)
    labels = gold["labels"]

    results = []
    for s in samples:
        sid = s["id"]
        diag = labels.get(sid, "Unknown")
        reasoning = " ".join(s.get("gold_reasoning", []))
        results.append({
            "id": sid,
            "study": "A",
            "condition": diag.lower().strip(),
            "reasoning": reasoning,
            "prompt": s.get("prompt", ""),
            "gold_answer": s.get("gold_answer", ""),
        })
    return results


def load_study_b():
    """Study B: sycophancy evaluation, 2000 single-turn samples."""
    with open(DATA / "study_b_test.json") as f:
        samples = json.load(f)

    results = []
    for s in samples:
        results.append({
            "id": s["id"],
            "study": "B",
            "condition": s.get("gold_answer", "Unknown").lower().strip(),
            "reasoning": "",
            "prompt": s.get("prompt", ""),
            "gold_answer": s.get("gold_answer", ""),
        })
    return results


def _stable_sample_cases(cases: list[dict], sample_size: int, *, seed: int) -> list[dict]:
    if sample_size >= len(cases):
        return list(cases)
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(len(cases), size=sample_size, replace=False).tolist())
    return [cases[index] for index in selected]


def load_study_a_bias(default_sample_size: int | None = None):
    """Load Study A bias cases, optionally from a fallback root."""
    local_path = DATA / "adversarial_bias" / "biased_vignettes.json"
    fallback_path = (
        (BIAS_ROOT / "adversarial_bias" / "biased_vignettes.json")
        if BIAS_ROOT is not None
        else None
    )

    source_path: Path | None = None
    used_fallback = False
    if local_path.exists():
        source_path = local_path
    elif fallback_path is not None and fallback_path.exists():
        source_path = fallback_path
        used_fallback = True

    if source_path is None:
        return [], {
            "available": False,
            "used_fallback": False,
            "sampled": False,
            "sample_size": 0,
            "source_path": None,
        }

    with open(source_path) as f:
        raw = json.load(f)
    cases = list(raw.get("cases", raw) if isinstance(raw, dict) else raw)

    sample_size = BIAS_SAMPLE_SIZE
    if sample_size is None and used_fallback and default_sample_size is not None:
        sample_size = default_sample_size
    if sample_size is not None:
        cases = _stable_sample_cases(cases, sample_size, seed=BIAS_SAMPLE_SEED)

    results = []
    for case in cases:
        results.append({
            "id": case["id"],
            "study": "A_bias",
            "condition": str(case.get("bias_label") or "bias probe").lower().strip(),
            "reasoning": "",
            "prompt": case.get("prompt", ""),
            "gold_answer": "",
        })

    return results, {
        "available": True,
        "used_fallback": used_fallback,
        "sampled": sample_size is not None,
        "sample_size": len(results),
        "source_path": str(source_path),
    }


def load_study_b_multi_turn():
    """Study B multi-turn: sustained pressure conversations."""
    multi_turn_path = DATA / "study_b_multi_turn_test.json"
    with open(multi_turn_path) as f:
        raw = json.load(f)

    cases = raw if isinstance(raw, list) else raw.get("multi_turn_cases", [])
    results = []
    for case in cases:
        results.append({
            "id": case["id"],
            "study": "B_multi",
            "condition": case.get("gold_answer", "Unknown").lower().strip(),
            "reasoning": "",
            "prompt": case.get("turns", [{}])[0].get("message", "") if case.get("turns") else "",
            "gold_answer": case.get("gold_answer", ""),
        })
    return results


def load_study_c():
    """Study C: longitudinal drift, 100 multi-turn cases."""
    with open(DATA / "study_c_test.json") as f:
        raw = json.load(f)

    results = []
    for case in raw["cases"]:
        entities = case.get("critical_entities", [])
        cond = entities[0].lower().strip() if entities else "unknown"
        results.append({
            "id": case["id"],
            "study": "C",
            "condition": cond,
            "reasoning": "",
            "prompt": case.get("patient_summary", ""),
            "gold_answer": "",
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_category(condition: str, *, study: str | None = None) -> str:
    """Map a specific condition to its DSM-5 chapter category."""
    if study == "A_bias":
        return BIAS_CATEGORY
    return CONDITION_TO_CATEGORY.get(condition, "Unclassified")


def classify_severity(condition: str, *, study: str | None = None) -> str:
    """Map a specific condition to its severity level."""
    if study == "A_bias":
        return "Unknown"
    return SEVERITY_MAP.get(condition, "Unknown")


def detect_modalities(text: str) -> list[str]:
    """Detect therapeutic modalities mentioned in reasoning text."""
    text_lower = text.lower()
    found = []
    for modality, patterns in MODALITY_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                found.append(modality)
                break
    return found


def normalise_condition_name(cond: str) -> str:
    """Capitalise condition name for display."""
    # Special acronyms
    acronyms = {"ocd", "ptsd", "adhd", "dbt", "cbt", "act", "emdr", "bpd"}
    words = cond.split()
    result = []
    for w in words:
        if w.lower() in acronyms:
            result.append(w.upper())
        elif w.startswith("("):
            result.append("(" + w[1:].capitalize())
        else:
            # Don't capitalise articles/prepositions in middle
            if w.lower() in {"with", "and", "in", "of", "on", "for", "the", "a"} and result:
                result.append(w.lower())
            else:
                result.append(w.capitalize())
    return " ".join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis():
    print("Loading data...")
    study_a = load_study_a()
    study_a_bias, bias_meta = load_study_a_bias(default_sample_size=len(study_a))
    study_b = load_study_b()
    study_b_multi = load_study_b_multi_turn()
    study_c = load_study_c()
    samples_by_study = {
        "A": study_a,
        "A_bias": study_a_bias,
        "B": study_b,
        "B_multi": study_b_multi,
        "C": study_c,
    }
    active_study_order = [key for key in DEFAULT_STUDY_ORDER if samples_by_study[key]]
    all_samples = []
    for study_key in active_study_order:
        all_samples.extend(samples_by_study[study_key])

    print(f"  Study A: {len(study_a)} samples")
    if study_a_bias:
        fallback_note = " (fallback)" if bias_meta["used_fallback"] else ""
        sampled_note = " sampled" if bias_meta["sampled"] else ""
        print(f"  Study A Bias: {len(study_a_bias)} cases{sampled_note}{fallback_note}")
    print(f"  Study B: {len(study_b)} samples")
    if study_b_multi:
        print(f"  Study B multi-turn: {len(study_b_multi)} cases")
    print(f"  Study C: {len(study_c)} cases")
    print(f"  Total:   {len(all_samples)}")

    # ── Diagnostic categories ────────────────────────────────────────────
    print("\nClassifying diagnostic categories...")
    category_counts = Counter()
    category_per_study = defaultdict(lambda: Counter())
    condition_counts = Counter()
    condition_per_study = defaultdict(lambda: Counter())

    for s in all_samples:
        cat = classify_category(s["condition"], study=s["study"])
        category_counts[cat] += 1
        category_per_study[s["study"]][cat] += 1
        condition_counts[s["condition"]] += 1
        condition_per_study[s["study"]][s["condition"]] += 1

    unclassified = [
        c for c in condition_counts
        if classify_category(c) == "Unclassified" and condition_per_study["A_bias"].get(c, 0) == 0
    ]
    if unclassified:
        print(f"  WARNING: {len(unclassified)} unclassified conditions:")
        for c in unclassified:
            print(f"    '{c}' ({condition_counts[c]} samples)")

    # ── Severity ─────────────────────────────────────────────────────────
    print("\nClassifying severity levels...")
    severity_counts = Counter()
    severity_per_study = defaultdict(lambda: Counter())

    for s in all_samples:
        sev = classify_severity(s["condition"], study=s["study"])
        severity_counts[sev] += 1
        severity_per_study[s["study"]][sev] += 1

    unknown_sev = [
        (c, condition_counts[c])
        for c in condition_counts
        if classify_severity(c) == "Unknown" and condition_per_study["A_bias"].get(c, 0) == 0
    ]
    if unknown_sev:
        print(f"  WARNING: {len(unknown_sev)} conditions with unknown severity:")
        for c, n in unknown_sev:
            print(f"    '{c}' ({n} samples)")

    # ── Therapeutic modalities (Study A only — has gold reasoning) ────────
    print("\nDetecting therapeutic modalities from Study A gold reasoning...")
    modality_counts = Counter()
    samples_with_modality = 0

    for s in study_a:
        text = s["reasoning"] + " " + s["gold_answer"]
        mods = detect_modalities(text)
        if mods:
            samples_with_modality += 1
        for m in mods:
            modality_counts[m] += 1

    print(f"  {samples_with_modality}/{len(study_a)} samples reference at least one modality")

    # ── Build output ─────────────────────────────────────────────────────
    analysis = {
        "profile_label": "Clinical Benchmark Overview",
        "total_samples": len(all_samples),
        "study_order": active_study_order,
        "study_labels": {key: DEFAULT_STUDY_LABELS[key] for key in active_study_order},
        "radar_series": [
            {"key": "overall", "label": "Overall", "style": "overall"},
            *[
                {
                    "key": study_key,
                    "label": DEFAULT_STUDY_LABELS[study_key],
                    "style": {
                        "A": "study_a",
                        "A_bias": "study_a_bias_dotted",
                        "B": "study_b",
                        "B_multi": "study_b_multiturn",
                        "C": "study_c",
                    }[study_key],
                }
                for study_key in active_study_order
            ],
        ],
        "readme_notes": [
            "Study A Bias uses adversarial bias labels rather than gold clinical diagnoses, so it is shown under a dedicated `Bias / Adversarial` category.",
            *(
                [
                    f"Study A Bias was sourced from `{bias_meta['source_path']}` because the selected data root does not contain a local bias file."
                ]
                if bias_meta["used_fallback"]
                else []
            ),
        ],
        "per_study": {
            study_key: len(samples_by_study[study_key])
            for study_key in active_study_order
        },
        "diagnostic_categories": {
            cat: {
                "total": count,
                "pct": round(100.0 * count / len(all_samples), 2),
                "per_study": {
                    st: category_per_study[st][cat]
                    for st in active_study_order
                },
            }
            for cat, count in category_counts.most_common()
        },
        "specific_conditions": {
            cond: {
                "total": count,
                "category": BIAS_CATEGORY if condition_per_study["A_bias"].get(cond, 0) else classify_category(cond),
                "severity": "Unknown" if condition_per_study["A_bias"].get(cond, 0) else classify_severity(cond),
                "per_study": {
                    st: condition_per_study[st][cond]
                    for st in active_study_order
                },
            }
            for cond, count in condition_counts.most_common()
        },
        "severity": {
            sev: {
                "total": count,
                "pct": round(100.0 * count / len(all_samples), 2),
                "per_study": {
                    st: severity_per_study[st][sev]
                    for st in active_study_order
                },
            }
            for sev, count in severity_counts.most_common()
        },
        "therapeutic_modalities": {
            mod: {
                "mentions": count,
                "pct_of_study_a": round(100.0 * count / len(study_a), 2),
            }
            for mod, count in modality_counts.most_common()
        },
        "unique_conditions_per_study": {
            study_key: len(set(s["condition"] for s in samples_by_study[study_key]))
            for study_key in active_study_order
        },
        "total_unique_conditions": len(condition_counts),
    }

    # Save JSON
    with open(OUT / "distribution_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\nAnalysis saved to {OUT / 'distribution_analysis.json'}")

    return analysis, study_a


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_fig1_clinical_coverage(analysis):
    """Three-panel figure: diagnostic categories | top conditions | severity donut."""
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 7.4),
                             gridspec_kw={"width_ratios": [1.0, 1.15, 0.85]})
    fig.suptitle("Clinical Coverage Profile", fontsize=14, fontweight="bold", y=1.03)

    # ── Panel A: Diagnostic Categories (horizontal bar) ──────────────────
    ax = axes[0]
    cats = analysis["diagnostic_categories"]
    # Sort ascending for horizontal bar (top = highest)
    names = list(reversed(list(cats.keys())))
    counts = [cats[n]["total"] for n in names]
    pcts = [cats[n]["pct"] for n in names]

    cat_colours = [BLUE for _ in names]
    bars = ax.barh(range(len(names)), counts, color=cat_colours, edgecolor=WHITE,
                   linewidth=0.7, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Samples")
    ax.set_title("A. DSM-5 Category Coverage", loc="left", pad=10)

    # Add compact count + percentage labels
    max_count = max(counts)
    for i, (count, pct) in enumerate(zip(counts, pcts)):
        ax.text(count + max_count * 0.012, i, f"{count:,} ({pct:.1f}%)",
                ha="left", va="center", fontsize=7.3, color=TEXT_DARK)
    ax.margins(x=0.18)
    style_axis(ax, grid_axis="x")

    # ── Panel B: Top 25 Specific Conditions (horizontal bar) ─────────────
    ax = axes[1]
    conds = analysis["specific_conditions"]
    top_n = 25
    top_conds = list(conds.keys())[:top_n]
    cond_names = [normalise_condition_name(c) for c in reversed(top_conds)]
    cond_counts = [conds[c]["total"] for c in reversed(top_conds)]

    bar_colours = [GOLD for _ in reversed(top_conds)]

    bars = ax.barh(range(len(cond_names)), cond_counts, color=bar_colours,
                   edgecolor=WHITE, linewidth=0.6, height=0.75)
    ax.set_yticks(range(len(cond_names)))
    ax.set_yticklabels(cond_names, fontsize=7.2)
    ax.set_xlabel("Samples")
    ax.set_title(f"B. Top {top_n} Conditions", loc="left", pad=10)

    max_count = max(cond_counts) if cond_counts else 1
    for i, count in enumerate(cond_counts):
        ax.text(count + max_count * 0.01, i, f"{count}",
                ha="left", va="center", fontsize=6.6, color=TEXT_DARK)
    ax.margins(x=0.14)
    style_axis(ax, grid_axis="x")

    # ── Panel C: Severity Donut ──────────────────────────────────────────
    ax = axes[2]
    sev_data = analysis["severity"]
    # Order: Severe, Moderate, Mild, Critical (like OpenR1-Psy)
    sev_order = ["Critical", "Severe", "Moderate", "Mild"]
    sev_order = [s for s in sev_order if s in sev_data]  # only existing

    sev_vals = [sev_data[s]["total"] for s in sev_order]
    sev_pcts = [sev_data[s]["pct"] for s in sev_order]
    sev_cols = [SEV_COLOURS.get(s, "#999") for s in sev_order]

    wedges, _ = ax.pie(
        sev_vals, labels=None, colors=sev_cols,
        startangle=90, counterclock=False,
        wedgeprops={"width": 0.4, "edgecolor": WHITE, "linewidth": 1.4},
    )
    centre = plt.Circle((0, 0), 0.58, fc=WHITE, ec=WHITE, lw=1.0)
    ax.add_artist(centre)

    # Add labels outside
    for i, (s, pct, val) in enumerate(zip(sev_order, sev_pcts, sev_vals)):
        angle = (wedges[i].theta1 + wedges[i].theta2) / 2
        x = 1.35 * np.cos(np.radians(angle))
        y = 1.35 * np.sin(np.radians(angle))
        ha = "left" if x > 0 else "right"
        ax.text(x, y, f"{s}\n{val:,} ({pct:.1f}%)",
                ha=ha, va="center", fontsize=8, fontweight="bold", color=TEXT_DARK)

    ax.text(0, 0, f"Severity\nN={sum(sev_vals):,}", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=TEXT_DARK)
    ax.set_title("C. Severity Mix", loc="left", pad=10)
    ax.set_aspect("equal")

    plt.tight_layout(w_pad=3.4)
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig1_clinical_coverage.{ext}")
    print(f"  Saved fig1_clinical_coverage.pdf/png")
    plt.close(fig)


def plot_fig2_per_study(analysis):
    """Stacked horizontal bar: diagnostic categories broken down by study."""
    cats = analysis["diagnostic_categories"]
    cat_names = list(cats.keys())
    cat_names_rev = list(reversed(cat_names))

    fig, ax = plt.subplots(figsize=(11.0, 6.4))

    studies = analysis.get("study_order", [key for key in DEFAULT_STUDY_ORDER if key in analysis["per_study"]])
    left = np.zeros(len(cat_names_rev))

    for study in studies:
        vals = [cats[c]["per_study"].get(study, 0) for c in cat_names_rev]
        ax.barh(range(len(cat_names_rev)), vals, left=left,
                color=STUDY_COLOURS[study], edgecolor=WHITE, linewidth=0.6,
                height=0.72, label=analysis.get("study_labels", {}).get(study, f"Study {study}"))
        left += np.array(vals)

    ax.set_yticks(range(len(cat_names_rev)))
    ax.set_yticklabels(cat_names_rev, fontsize=8.4)
    ax.set_xlabel("Samples")
    ax.set_title("Diagnostic Category Split by Study", loc="left", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.03),
              frameon=False, ncol=3, title="Dataset")
    style_axis(ax, grid_axis="x")

    # Total labels
    for i, name in enumerate(cat_names_rev):
        total = cats[name]["total"]
        ax.text(total + 10, i, f"{total:,}", ha="left", va="center",
                fontsize=7, color=TEXT_DARK)
    ax.margins(x=0.08)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig2_per_study_breakdown.{ext}")
    print(f"  Saved fig2_per_study_breakdown.pdf/png")
    plt.close(fig)


def plot_fig3_modalities(analysis):
    """Horizontal bar: therapeutic modality mentions in Study A reasoning."""
    mods = analysis["therapeutic_modalities"]
    if not mods:
        print("  Skipping modality plot — no modalities detected.")
        return

    mod_names = list(reversed(list(mods.keys())))
    mod_counts = [mods[m]["mentions"] for m in mod_names]
    mod_pcts = [mods[m]["pct_of_study_a"] for m in mod_names]

    fig, ax = plt.subplots(figsize=(9.2, 5.3))

    modality_colours = plt.cm.Blues(np.linspace(0.48, 0.82, len(mod_names)))
    bars = ax.barh(range(len(mod_names)), mod_counts, color=modality_colours,
                   edgecolor=WHITE, linewidth=0.6, height=0.68)
    ax.set_yticks(range(len(mod_names)))
    ax.set_yticklabels(mod_names, fontsize=8.2)
    ax.set_xlabel("Mentions in Study A Gold Reasoning")
    ax.set_title("Therapeutic Modality Mentions in Study A Reasoning", loc="left", pad=10)

    max_count = max(mod_counts) if mod_counts else 1
    for i, (bar, count, pct) in enumerate(zip(bars, mod_counts, mod_pcts)):
        ax.text(count + max_count * 0.01, i,
                f"{count} ({pct:.1f}%)",
                ha="left", va="center", fontsize=7.2, color=TEXT_DARK)
    ax.margins(x=0.12)
    style_axis(ax, grid_axis="x")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig3_therapeutic_modalities.{ext}")
    print(f"  Saved fig3_therapeutic_modalities.pdf/png")
    plt.close(fig)


def plot_fig4_condition_detail(analysis):
    """Detailed condition breakdown showing all conditions grouped by category."""
    conds = analysis["specific_conditions"]
    cats = analysis["diagnostic_categories"]

    # Build grouped data
    cat_order = list(cats.keys())
    groups = defaultdict(list)
    for cond, info in conds.items():
        groups[info["category"]].append((cond, info["total"]))

    # Sort within each group
    for cat in groups:
        groups[cat].sort(key=lambda x: x[1], reverse=True)

    # Build flat list for plotting
    labels = []
    values = []
    colours = []
    separators = []

    cat_colour_idx = {
        cat: CATEGORY_COLOURS.get(cat, CATEGORY_COLOURS["Unclassified"])
        for cat in cat_order
    }

    pos = 0
    for cat in cat_order:
        if cat not in groups:
            continue
        separators.append((pos, cat))
        for cond, count in groups[cat]:
            labels.append(normalise_condition_name(cond))
            values.append(count)
            colours.append(cat_colour_idx[cat])
            pos += 1
        pos += 0.5  # gap between groups

    # Compute y positions with gaps
    y_positions = []
    current_y = 0
    group_idx = 0
    next_sep = separators[group_idx + 1][0] if group_idx + 1 < len(separators) else float("inf")
    for i in range(len(labels)):
        y_positions.append(current_y)
        current_y += 1
        if i + 1 == next_sep:
            current_y += 0.8  # gap
            group_idx += 1
            next_sep = separators[group_idx + 1][0] if group_idx + 1 < len(separators) else float("inf")

    y_positions.reverse()
    labels.reverse()
    values.reverse()
    colours.reverse()

    fig_height = max(11.5, len(labels) * 0.30)
    fig, ax = plt.subplots(figsize=(12.8, fig_height))

    bars = ax.barh(y_positions, values, color=colours, edgecolor=WHITE,
                   linewidth=0.35, height=0.82)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=6.7)
    ax.set_xlabel("Samples")
    ax.set_title("All Clinical Conditions Grouped by DSM-5 Category", loc="left", pad=10)

    max_val = max(values) if values else 1
    for yp, val in zip(y_positions, values):
        if val >= 25:
            ax.text(val + max_val * 0.006, yp, f"{val}",
                    ha="left", va="center", fontsize=5.8, color=TEXT_DARK)

    # Legend for categories
    handles = [mpatches.Patch(color=cat_colour_idx[cat], label=cat)
               for cat in cat_order if cat in groups]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=7.2, frameon=False, ncol=1, title="DSM-5 Category")
    ax.margins(x=0.1)
    style_axis(ax, grid_axis="x")

    plt.tight_layout(rect=[0, 0, 0.84, 1])
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"fig4_all_conditions_by_category.{ext}")
    print(f"  Saved fig4_all_conditions_by_category.pdf/png")
    plt.close(fig)


def plot_single_overview_radar(analysis):
    """Single paper-ready overview figure with a radar chart and compact summary."""
    cat_data = analysis["diagnostic_categories"]
    # Use top-7 categories for readability and parity with paper-style radar examples.
    radar_categories = list(cat_data.keys())[:7]

    active_study_order = analysis.get("study_order", [key for key in DEFAULT_STUDY_ORDER if key in analysis["per_study"]])
    study_labels = analysis.get("study_labels", {})
    series_names = ["Overall", *[study_labels.get(key, key) for key in active_study_order]]
    series_colours = {"Overall": "#EF4444"}
    series_colours.update({
        study_labels.get(key, key): STUDY_COLOURS[key]
        for key in active_study_order
    })

    values = {k: [] for k in series_names}
    for cat in radar_categories:
        total = cat_data[cat]["total"]
        values["Overall"].append(100.0 * total / analysis["total_samples"])
        for study_key in active_study_order:
            label = study_labels.get(study_key, study_key)
            study_total = analysis["per_study"][study_key]
            study_pct = 0.0 if study_total == 0 else (100.0 * cat_data[cat]["per_study"].get(study_key, 0) / study_total)
            values[label].append(study_pct)

    # Radar setup
    n = len(radar_categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(12.8, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.75], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0], polar=True)
    ax_text = fig.add_subplot(gs[0, 1])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_categories, fontsize=10)
    ax.set_ylim(0, 40)
    ax.set_yticks([10, 20, 30, 40])
    ax.set_yticklabels(["10%", "20%", "30%", "40%"], fontsize=8, color=TEXT_MID)
    ax.grid(color=GRID_LIGHT, linewidth=0.7, alpha=0.8)
    ax.set_facecolor(WHITE)

    for name in series_names:
        v = values[name] + values[name][:1]
        ax.plot(angles, v, color=series_colours[name], linewidth=2.0, label=name)
        ax.fill(angles, v, color=series_colours[name], alpha=0.08)

    # Legend below the radar, paper style
    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        frameon=True,
        facecolor=WHITE,
        edgecolor="#D1D5DB",
    )
    for text in legend.get_texts():
        text.set_fontsize(9)

    # Side summary panel
    ax_text.axis("off")
    severity = analysis["severity"]
    top_mod = next(iter(analysis["therapeutic_modalities"].items()))
    summary_lines = [
        "Clinical Coverage Summary",
        "",
        f"Total samples: {analysis['total_samples']:,}",
        "Studies: " + "  ".join(
            f"{study_labels.get(key, key)}={analysis['per_study'][key]:,}"
            for key in active_study_order
        ),
        f"DSM-5 categories: {len(analysis['diagnostic_categories'])}",
        f"Unique conditions: {analysis['total_unique_conditions']}",
        "",
        "Severity mix:",
        f"Moderate: {severity.get('Moderate', {}).get('pct', 0):.1f}%",
        f"Mild: {severity.get('Mild', {}).get('pct', 0):.1f}%",
        f"Severe: {severity.get('Severe', {}).get('pct', 0):.1f}%",
        f"Critical: {severity.get('Critical', {}).get('pct', 0):.1f}%",
        "",
        "Top therapeutic modality:",
        f"{top_mod[0]} ({top_mod[1]['pct_of_study_a']:.1f}% of Study A)",
    ]
    ax_text.text(
        0.0, 0.98, "\n".join(summary_lines),
        va="top", ha="left", fontsize=11, color=TEXT_DARK, linespacing=1.35
    )

    fig.suptitle(
        "Clinical Benchmark Overview (Radar Profile)",
        fontsize=15, fontweight="bold", y=0.98
    )
    fig.text(
        0.055, 0.03,
        "Figure: Radar profile of top diagnostic categories across Overall and the available study distributions.",
        fontsize=10, color=TEXT_DARK
    )

    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"clinical_overview_single.{ext}")
    print("  Saved clinical_overview_single.pdf/png")
    plt.close(fig)


def print_summary(analysis):
    """Print a concise summary table to stdout."""
    print("\n" + "=" * 70)
    print("CLINICAL COVERAGE SUMMARY")
    print("=" * 70)

    print(f"\nTotal samples: {analysis['total_samples']:,}")
    for st in analysis.get("study_order", list(analysis["per_study"].keys())):
        label = analysis.get("study_labels", {}).get(st, f"Study {st}")
        print(f"  {label}: {analysis['per_study'][st]:,}")

    print(f"\nTotal unique conditions: {analysis['total_unique_conditions']}")
    for st in analysis.get("study_order", list(analysis["unique_conditions_per_study"].keys())):
        label = analysis.get("study_labels", {}).get(st, f"Study {st}")
        print(f"  {label}: {analysis['unique_conditions_per_study'][st]}")

    print(f"\nDSM-5 Diagnostic Categories ({len(analysis['diagnostic_categories'])}):")
    for cat, info in analysis["diagnostic_categories"].items():
        print(f"  {info['total']:>5d} ({info['pct']:>5.1f}%)  {cat}")

    print(f"\nSeverity Distribution:")
    for sev, info in analysis["severity"].items():
        print(f"  {info['total']:>5d} ({info['pct']:>5.1f}%)  {sev}")

    print(f"\nTherapeutic Modalities (from Study A gold reasoning, {analysis['per_study']['A']} samples):")
    for mod, info in analysis["therapeutic_modalities"].items():
        print(f"  {info['mentions']:>5d} ({info['pct_of_study_a']:>5.1f}%)  {mod}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Analyse clinical distribution for a frozen split root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional frozen split root to analyse.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Optional output directory for analysis artefacts.")
    parser.add_argument(
        "--bias-root",
        type=Path,
        default=None,
        help="Optional fallback root that provides `adversarial_bias/biased_vignettes.json`.",
    )
    parser.add_argument(
        "--bias-sample-size",
        type=int,
        default=None,
        help="Optional deterministic sample size for fallback Study A Bias loading.",
    )
    parser.add_argument(
        "--bias-sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic fallback Study A Bias sampling.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Only write distribution_analysis.json and the console summary.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_paths(
        data_root=args.data_root,
        out_dir=args.out_dir,
        bias_root=args.bias_root,
        bias_sample_size=args.bias_sample_size,
        bias_sample_seed=args.bias_sample_seed,
    )
    analysis, study_a = run_analysis()
    print_summary(analysis)

    if not args.skip_figures:
        print("\nGenerating figures...")
        plot_single_overview_radar(analysis)

    print(f"\nDone. All outputs in {OUT}/")
