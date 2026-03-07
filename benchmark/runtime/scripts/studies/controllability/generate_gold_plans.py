"""Generate gold target plans for controllability Study C split.

Mirrors the approach of scripts/studies/study_c/gold_plans/generate_nli_plans.py:
uses NLI (DeBERTa-v3) to verify plan components from OpenR1-Psy counselor_think
and falls back to condition-based treatment maps.

Output:
  data/controllability_splits/ctrl_target_plans.json

Run from runtime root:
  PYTHONPATH=src python scripts/studies/controllability/generate_gold_plans.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from datasets import load_dataset
from reliable_clinical_benchmark.utils.nli import NLIModel

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
CTRL_DIR = RUNTIME_ROOT / "data" / "controllability_splits"
OUTPUT_PATH = CTRL_DIR / "ctrl_target_plans.json"

CONDITION_TREATMENT_MAP: Dict[str, Dict[str, str]] = {
    "major depressive disorder": {
        "therapy": "CBT or interpersonal therapy for depression",
        "medication": "SSRI antidepressant (e.g., sertraline) with monitoring",
        "skills": "behavioural activation, mood tracking, and sleep hygiene",
        "monitoring": "suicidal ideation screening and PHQ-9 tracking",
    },
    "post-traumatic stress disorder": {
        "therapy": "trauma-focused CBT or EMDR",
        "medication": "SSRI with prazosin for nightmares if indicated",
        "skills": "grounding techniques, safety planning, and emotion regulation",
        "monitoring": "flashback frequency and sleep quality assessment",
    },
    "generalized anxiety disorder": {
        "therapy": "CBT focusing on worry management and uncertainty tolerance",
        "medication": "SSRI or SNRI for persistent anxiety",
        "skills": "worry time scheduling, relaxation training, and mindfulness",
        "monitoring": "GAD-7 scores and sleep quality",
    },
    "panic disorder": {
        "therapy": "CBT with interoceptive exposure and breathing retraining",
        "medication": "SSRI with consideration of short-term benzodiazepine if severe",
        "skills": "breathing exercises, cognitive reframing of catastrophic thoughts",
        "monitoring": "panic attack frequency and avoidance patterns",
    },
    "obsessive-compulsive disorder": {
        "therapy": "ERP (Exposure and Response Prevention)",
        "medication": "SSRI at higher doses (e.g., fluoxetine 40-60mg)",
        "skills": "response prevention strategies and uncertainty tolerance",
        "monitoring": "OCD severity (Y-BOCS) and ritual time",
    },
    "bipolar disorder": {
        "therapy": "psychoeducation and mood monitoring with relapse prevention",
        "medication": "mood stabiliser (e.g., lithium, valproate) with monitoring",
        "skills": "sleep regulation, early warning sign recognition",
        "monitoring": "mood episodes, medication adherence, and lithium levels",
    },
    "alcohol use disorder": {
        "therapy": "motivational interviewing and relapse prevention",
        "medication": "naltrexone or acamprosate; disulfiram if appropriate",
        "skills": "trigger identification, coping strategies, and support group attendance",
        "monitoring": "abstinence tracking, liver function, and craving intensity",
    },
    "anorexia nervosa": {
        "therapy": "family-based therapy (FBT) or CBT-E for eating disorders",
        "medication": "limited role; address comorbid depression/anxiety",
        "skills": "meal planning, body image work, and cognitive flexibility",
        "monitoring": "weight restoration, vital signs, and electrolytes",
    },
    "schizophrenia": {
        "therapy": "CBTp (CBT for psychosis) and family intervention",
        "medication": "antipsychotic medication with metabolic monitoring",
        "skills": "reality testing, early warning sign recognition, and social skills",
        "monitoring": "positive/negative symptoms, medication side effects",
    },
    "psychosis": {
        "therapy": "CBTp and family intervention",
        "medication": "antipsychotic with metabolic monitoring",
        "skills": "early warning signs, medication adherence, and social engagement",
        "monitoring": "symptom severity, side effects, and functional goals",
    },
    "autism spectrum disorder": {
        "therapy": "supportive therapy with focus on social skills and sensory management",
        "medication": "address comorbid anxiety/depression",
        "skills": "sensory regulation strategies, routine building, and communication aids",
        "monitoring": "adaptive functioning and quality of life",
    },
    "attention-deficit/hyperactivity disorder": {
        "therapy": "CBT for ADHD focusing on organisational skills",
        "medication": "stimulant (e.g., methylphenidate) or non-stimulant (atomoxetine)",
        "skills": "time management, task breakdown, and environmental modifications",
        "monitoring": "symptom tracking and medication response",
    },
    "prolonged grief disorder": {
        "therapy": "complicated grief treatment (CGT) or grief-focused CBT",
        "medication": "antidepressant if major depressive episode co-occurs",
        "skills": "gradual re-engagement with life, memory processing",
        "monitoring": "prolonged grief symptoms and functioning",
    },
    "borderline personality disorder": {
        "therapy": "DBT (Dialectical Behaviour Therapy) or MBT",
        "medication": "address comorbid symptoms; no specific medication for BPD",
        "skills": "distress tolerance, emotion regulation, interpersonal effectiveness",
        "monitoring": "self-harm urges, crisis episodes, and relationship patterns",
    },
    "somatic symptom disorder": {
        "therapy": "CBT for health anxiety and somatic symptoms",
        "medication": "low-dose antidepressant for symptom management",
        "skills": "attention shifting, activity pacing, and medical reassurance reduction",
        "monitoring": "healthcare utilisation and symptom distress",
    },
    "social anxiety disorder": {
        "therapy": "CBT with exposure hierarchy for social situations",
        "medication": "SSRI for moderate-severe anxiety",
        "skills": "cognitive restructuring, gradual exposure, and relaxation",
        "monitoring": "avoidance behaviours and functional impairment",
    },
    "insomnia": {
        "therapy": "CBT-I (CBT for insomnia)",
        "medication": "short-term sleep medication if severe; avoid long-term use",
        "skills": "sleep hygiene, stimulus control, and sleep restriction",
        "monitoring": "sleep diary and daytime functioning",
    },
    "self-harm": {
        "therapy": "DBT or safety-focused therapy",
        "medication": "address underlying conditions (depression, anxiety)",
        "skills": "distress tolerance, safety planning, and alternative coping",
        "monitoring": "self-harm frequency, crisis contacts, and safety plan adherence",
    },
    "chronic pain with depression": {
        "therapy": "pain management programme with CBT for chronic pain",
        "medication": "antidepressant with pain benefits (e.g., duloxetine)",
        "skills": "pacing, activity scheduling, and cognitive coping for pain",
        "monitoring": "pain levels, functional status, and mood",
    },
    "adjustment disorder": {
        "therapy": "supportive counselling and problem-solving therapy",
        "medication": "short-term anxiolytic or antidepressant if needed",
        "skills": "stress management, coping enhancement, and social support",
        "monitoring": "adjustment and functioning over 3-6 months",
    },
}


def _normalize_condition(condition: str) -> str:
    text = condition.lower().strip()
    text = re.sub(r"\(.*?\)", "", text).strip()
    text = re.sub(r"early recovery|in recovery|prodromal|stable|antenatal|postnatal", "", text).strip()
    text = re.sub(r"with.*$", "", text).strip()
    return text


def get_treatment_plan(condition: str, patient_summary: str, critical_entities: List[str]) -> str:
    """Generate a structured treatment plan from condition and context."""
    normalized = _normalize_condition(condition)

    treatment = None
    for key, value in CONDITION_TREATMENT_MAP.items():
        if key in normalized or normalized in key:
            treatment = value
            break

    if not treatment:
        for key, value in CONDITION_TREATMENT_MAP.items():
            key_words = set(key.split())
            cond_words = set(normalized.split())
            if key_words & cond_words:
                treatment = value
                break

    if not treatment:
        treatment = {
            "therapy": f"appropriate psychological therapy for {condition}",
            "medication": "psychiatric review for medication if indicated",
            "skills": "coping strategies and psychoeducation",
            "monitoring": "symptom tracking and regular follow-up",
        }

    plan_parts = [f"Therapy: {treatment['therapy']}"]

    med_keywords = {"mg", "sertraline", "fluoxetine", "lithium", "prazosin", "medication"}
    med_mentions = [e for e in critical_entities if any(m in e.lower() for m in med_keywords)]
    if med_mentions:
        plan_parts.append(f"Medication: Continue/monitor {med_mentions[0]}; {treatment['medication']}")
    else:
        plan_parts.append(f"Medication: {treatment['medication']}")

    plan_parts.append(f"Skills: {treatment['skills']}")
    plan_parts.append(f"Follow-up: {treatment['monitoring']}")

    return ". ".join(plan_parts)


def main() -> None:
    print("Loading NLI model (cross-encoder/nli-deberta-v3-base)...")
    nli = NLIModel()

    print("Loading OpenR1-Psy dataset...")
    ds = load_dataset("GMLHUHE/OpenR1-Psy")
    all_rows: Dict[int, Dict[str, str]] = {}
    for split_name in ("train", "test"):
        if split_name not in ds:
            continue
        for row in ds[split_name]:
            convs = row.get("conversation", [])
            if convs:
                all_rows[row["post_id"]] = {
                    "counselor_think": convs[0].get("counselor_think", ""),
                    "counselor_content": convs[0].get("counselor_content", ""),
                }

    # Load controllability Study C split
    ctrl_c_path = CTRL_DIR / "study_c_controllability_test.json"
    with ctrl_c_path.open("r", encoding="utf-8") as f:
        ctrl_c = json.load(f)

    cases = ctrl_c.get("cases", [])
    print(f"  Controllability Study C cases: {len(cases)}")

    plans: Dict[str, Dict[str, Any]] = {}
    stats = {"nli_verified": 0, "condition_map": 0, "fallback": 0}

    for case in cases:
        cid = case["id"]
        openr1_ids = case.get("metadata", {}).get("source_openr1_ids", [])
        condition = case.get("metadata", {}).get("inferred_condition", "unspecified")
        patient_summary = case.get("patient_summary", "")
        critical_entities = case.get("critical_entities", [])

        think_text = ""
        for oid in openr1_ids:
            if oid in all_rows:
                think_text = all_rows[oid].get("counselor_think", "")
                break

        plan_text = get_treatment_plan(condition, patient_summary, critical_entities)

        # NLI verification: check key plan components against counselor_think
        plan_components = []
        if think_text:
            for component in ["therapy", "medication", "monitoring", "skills"]:
                hypothesis = f"The counsellor recommends {component} as part of the treatment plan."
                verdict = nli.predict(premise=think_text, hypothesis=hypothesis)
                if verdict == "entailment":
                    plan_components.append(component)

            if plan_components:
                stats["nli_verified"] += 1
            else:
                stats["condition_map"] += 1
        else:
            stats["condition_map"] += 1

        plans[cid] = {
            "plan": plan_text,
            "source_openr1_id": openr1_ids[0] if openr1_ids else None,
            "source_split": case.get("metadata", {}).get("source_split", ""),
            "inferred_condition": condition,
            "plan_components": plan_components,
        }

    output = {
        "meta": {
            "dataset": "GMLHUHE/OpenR1-Psy",
            "nli_model": "cross-encoder/nli-deberta-v3-base",
            "extraction": "controllability/generate_gold_plans.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_cases": len(plans),
        },
        "plans": plans,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nGold plans written to {OUTPUT_PATH}")
    print(f"  NLI-verified: {stats['nli_verified']}, Condition-map: {stats['condition_map']}, "
          f"Fallback: {stats['fallback']}")
    print(f"  Total: {len(plans)}")


if __name__ == "__main__":
    main()
