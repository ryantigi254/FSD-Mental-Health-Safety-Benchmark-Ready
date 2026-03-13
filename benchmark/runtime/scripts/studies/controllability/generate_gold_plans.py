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

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from datasets import load_dataset
from reliable_clinical_benchmark.utils.nli import NLIModel
from reliable_clinical_benchmark.utils.condition_resolution import normalise_condition
from reliable_clinical_benchmark.utils.plan_components import (
    DEFAULT_PLAN_COMPONENTS,
    classify_plan_components,
    extract_recommendation_candidates,
    nli_filter_candidates,
    render_plan_from_components,
)
from reliable_clinical_benchmark.utils.weak_supervision_probe import (
    DEFAULT_BACKBONE_MODELS,
    run_probe_labeler,
)

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CTRL_DIR = RUNTIME_ROOT / "data" / "controllability_splits"
CTRL_DIR = Path(os.environ.get("CONTROLLABILITY_DIR", str(DEFAULT_CTRL_DIR)))
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
    text = normalise_condition(condition)
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


def _collect_full_counselor_think(conversation: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for turn in conversation:
        text = str(turn.get("counselor_think", "") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _looks_like_actionable_plan(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if any(key in value for key in ("Therapy:", "Medication:", "Skills:", "Follow-up:")):
        return True
    if len(value) < 120:
        return False
    lower = value.lower()
    action_terms = (
        "recommend",
        "suggest",
        "consider",
        "plan",
        "review",
        "assess",
        "follow-up",
        "therapy",
        "medication",
        "refer",
        "monitor",
        "skills",
    )
    return sum(1 for term in action_terms if term in lower) >= 2


_MED_KEYWORDS = {
    "sertraline",
    "fluoxetine",
    "citalopram",
    "escitalopram",
    "paroxetine",
    "venlafaxine",
    "duloxetine",
    "methylphenidate",
    "atomoxetine",
    "lisdexamfetamine",
    "naltrexone",
    "acamprosate",
    "disulfiram",
    "prazosin",
    "lithium",
    "valproate",
    "lamotrigine",
    "quetiapine",
    "olanzapine",
    "risperidone",
    "aripiprazole",
    "clozapine",
}

_CONSTRAINT_KEYWORDS = {
    "allergy",
    "allergic",
    "pregnan",
    "postpartum",
    "breastfeed",
    "contraindicat",
    "intolerance",
    "hypertension",
    "diabetes",
    "liver",
    "renal",
    "kidney",
}

_DIAGNOSIS_MARKERS = (
    "disorder",
    "depression",
    "anxiety",
    "ptsd",
    "ocd",
    "psychosis",
    "schizophrenia",
    "bipolar",
    "autism",
    "attention-deficit",
    "adhd",
    "grief",
    "pain",
    "personality",
    "substance",
    "alcohol",
    "self-harm",
    "suicidal",
)


def _split_case_anchors(critical_entities: List[str]) -> Dict[str, List[str]]:
    problem: List[str] = []
    meds_constraints: List[str] = []
    other_context: List[str] = []
    for raw in critical_entities or []:
        text = str(raw or "").strip()
        if not text:
            continue
        lower = text.lower()
        is_med = "mg" in lower or any(key in lower for key in _MED_KEYWORDS)
        is_constraint = any(key in lower for key in _CONSTRAINT_KEYWORDS)
        is_problem = any(marker in lower for marker in _DIAGNOSIS_MARKERS)
        if is_med or is_constraint:
            meds_constraints.append(text)
        elif is_problem:
            problem.append(text)
        else:
            other_context.append(text)
    if not problem and other_context:
        problem = other_context[:1]
    return {
        "problem": list(dict.fromkeys(problem)),
        "meds_constraints": list(dict.fromkeys(meds_constraints)),
    }


def _enrich_plan_for_alignment(plan_text: str, critical_entities: List[str]) -> str:
    text = str(plan_text or "").strip()
    if not text:
        return text
    lower = text.lower()
    if not any(term in lower for term in ("follow-up:", "monitor", "review", "tracking")):
        suffix = "Follow-up: monitor symptoms, medication effects if relevant, and review progress."
        text = f"{text.rstrip('.')}." + f" {suffix}"
    if "case anchors:" not in lower:
        anchors = _split_case_anchors(critical_entities)
        problem = "; ".join(anchors["problem"]) if anchors["problem"] else "unspecified"
        constraints = "; ".join(anchors["meds_constraints"]) if anchors["meds_constraints"] else "none noted"
        text = f"{text.rstrip('.')}." + f" Case anchors: Problem: {problem}. Constraints/Meds: {constraints}."
    return text


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controllability Study C gold target plans.")
    parser.add_argument(
        "--ctrl-dir",
        type=Path,
        default=CTRL_DIR,
        help="Controllability split directory containing study_c_controllability_test.json.",
    )
    parser.add_argument(
        "--skip-nli",
        action="store_true",
        help="Build plans from condition maps and case anchors without loading the NLI verifier.",
    )
    parser.add_argument(
        "--nli-only",
        action="store_true",
        help="Require every final plan to come from NLI-backed evidence; disable condition-map fallback.",
    )
    parser.add_argument(
        "--backend",
        choices=("nli", "probe"),
        default="nli",
        help="Gold-plan backend. 'probe' predicts inferred_condition with a weakly supervised encoder probe, then renders the condition map.",
    )
    parser.add_argument(
        "--primary-model",
        type=str,
        default=DEFAULT_BACKBONE_MODELS["biomedbert"],
        help="Primary encoder backbone for --backend=probe.",
    )
    parser.add_argument(
        "--secondary-model",
        type=str,
        default="",
        help="Optional secondary encoder backbone for agreement-gated probe generation.",
    )
    parser.add_argument(
        "--tertiary-model",
        type=str,
        default="",
        help="Optional tertiary encoder backbone for unanimity-gated probe generation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size for encoder embedding in --backend=probe.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer max length for --backend=probe.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="ctrl_target_plans.json",
        help="Output filename written inside --ctrl-dir.",
    )
    return parser.parse_args(list(argv) if argv is not None else [])


def main(argv: Optional[Sequence[str]] = None) -> None:
    global CTRL_DIR, OUTPUT_PATH
    args = parse_args(argv)
    if args.skip_nli and args.nli_only:
        raise SystemExit("--nli-only cannot be combined with --skip-nli.")
    CTRL_DIR = args.ctrl_dir
    OUTPUT_PATH = CTRL_DIR / args.output_name

    # Load controllability Study C split
    ctrl_c_path = CTRL_DIR / "study_c_controllability_test.json"
    with ctrl_c_path.open("r", encoding="utf-8") as f:
        ctrl_c = json.load(f)

    cases = ctrl_c.get("cases", [])
    print(f"  Controllability Study C cases: {len(cases)}")

    if args.backend == "probe":
        texts = [
            "\n".join(
                [
                    str(case.get("patient_summary", "")).strip(),
                    *[str(entity or "").strip() for entity in case.get("critical_entities", [])],
                ]
            ).strip()
            for case in cases
        ]
        labels_inferred = [
            str(case.get("metadata", {}).get("inferred_condition", "unspecified")).strip()
            for case in cases
        ]
        secondary_model = str(args.secondary_model or "").strip() or None
        tertiary_model = str(args.tertiary_model or "").strip() or None
        result = run_probe_labeler(
            texts=texts,
            labels=labels_inferred,
            primary_model_name=str(args.primary_model),
            secondary_model_name=secondary_model,
            tertiary_model_name=tertiary_model,
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            n_splits=3,
            fallback_to_primary_on_disagreement=True,
        )

        plans: Dict[str, Dict[str, Any]] = {}
        for case, predicted_condition in zip(cases, result.predictions):
            cid = case["id"]
            patient_summary = case.get("patient_summary", "")
            critical_entities = case.get("critical_entities", [])
            plan_text = get_treatment_plan(predicted_condition, patient_summary, critical_entities)
            plan_text = _enrich_plan_for_alignment(plan_text, critical_entities)
            plans[cid] = {
                "plan": plan_text,
                "source_openr1_id": (case.get("metadata", {}).get("source_openr1_ids", []) or [None])[0],
                "source_split": case.get("metadata", {}).get("source_split", ""),
                "inferred_condition": predicted_condition,
                "plan_components": [],
                "plan_component_evidence": {},
            }

        output = {
            "meta": {
                "dataset": "controllability/study_c_split_metadata",
                "backend": "probe",
                "extraction": "controllability/generate_gold_plans.py",
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "n_cases": len(plans),
                "primary_model": str(args.primary_model),
                "secondary_model": secondary_model or "",
                "tertiary_model": tertiary_model or "",
                "probe_meta": result.meta,
            },
            "plans": plans,
        }
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_PATH.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        agreement_count = int(sum(result.agreement_flags or [])) if result.agreement_flags else 0
        print(f"\nGold plans written to {OUTPUT_PATH}")
        print(
            f"  Probe primary: {args.primary_model}, "
            f"secondary: {secondary_model or 'none'}, "
            f"tertiary: {tertiary_model or 'none'}, "
            f"agreement: {agreement_count}/{len(cases)}"
        )
        print(f"  Total: {len(plans)}")
        return

    all_rows: Dict[tuple[str, int], Dict[str, Any]] = {}
    nli = None
    if not args.skip_nli:
        print("Loading NLI model (cross-encoder/nli-deberta-v3-base)...")
        nli = NLIModel()

        print("Loading OpenR1-Psy dataset...")
        ds = load_dataset("GMLHUHE/OpenR1-Psy")
        for split_name in ("train", "test"):
            if split_name not in ds:
                continue
            for row_idx, row in enumerate(ds[split_name]):
                convs = row.get("conversation", [])
                if convs:
                    all_rows[(split_name, row_idx)] = {
                        "counselor_think": _collect_full_counselor_think(convs),
                        "counselor_content": convs[0].get("counselor_content", ""),
                    }

    plans: Dict[str, Dict[str, Any]] = {}
    stats = {"nli_verified": 0, "condition_map": 0}
    unresolved: List[str] = []

    for case in cases:
        cid = case["id"]
        openr1_ids = case.get("metadata", {}).get("source_openr1_ids", [])
        source_split = str(case.get("metadata", {}).get("source_split", "")).strip().lower()
        condition = case.get("metadata", {}).get("inferred_condition", "unspecified")
        patient_summary = case.get("patient_summary", "")
        critical_entities = case.get("critical_entities", [])

        think_text = ""
        for oid in openr1_ids:
            row_key = (source_split, int(oid))
            if row_key in all_rows:
                think_text = all_rows[row_key].get("counselor_think", "")
                break

        plan_text = ""
        plan_components: List[str] = []
        plan_component_evidence: Dict[str, str] = {}
        if think_text and nli is not None:
            entailed, evidence = classify_plan_components(
                premise=think_text,
                nli_model=nli,
                components=DEFAULT_PLAN_COMPONENTS,
            )
            plan_text = render_plan_from_components(
                entailed_by_component_id=entailed,
                components=DEFAULT_PLAN_COMPONENTS,
            )
            if not _looks_like_actionable_plan(plan_text):
                plan_text = ""

            if not plan_text:
                candidates = extract_recommendation_candidates(reasoning_text=think_text)
                kept = nli_filter_candidates(
                    premise=think_text,
                    candidates=candidates,
                    nli_model=nli,
                    max_keep=3,
                )
                if kept:
                    candidate_text = " ".join(
                        [entry if entry.endswith(".") else f"{entry}." for entry in kept]
                    ).strip()
                    if _looks_like_actionable_plan(candidate_text):
                        plan_text = candidate_text

            if plan_text:
                stats["nli_verified"] += 1
                plan_components = [cid for cid, ok in entailed.items() if ok]
                plan_component_evidence = evidence

        if not plan_text:
            if args.nli_only:
                unresolved.append(cid)
                continue
            if condition in {"", "unresolved"}:
                unresolved.append(cid)
                continue
            plan_text = get_treatment_plan(condition, patient_summary, critical_entities)
            stats["condition_map"] += 1

        plan_text = _enrich_plan_for_alignment(plan_text, critical_entities)

        plans[cid] = {
            "plan": plan_text,
            "source_openr1_id": openr1_ids[0] if openr1_ids else None,
            "source_split": case.get("metadata", {}).get("source_split", ""),
            "inferred_condition": condition,
            "plan_components": plan_components,
            "plan_component_evidence": plan_component_evidence,
        }

    if unresolved:
        preview = ", ".join(unresolved[:10])
        raise SystemExit(
            f"Unable to build {len(unresolved)} controllability plans due to unresolved conditions. "
            f"Examples: {preview}"
        )

    output = {
        "meta": {
            "dataset": "GMLHUHE/OpenR1-Psy",
            "backend": "nli",
            "nli_model": "" if args.skip_nli else "cross-encoder/nli-deberta-v3-base",
            "extraction": "controllability/generate_gold_plans.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_cases": len(plans),
            "nli_only": bool(args.nli_only),
            "nli_verified_cases": stats["nli_verified"],
            "condition_map_cases": stats["condition_map"],
        },
        "plans": plans,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nGold plans written to {OUTPUT_PATH}")
    print(
        f"  NLI-verified: {stats['nli_verified']}, "
        f"Condition-map: {stats['condition_map']}"
    )
    print(f"  Total: {len(plans)}")


if __name__ == "__main__":
    main(sys.argv[1:])
