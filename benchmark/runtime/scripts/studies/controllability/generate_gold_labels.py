"""Generate gold diagnosis labels for controllability Study A split.

Mirrors the approach of scripts/studies/study_a/gold_labels/populate_from_openr1.py:
uses NLI (DeBERTa-v3) to extract diagnosis labels from the OpenR1-Psy counselor_think
field for each controllability sample.

Output:
  data/controllability_splits/ctrl_gold_diagnosis_labels.json

Run from runtime root:
  PYTHONPATH=src python scripts/studies/controllability/generate_gold_labels.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from datasets import load_dataset
from reliable_clinical_benchmark.utils.nli import NLIModel

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
CTRL_DIR = RUNTIME_ROOT / "data" / "controllability_splits"
OUTPUT_PATH = CTRL_DIR / "ctrl_gold_diagnosis_labels.json"

DIAGNOSIS_CANDIDATES = [
    "Adjustment Disorder",
    "Major Depressive Disorder",
    "Generalized Anxiety Disorder",
    "Post-Traumatic Stress Disorder",
    "Social Anxiety Disorder",
    "Panic Disorder",
    "Obsessive-Compulsive Disorder",
    "Bipolar Disorder",
    "Schizophrenia",
    "Borderline Personality Disorder",
    "Alcohol Use Disorder",
    "Substance Use Disorder",
    "Anorexia Nervosa",
    "Eating Disorder",
    "Autism Spectrum Disorder",
    "Attention-Deficit/Hyperactivity Disorder",
    "Insomnia",
    "Somatic Symptom Disorder",
    "Prolonged Grief Disorder",
    "Self-Harm",
    "Psychosis",
    "Body Dysmorphic Disorder",
    "Agoraphobia",
    "Specific Phobia",
]

DIAGNOSIS_PATTERNS = [
    (r"\bmajor depressi", "Major Depressive Disorder"),
    (r"\bgeneralized anxiety|generalised anxiety", "Generalized Anxiety Disorder"),
    (r"\bpost-traumatic|ptsd\b", "Post-Traumatic Stress Disorder"),
    (r"\bsocial anxiety", "Social Anxiety Disorder"),
    (r"\bpanic disorder|panic attack", "Panic Disorder"),
    (r"\bobsessive.compulsive|ocd\b", "Obsessive-Compulsive Disorder"),
    (r"\bbipolar\b", "Bipolar Disorder"),
    (r"\bschizophreni", "Schizophrenia"),
    (r"\bborderline personality|bpd\b", "Borderline Personality Disorder"),
    (r"\balcohol use disorder|alcoholism", "Alcohol Use Disorder"),
    (r"\bsubstance use", "Substance Use Disorder"),
    (r"\banorexia\b", "Anorexia Nervosa"),
    (r"\beating disorder", "Eating Disorder"),
    (r"\bautis", "Autism Spectrum Disorder"),
    (r"\battention.deficit|adhd\b", "Attention-Deficit/Hyperactivity Disorder"),
    (r"\binsomnia\b", "Insomnia"),
    (r"\bsomatic\b", "Somatic Symptom Disorder"),
    (r"\bgrief\b", "Prolonged Grief Disorder"),
    (r"\bself.harm\b", "Self-Harm"),
    (r"\bpsychos[ie]s\b|hallucin", "Psychosis"),
    (r"\badjustment disorder", "Adjustment Disorder"),
]


class ScoringNLIModel(NLIModel):
    """NLI model with confidence scoring for diagnosis ranking."""

    def predict_with_score(self, premise: str, hypothesis: str) -> tuple:
        import torch

        inputs = self.tokenizer(
            str(premise or ""), str(hypothesis or ""),
            truncation=True, max_length=self.max_length, return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred_id = int(torch.argmax(probs).item())
            score = float(probs[pred_id].item())

        raw_label = str(self.model.config.id2label.get(pred_id, "")).lower()
        if "entail" in raw_label:
            return "entailment", score
        if "contradict" in raw_label:
            return "contradiction", score
        return "neutral", score


def extract_label_heuristic(think_text: str) -> Optional[str]:
    """Extract diagnosis from counselor_think using regex patterns."""
    text = (think_text or "").lower()
    for pattern, label in DIAGNOSIS_PATTERNS:
        if re.search(pattern, text):
            return label
    return None


def extract_label_nli(
    nli: ScoringNLIModel,
    think_text: str,
    patient_text: str,
) -> Optional[str]:
    """Extract diagnosis using NLI entailment scoring."""
    premise = f"{patient_text}\n\n{think_text}"
    best_label = None
    best_score = 0.0

    for candidate in DIAGNOSIS_CANDIDATES:
        hypothesis = f"The clinical reasoning indicates the patient has {candidate}."
        verdict, score = nli.predict_with_score(premise, hypothesis)
        if verdict == "entailment" and score > best_score:
            best_label = candidate
            best_score = score

    return best_label if best_score >= 0.5 else None


def main() -> None:
    print("Loading NLI model (cross-encoder/nli-deberta-v3-base)...")
    nli = ScoringNLIModel()

    print("Loading OpenR1-Psy dataset...")
    ds = load_dataset("GMLHUHE/OpenR1-Psy")
    all_rows = {}
    for split_name in ("train", "test"):
        if split_name not in ds:
            continue
        for row in ds[split_name]:
            pid = row["post_id"]
            convs = row.get("conversation", [])
            if convs:
                all_rows[pid] = {
                    "patient": convs[0].get("patient", ""),
                    "counselor_think": convs[0].get("counselor_think", ""),
                }

    print(f"  OpenR1-Psy rows indexed: {len(all_rows)}")

    # Load controllability Study A split
    ctrl_a_path = CTRL_DIR / "study_a_controllability_test.json"
    with ctrl_a_path.open("r", encoding="utf-8") as f:
        ctrl_a = json.load(f)

    samples = ctrl_a.get("samples", [])
    print(f"  Controllability Study A samples: {len(samples)}")

    labels: Dict[str, str] = {}
    stats = {"nli": 0, "heuristic": 0, "inferred": 0, "fallback": 0}

    for sample in samples:
        sid = sample["id"]
        openr1_ids = sample.get("metadata", {}).get("source_openr1_ids", [])
        inferred = sample.get("metadata", {}).get("inferred_condition", "")

        think_text = ""
        patient_text = sample.get("prompt", "")
        for oid in openr1_ids:
            if oid in all_rows:
                think_text = all_rows[oid].get("counselor_think", "")
                patient_text = all_rows[oid].get("patient", "") or patient_text
                break

        label = None

        # Strategy 1: NLI from counselor_think
        if think_text:
            label = extract_label_nli(nli, think_text, patient_text)
            if label:
                stats["nli"] += 1

        # Strategy 2: Heuristic from counselor_think
        if not label and think_text:
            label = extract_label_heuristic(think_text)
            if label:
                stats["heuristic"] += 1

        # Strategy 3: Use inferred condition from build script
        if not label and inferred and inferred != "unspecified":
            label = inferred.title()
            stats["inferred"] += 1

        # Strategy 4: Fallback
        if not label:
            label = "Adjustment Disorder"
            stats["fallback"] += 1

        labels[sid] = label

    output = {"labels": labels}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nGold labels written to {OUTPUT_PATH}")
    print(f"  NLI: {stats['nli']}, Heuristic: {stats['heuristic']}, "
          f"Inferred: {stats['inferred']}, Fallback: {stats['fallback']}")
    print(f"  Total: {len(labels)}")


if __name__ == "__main__":
    main()
