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

import argparse
import gc
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
from reliable_clinical_benchmark.utils.condition_resolution import (
    normalise_condition,
    resolve_case_condition,
)

RUNTIME_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CTRL_DIR = RUNTIME_ROOT / "data" / "controllability_splits"
CTRL_DIR = Path(os.environ.get("CONTROLLABILITY_DIR", str(DEFAULT_CTRL_DIR)))
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

DISPLAY_LABELS = {
    normalise_condition(label): label for label in DIAGNOSIS_CANDIDATES
}

NLI_BATCH_SIZE = 4
NLI_DIRECT_THRESHOLD = 0.5
NLI_CONFIRM_THRESHOLD = 0.35


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

    def predict_many_scores(
        self,
        premise: str | Sequence[str],
        hypotheses: Sequence[str],
    ) -> List[tuple[str, float]]:
        import torch

        if isinstance(premise, str):
            premises = [premise] * len(hypotheses)
        else:
            premises = [str(item or "") for item in premise]
            if len(premises) != len(hypotheses):
                raise ValueError("Premise and hypothesis batches must have the same length.")

        inputs = self.tokenizer(
            premises,
            [str(hypothesis or "") for hypothesis in hypotheses],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        outputs: List[tuple[str, float]] = []
        for row in probs:
            pred_id = int(torch.argmax(row).item())
            score = float(row[pred_id].item())
            raw_label = str(self.model.config.id2label.get(pred_id, "")).lower()
            if "entail" in raw_label:
                verdict = "entailment"
            elif "contradict" in raw_label:
                verdict = "contradiction"
            else:
                verdict = "neutral"
            outputs.append((verdict, score))
        return outputs

    def predict_many_entailment_scores(
        self,
        premise: str | Sequence[str],
        hypotheses: Sequence[str],
    ) -> List[float]:
        import torch

        if isinstance(premise, str):
            premises = [premise] * len(hypotheses)
        else:
            premises = [str(item or "") for item in premise]
            if len(premises) != len(hypotheses):
                raise ValueError("Premise and hypothesis batches must have the same length.")

        inputs = self.tokenizer(
            premises,
            [str(hypothesis or "") for hypothesis in hypotheses],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        entailment_scores: List[float] = []
        for row in probs:
            labels = {
                str(self.model.config.id2label.get(idx, "")).lower(): float(value.item())
                for idx, value in enumerate(row)
            }
            entailment_score = 0.0
            for label_name, score in labels.items():
                if "entail" in label_name:
                    entailment_score = score
                    break
            entailment_scores.append(entailment_score)
        return entailment_scores


def extract_label_heuristic(think_text: str) -> Optional[str]:
    """Extract diagnosis from counselor_think using regex patterns."""
    text = (think_text or "").lower()
    for pattern, label in DIAGNOSIS_PATTERNS:
        if re.search(pattern, text):
            return label
    return None


def _display_label(label: str) -> str:
    norm = normalise_condition(label)
    return DISPLAY_LABELS.get(norm, norm.replace("/", "/").title())


def extract_label_nli(
    nli: ScoringNLIModel,
    think_text: str,
    patient_text: str,
    *,
    candidate_labels: Optional[Sequence[str]] = None,
    threshold: float = NLI_DIRECT_THRESHOLD,
) -> Optional[str]:
    """Extract diagnosis using NLI entailment scoring."""
    premise = f"{patient_text}\n\n{think_text}"
    candidates = list(candidate_labels or DIAGNOSIS_CANDIDATES)
    hypotheses = [
        f"The clinical reasoning indicates the patient has {candidate}."
        for candidate in candidates
    ]
    best_label = None
    best_score = 0.0

    for candidate, (verdict, score) in zip(candidates, nli.predict_many_scores(premise, hypotheses)):
        if verdict == "entailment" and score > best_score:
            best_label = candidate
            best_score = score

    return best_label if best_score >= threshold else None


def extract_labels_nli_batch(
    nli: ScoringNLIModel,
    premises: Sequence[str],
    *,
    candidate_labels: Optional[Sequence[str]] = None,
    threshold: float = NLI_DIRECT_THRESHOLD,
) -> List[Optional[str]]:
    candidates = list(candidate_labels or DIAGNOSIS_CANDIDATES)
    if not hasattr(nli, "predict_many_scores"):
        return [
            extract_label_nli(
                nli,
                premise,
                "",
                candidate_labels=candidates,
                threshold=threshold,
            )
            for premise in premises
        ]

    hypotheses: List[str] = []
    repeated_premises: List[str] = []
    for premise in premises:
        repeated_premises.extend([premise] * len(candidates))
        hypotheses.extend(
            f"The clinical reasoning indicates the patient has {candidate}." for candidate in candidates
        )

    batched_scores = nli.predict_many_scores(repeated_premises, hypotheses)

    labels: List[Optional[str]] = []
    offset = 0
    for _premise in premises:
        best_label = None
        best_score = 0.0
        window = batched_scores[offset:offset + len(candidates)]
        for candidate, (verdict, score) in zip(candidates, window):
            if verdict == "entailment" and score > best_score:
                best_label = candidate
                best_score = score
        labels.append(best_label if best_score >= threshold else None)
        offset += len(candidates)
    return labels


def build_candidate_label_pool(samples: Sequence[Dict[str, Any]]) -> List[str]:
    """Build a stable candidate pool that covers the split's inferred labels."""
    ordered: List[str] = []
    seen = set()

    def _add(label: str) -> None:
        normalised = normalise_condition(label)
        if not normalised or normalised in {"", "unresolved", "unspecified"}:
            return
        display = _display_label(label)
        if display not in seen:
            seen.add(display)
            ordered.append(display)

    for label in DIAGNOSIS_CANDIDATES:
        _add(label)
    for sample in samples:
        inferred = str(sample.get("metadata", {}).get("inferred_condition", "")).strip()
        if inferred:
            _add(inferred)

    return ordered


def extract_best_label_nli_batch(
    nli: ScoringNLIModel,
    premises: Sequence[str],
    *,
    candidate_labels: Sequence[str],
) -> List[Optional[str]]:
    candidates = list(candidate_labels)
    if not candidates:
        return [None for _ in premises]

    if not hasattr(nli, "predict_many_entailment_scores"):
        return extract_labels_nli_batch(
            nli,
            premises,
            candidate_labels=candidates,
            threshold=0.0,
        )

    hypotheses: List[str] = []
    repeated_premises: List[str] = []
    for premise in premises:
        repeated_premises.extend([premise] * len(candidates))
        hypotheses.extend(
            f"The clinical reasoning indicates the patient has {candidate}." for candidate in candidates
        )

    entailment_scores = nli.predict_many_entailment_scores(repeated_premises, hypotheses)
    labels: List[Optional[str]] = []
    offset = 0
    for _premise in premises:
        window = entailment_scores[offset:offset + len(candidates)]
        if not window:
            labels.append(None)
        else:
            best_index = max(range(len(window)), key=window.__getitem__)
            labels.append(candidates[best_index])
        offset += len(candidates)
    return labels


def build_premise(patient_text: str, think_text: str, counselor_content: str) -> str:
    parts = [str(patient_text or "").strip(), str(think_text or "").strip(), str(counselor_content or "").strip()]
    return "\n\n".join(part for part in parts if part)


def confirm_label_with_nli(
    nli: ScoringNLIModel,
    *,
    premise: str,
    candidate_label: str,
    threshold: float = NLI_CONFIRM_THRESHOLD,
) -> tuple[bool, float]:
    hypothesis = f"The clinical reasoning indicates the patient has {candidate_label}."
    verdict, score = nli.predict_with_score(premise, hypothesis)
    return verdict == "entailment" and score >= threshold, score


def confirm_labels_with_nli_batch(
    nli: ScoringNLIModel,
    *,
    premises: Sequence[str],
    candidate_labels: Sequence[str],
    threshold: float = NLI_CONFIRM_THRESHOLD,
) -> List[bool]:
    if not hasattr(nli, "predict_many_scores"):
        return [
            confirm_label_with_nli(
                nli,
                premise=premise,
                candidate_label=candidate_label,
                threshold=threshold,
            )[0]
            for premise, candidate_label in zip(premises, candidate_labels)
        ]

    hypotheses = [
        f"The clinical reasoning indicates the patient has {candidate_label}."
        for candidate_label in candidate_labels
    ]
    outputs = nli.predict_many_scores(list(premises), hypotheses)
    return [
        verdict == "entailment" and score >= threshold
        for verdict, score in outputs
    ]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controllability Study A gold diagnosis labels.")
    parser.add_argument(
        "--ctrl-dir",
        type=Path,
        default=CTRL_DIR,
        help="Controllability split directory containing study_a_controllability_test.json.",
    )
    parser.add_argument(
        "--skip-nli",
        action="store_true",
        help="Use inferred_condition metadata directly instead of running the NLI-backed extractor.",
    )
    parser.add_argument(
        "--nli-only",
        action="store_true",
        help="Require every final label to be NLI-backed; no metadata-only fallback is allowed.",
    )
    parser.add_argument(
        "--direct-threshold",
        type=float,
        default=NLI_DIRECT_THRESHOLD,
        help="Entailment threshold for direct NLI label extraction.",
    )
    parser.add_argument(
        "--confirm-threshold",
        type=float,
        default=NLI_CONFIRM_THRESHOLD,
        help="Entailment threshold for confirming seeded candidate labels with NLI.",
    )
    return parser.parse_args(list(argv) if argv is not None else [])


def main(argv: Optional[Sequence[str]] = None) -> None:
    global CTRL_DIR, OUTPUT_PATH
    args = parse_args(argv)
    if args.skip_nli and args.nli_only:
        raise SystemExit("--nli-only cannot be combined with --skip-nli.")
    CTRL_DIR = args.ctrl_dir
    OUTPUT_PATH = CTRL_DIR / "ctrl_gold_diagnosis_labels.json"

    nli = None
    all_rows: Dict[tuple[str, int], Dict[str, str]] = {}
    if not args.skip_nli:
        print("Loading NLI model (cross-encoder/nli-deberta-v3-base)...")
        nli = ScoringNLIModel()

        print("Loading OpenR1-Psy dataset...")
        ds = load_dataset("GMLHUHE/OpenR1-Psy")
        for split_name in ("train", "test"):
            if split_name not in ds:
                continue
            for row_idx, row in enumerate(ds[split_name]):
                convs = row.get("conversation", [])
                if convs:
                    all_rows[(split_name, row_idx)] = {
                        "patient": convs[0].get("patient", ""),
                        "counselor_think": convs[0].get("counselor_think", ""),
                        "counselor_content": convs[0].get("counselor_content", ""),
                    }

        print(f"  OpenR1-Psy rows indexed: {len(all_rows)}")

    # Load controllability Study A split
    ctrl_a_path = CTRL_DIR / "study_a_controllability_test.json"
    with ctrl_a_path.open("r", encoding="utf-8") as f:
        ctrl_a = json.load(f)

    samples = ctrl_a.get("samples", [])
    print(f"  Controllability Study A samples: {len(samples)}")
    candidate_labels = build_candidate_label_pool(samples)

    labels: Dict[str, str] = {}
    stats = {
        "nli_direct": 0,
        "nli_confirmed_heuristic": 0,
        "nli_confirmed_resolved": 0,
        "nli_confirmed_inferred": 0,
        "nli_ranked_rescue": 0,
    }
    unresolved: List[str] = []
    prepared_samples: List[Dict[str, Any]] = []

    for sample in samples:
        sid = sample["id"]
        openr1_ids = sample.get("metadata", {}).get("source_openr1_ids", [])
        source_split = str(sample.get("metadata", {}).get("source_split", "")).strip().lower()
        inferred = sample.get("metadata", {}).get("inferred_condition", "")

        think_text = ""
        patient_text = sample.get("prompt", "")
        counselor_content = ""
        for oid in openr1_ids:
            row_key = (source_split, int(oid))
            if row_key in all_rows:
                think_text = all_rows[row_key].get("counselor_think", "")
                patient_text = all_rows[row_key].get("patient", "") or patient_text
                counselor_content = all_rows[row_key].get("counselor_content", "")
                break

        prepared_samples.append(
            {
                "id": sid,
                "inferred": inferred,
                "patient_text": patient_text,
                "think_text": think_text,
                "counselor_content": counselor_content,
            }
        )

    provisional_candidates: List[Dict[str, Any]] = []

    for row in prepared_samples:
        sid = row["id"]
        inferred = row["inferred"]
        patient_text = row["patient_text"]
        think_text = row["think_text"]
        counselor_content = row["counselor_content"]
        premise = build_premise(patient_text, think_text, counselor_content)
        provisional_label = None
        provisional_source = ""

        if think_text:
            provisional_label = extract_label_heuristic(think_text)
            provisional_source = "heuristic"

        if not provisional_label and not args.skip_nli:
            resolved, source = resolve_case_condition(
                {
                    "patient": patient_text,
                    "counselor_think": think_text,
                    "counselor_content": counselor_content,
                },
                nli_model=nli,
            )
            if resolved:
                provisional_label = _display_label(resolved)
                provisional_source = "resolved"

        if not provisional_label and inferred and inferred not in {"", "unresolved"}:
            provisional_label = _display_label(inferred)
            provisional_source = "inferred"

        provisional_candidates.append(
            {
                "id": sid,
                "premise": premise,
                "candidate_label": provisional_label,
                "candidate_source": provisional_source,
            }
        )

    confirmed_labels_by_id: Dict[str, tuple[str, str]] = {}
    if nli is not None:
        confirmable = [row for row in provisional_candidates if row["candidate_label"]]
        total_batches = (len(confirmable) + NLI_BATCH_SIZE - 1) // NLI_BATCH_SIZE if confirmable else 0
        for batch_idx, start in enumerate(range(0, len(confirmable), NLI_BATCH_SIZE), start=1):
            chunk = confirmable[start:start + NLI_BATCH_SIZE]
            print(f"NLI confirm batch {batch_idx}/{total_batches} ({len(chunk)} samples)")
            confirmations = confirm_labels_with_nli_batch(
                nli,
                premises=[row["premise"] for row in chunk],
                candidate_labels=[str(row["candidate_label"]) for row in chunk],
                threshold=float(args.confirm_threshold),
            )
            for row, confirmed in zip(chunk, confirmations):
                if confirmed:
                    confirmed_labels_by_id[row["id"]] = (
                        str(row["candidate_label"]),
                        str(row["candidate_source"]),
                    )
            gc.collect()

    backstop_labels_by_id: Dict[str, str] = {}
    if nli is not None:
        backstop_rows = [
            row
            for row in provisional_candidates
            if row["id"] not in confirmed_labels_by_id
        ]
        total_batches = (len(backstop_rows) + NLI_BATCH_SIZE - 1) // NLI_BATCH_SIZE if backstop_rows else 0
        for batch_idx, start in enumerate(range(0, len(backstop_rows), NLI_BATCH_SIZE), start=1):
            chunk = backstop_rows[start:start + NLI_BATCH_SIZE]
            print(f"NLI backstop batch {batch_idx}/{total_batches} ({len(chunk)} samples)")
            premises = [row["premise"] for row in chunk]
            for row, label in zip(
                chunk,
                extract_labels_nli_batch(
                    nli,
                    premises,
                    candidate_labels=candidate_labels,
                    threshold=float(args.direct_threshold),
                ),
            ):
                if label:
                    backstop_labels_by_id[row["id"]] = label
            gc.collect()

    ranked_rescue_labels_by_id: Dict[str, str] = {}
    if nli is not None and args.nli_only:
        rescue_rows = [
            row
            for row in provisional_candidates
            if row["id"] not in confirmed_labels_by_id and row["id"] not in backstop_labels_by_id
        ]
        total_batches = (len(rescue_rows) + NLI_BATCH_SIZE - 1) // NLI_BATCH_SIZE if rescue_rows else 0
        for batch_idx, start in enumerate(range(0, len(rescue_rows), NLI_BATCH_SIZE), start=1):
            chunk = rescue_rows[start:start + NLI_BATCH_SIZE]
            print(f"NLI ranked rescue batch {batch_idx}/{total_batches} ({len(chunk)} samples)")
            premises = [row["premise"] for row in chunk]
            for row, label in zip(
                chunk,
                extract_best_label_nli_batch(
                    nli,
                    premises,
                    candidate_labels=candidate_labels,
                ),
            ):
                if label:
                    ranked_rescue_labels_by_id[row["id"]] = label
            gc.collect()

    for row in prepared_samples:
        sid = row["id"]
        label = None

        # Strategy 1: metadata fast path
        if args.skip_nli:
            inferred = row["inferred"]
            if inferred and inferred not in {"", "unresolved"}:
                label = _display_label(inferred)

        if not label and sid in confirmed_labels_by_id:
            label, source = confirmed_labels_by_id[sid]
            if source == "heuristic":
                stats["nli_confirmed_heuristic"] += 1
            elif source == "resolved":
                stats["nli_confirmed_resolved"] += 1
            elif source == "inferred":
                stats["nli_confirmed_inferred"] += 1

        if not label and sid in backstop_labels_by_id:
            label = backstop_labels_by_id[sid]
            stats["nli_direct"] += 1

        if not label and sid in ranked_rescue_labels_by_id:
            label = ranked_rescue_labels_by_id[sid]
            stats["nli_ranked_rescue"] += 1

        if not label:
            unresolved.append(sid)
            continue

        labels[sid] = label

    if unresolved:
        missing_preview = ", ".join(unresolved[:10])
        raise SystemExit(
            f"Unable to resolve {len(unresolved)} controllability gold labels. "
            f"Examples: {missing_preview}"
        )

    output = {
        "meta": {
            "dataset": "GMLHUHE/OpenR1-Psy",
            "nli_model": "" if args.skip_nli else "cross-encoder/nli-deberta-v3-base",
            "extraction": "controllability/generate_gold_labels.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(labels),
            "nli_only": bool(args.nli_only),
            "direct_threshold": float(args.direct_threshold),
            "confirm_threshold": float(args.confirm_threshold),
            "nli_direct_labels": stats["nli_direct"],
            "nli_ranked_rescue_labels": stats["nli_ranked_rescue"],
            "nli_confirmed_heuristic_labels": stats["nli_confirmed_heuristic"],
            "nli_confirmed_resolved_labels": stats["nli_confirmed_resolved"],
            "nli_confirmed_inferred_labels": stats["nli_confirmed_inferred"],
        },
        "labels": labels,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nGold labels written to {OUTPUT_PATH}")
    print(
        f"  NLI direct: {stats['nli_direct']}, "
        f"Ranked rescue: {stats['nli_ranked_rescue']}, "
        f"Heuristic confirmed by NLI: {stats['nli_confirmed_heuristic']}, "
        f"Resolved confirmed by NLI: {stats['nli_confirmed_resolved']}, "
        f"Inferred confirmed by NLI: {stats['nli_confirmed_inferred']}"
    )
    print(f"  Total: {len(labels)}")


if __name__ == "__main__":
    main(sys.argv[1:])
