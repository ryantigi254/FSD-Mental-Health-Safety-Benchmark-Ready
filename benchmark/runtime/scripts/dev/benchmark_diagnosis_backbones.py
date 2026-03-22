"""Benchmark diagnosis backbones on the controllability Study A split.

The benchmark uses frozen encoder embeddings plus a linear one-vs-rest probe so
we can compare backbones quickly on the local 20-label taxonomy without a full
fine-tuning cycle.

Run from the runtime root:

  KMP_USE_SHM=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=src python scripts/dev/benchmark_diagnosis_backbones.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_PATH = RUNTIME_ROOT / "data" / "controllability" / "misc" / "controllability_splits" / "study_a_controllability_test.json"
DEFAULT_OUTPUT_PATH = Path("/tmp/diagnosis_backbone_benchmark.json")

DEFAULT_MODELS: Dict[str, str] = {
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "biolinkbert": "michiyasunaga/BioLinkBERT-base",
    "debertav3": "microsoft/deberta-v3-base",
    "bluebert": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
}

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
PAIR_LABELS = [
    ("self-harm", "suicidal crisis"),
    ("psychosis", "bipolar disorder"),
    ("generalized anxiety disorder", "panic disorder"),
    ("major depressive disorder", "prolonged grief disorder"),
]


def _set_thread_guards() -> None:
    os.environ.setdefault("KMP_USE_SHM", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def load_split(split_path: Path, min_count: int) -> tuple[list[str], list[str], dict[str, int]]:
    payload = json.loads(split_path.read_text())
    samples = payload["samples"] if isinstance(payload, dict) else payload
    labels_all = [str(sample.get("metadata", {}).get("inferred_condition", "unknown")) for sample in samples]
    counts = Counter(labels_all)
    keep = {label for label, count in counts.items() if count >= min_count}
    texts = [str(sample.get("prompt", "")) for sample in samples if sample.get("metadata", {}).get("inferred_condition") in keep]
    labels = [str(sample.get("metadata", {}).get("inferred_condition", "unknown")) for sample in samples if sample.get("metadata", {}).get("inferred_condition") in keep]
    rare = {label: count for label, count in counts.items() if count < min_count}
    return texts, labels, rare


class Encoder:
    def __init__(self, model_name: str, max_length: int = 256):
        import torch

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.max_length = max_length

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            rows.append(pooled.cpu().numpy())
        return np.concatenate(rows, axis=0)


class EntailmentVerifier:
    def __init__(self, model_name: str = NLI_MODEL, max_length: int = 256):
        import torch

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.max_length = max_length

    def entailment_scores(self, premise: str, hypotheses: Sequence[str]) -> list[float]:
        inputs = self.tokenizer(
            [premise] * len(hypotheses),
            list(hypotheses),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1)
        entailment_scores: list[float] = []
        for row in probs:
            score = 0.0
            for idx, value in enumerate(row):
                label_name = str(self.model.config.id2label.get(idx, "")).lower()
                if "entail" in label_name:
                    score = float(value.item())
                    break
            entailment_scores.append(score)
        return entailment_scores


@dataclass
class FoldOutput:
    probabilities: np.ndarray
    predictions: np.ndarray


def probe_cross_val(embeddings: np.ndarray, y: np.ndarray, folds: StratifiedKFold) -> FoldOutput:
    n = len(y)
    n_classes = len(np.unique(y))
    probabilities = np.zeros((n, n_classes), dtype=np.float64)
    predictions = np.zeros(n, dtype=np.int64)
    for train_idx, test_idx in folds.split(embeddings, y):
        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                solver="liblinear",
            )
        )
        clf.fit(embeddings[train_idx], y[train_idx])
        fold_prob = clf.predict_proba(embeddings[test_idx])
        fold_pred = np.argmax(fold_prob, axis=1)
        probabilities[test_idx] = fold_prob
        predictions[test_idx] = fold_pred
    return FoldOutput(probabilities=probabilities, predictions=predictions)


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_one = np.eye(probabilities.shape[1])[y_true]
    return float(np.mean(np.sum((probabilities - y_one) ** 2, axis=1)))


def ece_score(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def abstain_stats(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    keep_mask = confidence >= threshold
    coverage = float(keep_mask.mean())
    if not np.any(keep_mask):
        return {"coverage": coverage, "accuracy": None, "macro_f1": None, "n": 0}
    return {
        "coverage": coverage,
        "accuracy": float((prediction[keep_mask] == y_true[keep_mask]).mean()),
        "macro_f1": float(f1_score(y_true[keep_mask], prediction[keep_mask], average="macro")),
        "n": int(keep_mask.sum()),
    }


def summarise_model(y_true: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "top2_recall": float(np.mean([truth in np.argsort(prob)[-2:] for truth, prob in zip(y_true, probabilities)])),
        "brier": multiclass_brier(y_true, probabilities),
        "ece": ece_score(y_true, probabilities),
        "abstain@0.15": abstain_stats(y_true, probabilities, 0.15),
        "abstain@0.18": abstain_stats(y_true, probabilities, 0.18),
        "abstain@0.20": abstain_stats(y_true, probabilities, 0.20),
        "confidence_quantiles": {
            "p10": float(np.quantile(probabilities.max(axis=1), 0.10)),
            "p50": float(np.quantile(probabilities.max(axis=1), 0.50)),
            "p90": float(np.quantile(probabilities.max(axis=1), 0.90)),
        },
    }


def pair_confusions(y_true: np.ndarray, predictions: np.ndarray, classes: Sequence[str]) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    for label_a, label_b in PAIR_LABELS:
        key = f"{label_a}__{label_b}"
        if label_a not in class_to_idx or label_b not in class_to_idx:
            outputs[key] = {"status": "not_evaluable_in_subset"}
            continue
        a_idx = class_to_idx[label_a]
        b_idx = class_to_idx[label_b]
        mask = np.isin(y_true, [a_idx, b_idx])
        subset_true = y_true[mask]
        subset_pred = predictions[mask]
        matrix = confusion_matrix(subset_true, subset_pred, labels=[a_idx, b_idx])
        outputs[key] = {
            "support_a": int((subset_true == a_idx).sum()),
            "support_b": int((subset_true == b_idx).sum()),
            "a_to_a": int(matrix[0, 0]),
            "a_to_b": int(matrix[0, 1]),
            "b_to_a": int(matrix[1, 0]),
            "b_to_b": int(matrix[1, 1]),
        }
    return outputs


def agreement_metrics(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    agree_mask = pred_a == pred_b
    coverage = float(agree_mask.mean())
    if not np.any(agree_mask):
        return {"coverage": coverage, "accuracy_on_covered": None, "macro_f1_on_covered": None, "n_covered": 0}
    return {
        "coverage": coverage,
        "accuracy_on_covered": float((pred_a[agree_mask] == y_true[agree_mask]).mean()),
        "macro_f1_on_covered": float(f1_score(y_true[agree_mask], pred_a[agree_mask], average="macro")),
        "n_covered": int(agree_mask.sum()),
    }


def average_probabilities(*arrays: np.ndarray) -> np.ndarray:
    stacked = np.stack(arrays, axis=0)
    return np.mean(stacked, axis=0)


def verifier_metrics(
    verifier: EntailmentVerifier,
    texts: Sequence[str],
    probabilities: np.ndarray,
    y_true: np.ndarray,
    classes: Sequence[str],
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    predictions = np.full(len(texts), -1, dtype=np.int64)
    confidences = np.zeros(len(texts), dtype=np.float64)
    for idx, (text, prob) in enumerate(zip(texts, probabilities)):
        top_two = np.argsort(prob)[-2:][::-1]
        hypotheses = [f"The most likely diagnosis is {classes[class_idx]}." for class_idx in top_two]
        scores = verifier.entailment_scores(text, hypotheses)
        best_local = int(np.argmax(scores))
        best_score = float(scores[best_local])
        other_score = float(scores[1 - best_local])
        if best_score >= score_threshold and (best_score - other_score) >= margin_threshold:
            predictions[idx] = int(top_two[best_local])
            confidences[idx] = best_score
    covered = predictions >= 0
    coverage = float(covered.mean())
    if not np.any(covered):
        return {"coverage": coverage, "accuracy_on_covered": None, "macro_f1_on_covered": None, "n_covered": 0}
    return {
        "coverage": coverage,
        "accuracy_on_covered": float((predictions[covered] == y_true[covered]).mean()),
        "macro_f1_on_covered": float(f1_score(y_true[covered], predictions[covered], average="macro")),
        "n_covered": int(covered.sum()),
        "entailment_score_quantiles": {
            "p10": float(np.quantile(confidences[covered], 0.10)),
            "p50": float(np.quantile(confidences[covered], 0.50)),
            "p90": float(np.quantile(confidences[covered], 0.90)),
        },
    }

def verifier_threshold_sweep(
    verifier: EntailmentVerifier,
    texts: Sequence[str],
    probabilities: np.ndarray,
    y_true: np.ndarray,
    classes: Sequence[str],
    score_thresholds: Sequence[float],
    margin_thresholds: Sequence[float],
) -> dict[str, Any]:
    predictions = np.full(len(texts), -1, dtype=np.int64)
    best_scores = np.zeros(len(texts), dtype=np.float64)
    gaps = np.zeros(len(texts), dtype=np.float64)
    for idx, (text, prob) in enumerate(zip(texts, probabilities)):
        top_two = np.argsort(prob)[-2:][::-1]
        hypotheses = [f"The most likely diagnosis is {classes[class_idx]}." for class_idx in top_two]
        scores = verifier.entailment_scores(text, hypotheses)
        best_local = int(np.argmax(scores))
        other_local = 1 - best_local
        predictions[idx] = int(top_two[best_local])
        best_scores[idx] = float(scores[best_local])
        gaps[idx] = float(scores[best_local] - scores[other_local])

    sweep: dict[str, Any] = {}
    for score_threshold in score_thresholds:
        for margin_threshold in margin_thresholds:
            mask = (best_scores >= score_threshold) & (gaps >= margin_threshold)
            key = f"score>={score_threshold:.2f}|gap>={margin_threshold:.2f}"
            coverage = float(mask.mean())
            if not np.any(mask):
                sweep[key] = {"coverage": coverage, "accuracy_on_covered": None, "macro_f1_on_covered": None, "n_covered": 0}
                continue
            sweep[key] = {
                "coverage": coverage,
                "accuracy_on_covered": float((predictions[mask] == y_true[mask]).mean()),
                "macro_f1_on_covered": float(f1_score(y_true[mask], predictions[mask], average="macro")),
                "n_covered": int(mask.sum()),
            }
    return sweep


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    return parser


def main() -> int:
    _set_thread_guards()
    parser = build_argument_parser()
    args = parser.parse_args()

    texts, labels, rare = load_split(args.split_path, args.min_count)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    classes = list(label_encoder.classes_)
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    encoded: dict[str, np.ndarray] = {}
    model_outputs: dict[str, FoldOutput] = {}
    model_metrics: dict[str, Any] = {}

    for alias, model_name in DEFAULT_MODELS.items():
        print(f"[encode] {alias} <- {model_name}", flush=True)
        encoder = Encoder(model_name=model_name, max_length=args.max_length)
        embeddings = encoder.encode(texts, batch_size=args.batch_size)
        encoded[alias] = embeddings
        outputs = probe_cross_val(embeddings, y, folds)
        model_outputs[alias] = outputs
        model_metrics[alias] = summarise_model(y, outputs.probabilities, outputs.predictions)

    ensemble_metrics: dict[str, Any] = {}
    avg_pair_probabilities: dict[str, np.ndarray] = {}

    for left, right in itertools.combinations(DEFAULT_MODELS.keys(), 2):
        avg_name = f"avg_{left}_{right}"
        avg_prob = average_probabilities(
            model_outputs[left].probabilities,
            model_outputs[right].probabilities,
        )
        avg_pred = np.argmax(avg_prob, axis=1)
        avg_pair_probabilities[avg_name] = avg_prob
        ensemble_metrics[avg_name] = summarise_model(y, avg_prob, avg_pred)

        agree_name = f"agree_{left}_{right}"
        ensemble_metrics[agree_name] = agreement_metrics(
            y,
            model_outputs[left].predictions,
            model_outputs[right].predictions,
        )

    trio_specs = [
        ("bioclinicalbert", "biomedbert", "biolinkbert"),
        ("bioclinicalbert", "biomedbert", "debertav3"),
        ("biomedbert", "biolinkbert", "debertav3"),
    ]
    for members in trio_specs:
        trio_name = "avg_" + "_".join(members)
        trio_prob = average_probabilities(*(model_outputs[member].probabilities for member in members))
        trio_pred = np.argmax(trio_prob, axis=1)
        ensemble_metrics[trio_name] = summarise_model(y, trio_prob, trio_pred)

    print("[verifier] loading NLI verifier", flush=True)
    verifier = EntailmentVerifier(max_length=args.max_length)
    best_single = max(model_metrics.items(), key=lambda item: item[1]["macro_f1"])[0]
    best_avg_pair_name = max(avg_pair_probabilities, key=lambda name: ensemble_metrics[name]["macro_f1"])
    ensemble_metrics[f"verifier_{best_single}_top2"] = verifier_metrics(
        verifier,
        texts,
        model_outputs[best_single].probabilities,
        y,
        classes,
        score_threshold=args.score_threshold,
        margin_threshold=args.margin_threshold,
    )
    ensemble_metrics[f"verifier_{best_avg_pair_name}_top2"] = verifier_metrics(
        verifier,
        texts,
        avg_pair_probabilities[best_avg_pair_name],
        y,
        classes,
        score_threshold=args.score_threshold,
        margin_threshold=args.margin_threshold,
    )
    ensemble_metrics[f"verifier_sweep_{best_single}_top2"] = verifier_threshold_sweep(
        verifier,
        texts,
        model_outputs[best_single].probabilities,
        y,
        classes,
        score_thresholds=[0.30, 0.35, 0.40, 0.45],
        margin_thresholds=[0.00, 0.02, 0.05],
    )
    ensemble_metrics[f"verifier_sweep_{best_avg_pair_name}_top2"] = verifier_threshold_sweep(
        verifier,
        texts,
        avg_pair_probabilities[best_avg_pair_name],
        y,
        classes,
        score_thresholds=[0.30, 0.35, 0.40, 0.45],
        margin_thresholds=[0.00, 0.02, 0.05],
    )
    per_label = classification_report(
        y,
        model_outputs[best_single].predictions,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    payload = {
        "split_path": str(args.split_path),
        "n_samples": len(texts),
        "n_labels": len(classes),
        "excluded_rare_labels": rare,
        "models": DEFAULT_MODELS,
        "single_models": model_metrics,
        "ensembles": ensemble_metrics,
        "best_single_model": best_single,
        "best_avg_pair": best_avg_pair_name,
        "per_label_recall_best_single": {
            label: {
                "recall": float(per_label[label]["recall"]),
                "support": int(per_label[label]["support"]),
            }
            for label in classes
        },
        "pair_confusions_best_single": pair_confusions(y, model_outputs[best_single].predictions, classes),
        "verifier_thresholds": {
            "score_threshold": args.score_threshold,
            "margin_threshold": args.margin_threshold,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[done] results written to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
