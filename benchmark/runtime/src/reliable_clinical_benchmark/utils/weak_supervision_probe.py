from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier


DEFAULT_BACKBONE_MODELS: Dict[str, str] = {
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "biolinkbert": "michiyasunaga/BioLinkBERT-base",
    "debertav3": "microsoft/deberta-v3-base",
    "bluebert": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
}


@dataclass(frozen=True)
class ProbeResult:
    predictions: List[str]
    confidences: List[float]
    agreement_flags: List[bool] | None
    meta: Dict[str, object]


def _set_thread_guards() -> None:
    os.environ.setdefault("KMP_USE_SHM", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


class FrozenEncoder:
    def __init__(self, model_name: str, *, max_length: int = 256):
        _set_thread_guards()
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.max_length = int(max_length)

    def encode(self, texts: Sequence[str], *, batch_size: int = 12) -> np.ndarray:
        rows: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = [str(text or "") for text in texts[start : start + batch_size]]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            rows.append(pooled.cpu().numpy())
        return np.concatenate(rows, axis=0)


def _oof_predictions(
    embeddings: np.ndarray,
    labels: Sequence[str],
    *,
    n_splits: int,
) -> tuple[List[str], List[float], Dict[str, object]]:
    counts = Counter(str(label or "") for label in labels)
    frequent_labels = {label for label, count in counts.items() if count >= n_splits}
    frequent_idx = [idx for idx, label in enumerate(labels) if label in frequent_labels]
    rare_idx = [idx for idx, label in enumerate(labels) if label not in frequent_labels]

    predictions = [str(label or "") for label in labels]
    confidences = [1.0 if idx in rare_idx else 0.0 for idx in range(len(labels))]

    if not frequent_idx:
        return predictions, confidences, {
            "n_splits": int(n_splits),
            "frequent_labels": [],
            "rare_labels": {label: count for label, count in counts.items()},
            "rare_label_fallbacks": len(labels),
        }

    subset_embeddings = embeddings[frequent_idx]
    subset_labels = [str(labels[idx]) for idx in frequent_idx]
    ordered_labels = sorted(set(subset_labels))
    label_to_idx = {label: idx for idx, label in enumerate(ordered_labels)}
    y = np.array([label_to_idx[label] for label in subset_labels], dtype=np.int64)

    probabilities = np.zeros((len(frequent_idx), len(ordered_labels)), dtype=np.float64)
    predictions_subset = np.zeros(len(frequent_idx), dtype=np.int64)

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in folds.split(subset_embeddings, y):
        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                solver="liblinear",
            )
        )
        clf.fit(subset_embeddings[train_idx], y[train_idx])
        fold_prob = clf.predict_proba(subset_embeddings[test_idx])
        probabilities[test_idx] = fold_prob
        predictions_subset[test_idx] = np.argmax(fold_prob, axis=1)

    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    for local_idx, global_idx in enumerate(frequent_idx):
        predictions[global_idx] = idx_to_label[int(predictions_subset[local_idx])]
        confidences[global_idx] = float(np.max(probabilities[local_idx]))

    return predictions, confidences, {
        "n_splits": int(n_splits),
        "frequent_labels": ordered_labels,
        "rare_labels": {
            label: count for label, count in counts.items() if label not in frequent_labels
        },
        "rare_label_fallbacks": len(rare_idx),
    }


def run_probe_labeler(
    *,
    texts: Sequence[str],
    labels: Sequence[str],
    primary_model_name: str,
    secondary_model_name: str | None = None,
    tertiary_model_name: str | None = None,
    batch_size: int = 12,
    max_length: int = 256,
    n_splits: int = 3,
    fallback_to_primary_on_disagreement: bool = True,
) -> ProbeResult:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    primary_encoder = FrozenEncoder(primary_model_name, max_length=max_length)
    primary_embeddings = primary_encoder.encode(texts, batch_size=batch_size)
    primary_predictions, primary_confidences, primary_meta = _oof_predictions(
        primary_embeddings,
        labels,
        n_splits=n_splits,
    )

    agreement_flags: List[bool] | None = None
    final_predictions = list(primary_predictions)
    if secondary_model_name:
        secondary_encoder = FrozenEncoder(secondary_model_name, max_length=max_length)
        secondary_embeddings = secondary_encoder.encode(texts, batch_size=batch_size)
        secondary_predictions, _, secondary_meta = _oof_predictions(
            secondary_embeddings,
            labels,
            n_splits=n_splits,
        )
        extra_predictions = [secondary_predictions]
        primary_meta["secondary_model_name"] = secondary_model_name
        primary_meta["secondary_probe"] = secondary_meta
        if tertiary_model_name:
            tertiary_encoder = FrozenEncoder(tertiary_model_name, max_length=max_length)
            tertiary_embeddings = tertiary_encoder.encode(texts, batch_size=batch_size)
            tertiary_predictions, _, tertiary_meta = _oof_predictions(
                tertiary_embeddings,
                labels,
                n_splits=n_splits,
            )
            extra_predictions.append(tertiary_predictions)
            primary_meta["tertiary_model_name"] = tertiary_model_name
            primary_meta["tertiary_probe"] = tertiary_meta
        agreement_flags = []
        for idx, primary in enumerate(primary_predictions):
            peers = [preds[idx] for preds in extra_predictions]
            agreement_flags.append(all(primary == peer for peer in peers))
        if not fallback_to_primary_on_disagreement:
            final_predictions = [
                primary if agreed else ""
                for primary, agreed in zip(primary_predictions, agreement_flags)
            ]
        primary_meta["agreement_count"] = int(sum(agreement_flags))
        primary_meta["agreement_rate"] = float(sum(agreement_flags) / len(agreement_flags))

    primary_meta["primary_model_name"] = primary_model_name
    primary_meta["fallback_to_primary_on_disagreement"] = bool(fallback_to_primary_on_disagreement)
    primary_meta["weak_supervision_source"] = "split_metadata_inferred_condition"

    return ProbeResult(
        predictions=final_predictions,
        confidences=primary_confidences,
        agreement_flags=agreement_flags,
        meta=primary_meta,
    )
