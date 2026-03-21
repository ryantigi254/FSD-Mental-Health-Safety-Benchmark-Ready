"""FAISS-based retrieval index for finding compatible source rows.

Uses MiniLM-L6-v2 embeddings (same model as drift.py / extraction.py)
with optional entity-overlap pre-filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..utils.ner import MedicalNER

logger = logging.getLogger(__name__)

# Singleton embedder — mirrors the pattern in metrics/extraction.py
_EMBEDDER = None


def _get_embedder():
    """Lazy-load a MiniLM sentence-transformer (singleton)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not available; retrieval disabled")
            return None
    return _EMBEDDER


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievalHit:
    """A single retrieval result."""
    row_id: str
    score: float
    entity_overlap: int
    condition_overlap: int
    row_data: Optional[Dict] = None


@dataclass
class RetrievalIndex:
    """Dense retrieval index backed by FAISS IndexFlatIP."""

    embeddings: np.ndarray                       # (N, D) L2-normalised
    row_ids: List[str]                           # parallel identifiers
    entity_sets: List[Set[str]]                  # per-row clinical entities
    texts: List[str]                             # per-row source text
    _faiss_index: object = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        rows: List[Dict],
        ner: "MedicalNER",
        *,
        id_key: str = "id",
        text_key: str = "prompt",
    ) -> "RetrievalIndex":
        """Build a retrieval index from a list of data rows.

        Parameters
        ----------
        rows : list of dict
            Each dict must contain at least *id_key* and *text_key*.
        ner : MedicalNER
            Pre-loaded NER model for entity extraction.
        id_key / text_key : str
            Field names for the row identifier and source text.
        """
        embedder = _get_embedder()
        if embedder is None:
            raise RuntimeError("sentence-transformers required for retrieval index")

        texts = [str(r.get(text_key, "")) for r in rows]
        row_ids = [str(r.get(id_key, f"row_{i}")) for i, r in enumerate(rows)]
        entity_sets = [ner.extract_clinical_entities(t) for t in texts]

        raw_embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64)
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = raw_embeddings / norms

        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        return cls(
            embeddings=embeddings,
            row_ids=row_ids,
            entity_sets=entity_sets,
            texts=texts,
            _faiss_index=index,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        query_entities: Set[str],
        *,
        top_k: int = 10,
        entity_overlap_min: int = 1,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[RetrievalHit]:
        """Retrieve compatible rows.

        Performs dense retrieval via FAISS, then post-filters by entity
        overlap.  Returns up to *top_k* hits sorted by descending score.
        """
        embedder = _get_embedder()
        if embedder is None or self._faiss_index is None:
            return []

        query_emb = embedder.encode([query_text])
        norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        if norm[0, 0] == 0:
            return []
        query_emb = query_emb / norm

        # Retrieve more than top_k to allow for filtering
        fetch_k = min(top_k * 5, len(self.row_ids))
        scores, indices = self._faiss_index.search(query_emb.astype(np.float32), fetch_k)

        exclude = exclude_ids or set()
        hits: List[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            rid = self.row_ids[idx]
            if rid in exclude:
                continue
            ent_overlap = len(query_entities & self.entity_sets[idx])
            if ent_overlap < entity_overlap_min:
                continue
            hits.append(RetrievalHit(
                row_id=rid,
                score=float(score),
                entity_overlap=ent_overlap,
                condition_overlap=ent_overlap,  # simplified: entity overlap proxies condition overlap
            ))
            if len(hits) >= top_k:
                break

        return hits
