"""Semantic matching between predictions and ground-truth research directions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

_EMBED_MODEL_CACHE: dict[str, object] = {}


class SemanticMatcher(Protocol):
    ref_texts: list[str]
    threshold: float
    _sim_threshold: float

    def similarity_matrix(self, preds: list[str], k: int | None = None): ...

    def precision_recall_at_k(self, preds: list[str], k: int) -> tuple[float, float, float]: ...

    def uncovered_references(self, preds: list[str], k: int) -> list[str]: ...


def _precision_recall_from_sims(
    sims,
    *,
    n_preds: int,
    n_gt: int,
    sim_threshold: float,
    k: int,
) -> tuple[float, float, float]:
    matched = float((sims.max(axis=0) >= sim_threshold).sum())
    precision = float((sims.max(axis=1) >= sim_threshold).sum()) / n_preds
    recall = matched / n_gt
    recall_normalized = matched / min(k, n_gt)
    return precision, recall, recall_normalized


def _uncovered_from_sims(sims, ref_texts: list[str], sim_threshold: float) -> list[str]:
    matched: set[int] = set()
    for row in sims:
        for ref_idx, sim in enumerate(row):
            if sim >= sim_threshold:
                matched.add(ref_idx)
    return [ref_texts[i] for i in range(len(ref_texts)) if i not in matched]


@dataclass
class TfidfMatcher:
    """Lexical TF-IDF cosine matcher (ablation / fallback)."""

    ref_texts: list[str]
    threshold: float

    def __post_init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._vectorizer = TfidfVectorizer(max_features=8000, stop_words="english")
        self._ref_matrix = self._vectorizer.fit_transform([t[:4000] for t in self.ref_texts])
        self._cosine = cosine_similarity
        # Historical TF-IDF configs used high nominal thresholds; effective bar was lower.
        self._sim_threshold = max(0.15, self.threshold - 0.35)

    def similarity_matrix(self, preds: list[str], k: int | None = None):
        top = [p[:4000] for p in (preds[:k] if k is not None else preds)]
        if not top:
            return None
        try:
            P = self._vectorizer.transform(top)
            return self._cosine(P, self._ref_matrix)
        except ValueError:
            return None

    def precision_recall_at_k(self, preds: list[str], k: int) -> tuple[float, float, float]:
        top = [p[:4000] for p in preds[:k]]
        if not top or not self.ref_texts:
            return 0.0, 0.0, 0.0
        sims = self.similarity_matrix(top)
        if sims is None:
            return 0.0, 0.0, 0.0
        return _precision_recall_from_sims(
            sims,
            n_preds=len(top),
            n_gt=len(self.ref_texts),
            sim_threshold=self._sim_threshold,
            k=k,
        )

    def uncovered_references(self, preds: list[str], k: int) -> list[str]:
        if not self.ref_texts:
            return []
        sims = self.similarity_matrix(preds, k=k)
        if sims is None:
            return list(self.ref_texts)
        return _uncovered_from_sims(sims, self.ref_texts, self._sim_threshold)


def _load_st_model(model_name: str):
    if model_name in _EMBED_MODEL_CACHE:
        return _EMBED_MODEL_CACHE[model_name]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    _EMBED_MODEL_CACHE[model_name] = model
    return model


@dataclass
class EmbeddingMatcher:
    """Semantic cosine matcher via sentence-transformers (default: MiniLM)."""

    ref_texts: list[str]
    threshold: float
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    _ref_matrix: np.ndarray = field(init=False, repr=False)
    _sim_threshold: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # MiniLM cosine is used as-is (no TF-IDF offset).
        self._sim_threshold = float(self.threshold)
        model = _load_st_model(self.model_name)
        texts = [t[:4000] for t in self.ref_texts]
        emb = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._ref_matrix = np.asarray(emb, dtype=np.float32)

    def similarity_matrix(self, preds: list[str], k: int | None = None):
        top = [p[:4000] for p in (preds[:k] if k is not None else preds)]
        if not top or self._ref_matrix.size == 0:
            return None
        model = _load_st_model(self.model_name)
        pred_emb = model.encode(
            top,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        pred_emb = np.asarray(pred_emb, dtype=np.float32)
        return pred_emb @ self._ref_matrix.T

    def precision_recall_at_k(self, preds: list[str], k: int) -> tuple[float, float, float]:
        top = [p[:4000] for p in preds[:k]]
        if not top or not self.ref_texts:
            return 0.0, 0.0, 0.0
        sims = self.similarity_matrix(top)
        if sims is None:
            return 0.0, 0.0, 0.0
        return _precision_recall_from_sims(
            sims,
            n_preds=len(top),
            n_gt=len(self.ref_texts),
            sim_threshold=self._sim_threshold,
            k=k,
        )

    def uncovered_references(self, preds: list[str], k: int) -> list[str]:
        if not self.ref_texts:
            return []
        sims = self.similarity_matrix(preds, k=k)
        if sims is None:
            return list(self.ref_texts)
        return _uncovered_from_sims(sims, self.ref_texts, self._sim_threshold)


def make_matcher(
    ref_texts: list[str],
    threshold: float,
    *,
    backend: str = "minilm",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> TfidfMatcher | EmbeddingMatcher:
    """Build a semantic matcher; fall back to TF-IDF if MiniLM deps are missing."""
    backend = (backend or "minilm").lower().strip()
    if backend in {"minilm", "embedding", "st", "sentence-transformers"}:
        try:
            return EmbeddingMatcher(ref_texts, threshold, model_name=model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers unavailable — falling back to TF-IDF matcher"
            )
            return TfidfMatcher(ref_texts, threshold)
    if backend in {"tfidf", "tf-idf"}:
        return TfidfMatcher(ref_texts, threshold)
    raise ValueError(f"Unknown matcher backend: {backend!r} (expected minilm|tfidf)")
