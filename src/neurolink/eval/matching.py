"""Semantic matching between predictions and ground-truth questions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TfidfMatcher:
    ref_texts: list[str]
    threshold: float

    def __post_init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._vectorizer = TfidfVectorizer(max_features=8000, stop_words="english")
        self._ref_matrix = self._vectorizer.fit_transform([t[:4000] for t in self.ref_texts])
        self._cosine = cosine_similarity
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

    def precision_recall_at_k(self, preds: list[str], k: int) -> tuple[float, float]:
        top = [p[:4000] for p in preds[:k]]
        if not top or not self.ref_texts:
            return 0.0, 0.0
        sims = self.similarity_matrix(top)
        if sims is None:
            return 0.0, 0.0

        precision = float((sims.max(axis=1) >= self._sim_threshold).sum()) / len(top)
        recall = float((sims.max(axis=0) >= self._sim_threshold).sum()) / len(self.ref_texts)
        return precision, recall

    def uncovered_references(self, preds: list[str], k: int) -> list[str]:
        """Ground-truth questions not semantically matched by any top-k prediction."""
        if not self.ref_texts:
            return []
        sims = self.similarity_matrix(preds, k=k)
        if sims is None:
            return list(self.ref_texts)

        matched: set[int] = set()
        for row in sims:
            for ref_idx, sim in enumerate(row):
                if sim >= self._sim_threshold:
                    matched.add(ref_idx)
        return [self.ref_texts[i] for i in range(len(self.ref_texts)) if i not in matched]
