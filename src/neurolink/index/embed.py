"""Embed research questions into dense vectors."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass

import numpy as np

from ..db import Database
from ..utils.config import load_config, make_run_id, resolve_path

logger = logging.getLogger(__name__)


@dataclass
class EmbedConfig:
    db_path: str = "data/neurolink.db"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    fallback: str = "tfidf"  # tfidf when sentence-transformers is missing


def _embed_tfidf(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=384, stop_words="english")
    return vec.fit_transform(texts).toarray().astype(np.float32)


def _embed_st(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True)


def run_embed(config_path: str | EmbedConfig, run_id: str | None = None) -> int:
    cfg = load_config(config_path, EmbedConfig)
    db = Database(resolve_path(cfg.db_path))
    run_id = run_id or make_run_id("embed")

    with db.connect() as conn:
        rows = conn.execute("SELECT id, question_text FROM questions ORDER BY id").fetchall()
    if not rows:
        logger.warning("No questions to embed")
        return 0

    ids = [r["id"] for r in rows]
    texts = [r["question_text"] for r in rows]

    if cfg.fallback == "tfidf":
        vectors = _embed_tfidf(texts)
        method = "tfidf"
    else:
        try:
            vectors = _embed_st(texts, cfg.model_name, cfg.batch_size)
            method = cfg.model_name
        except ImportError:
            logger.warning("sentence-transformers missing — TF-IDF fallback")
            vectors = _embed_tfidf(texts)
            method = "tfidf"

    with db.connect() as conn:
        for qid, vec in zip(ids, vectors):
            conn.execute(
                "UPDATE questions SET embedding = ? WHERE id = ?",
                (pickle.dumps(vec.astype(np.float32)), qid),
            )

    db.record_run(run_id, "embed", notes=method)
    logger.info("Embeddings: %d questions (%s)", len(ids), method)
    return len(ids)
