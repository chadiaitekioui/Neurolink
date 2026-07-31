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
    # "none"/"model" → MiniLM (default); "tfidf" → force sklearn TF-IDF.
    fallback: str = "none"


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

    with db.connect(readonly=True) as conn:
        total = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        rows = conn.execute(
            """
            SELECT id, question_text FROM questions
            WHERE embedding IS NULL
            ORDER BY id
            """
        ).fetchall()
        already = total - len(rows)

    if total == 0:
        logger.warning("No questions to embed — run impact after direction extraction first")
        return 0

    if not rows:
        logger.info("Embeddings already complete: %d/%d questions", already, total)
        return 0

    if already:
        logger.info(
            "Resuming embed: %d/%d already done (%d remaining)",
            already,
            total,
            len(rows),
        )
    else:
        logger.info("Embeddings: %d questions", len(rows))

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

    db.record_run(run_id, "embed", notes=f"{method}; {len(ids)} new")
    logger.info("Embeddings: %d new questions (%s); %d total", len(ids), method, total)
    return len(ids)
