"""Segment abstracts: rules (structure) + PubMedBERT (bucket assignment)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from ..db import Database
from ..utils.pubmed_clean import Bucket, is_junk_sentence, polish_segment_field, structure_abstract
from ..utils.config import load_config, make_run_id, resolve_path

logger = logging.getLogger(__name__)

BERT_TO_BUCKET: dict[str, Bucket] = {
    "OBJECTIVE": "question",
    "BACKGROUND": "question",
    "INTRODUCTION": "question",
    "METHODS": "results",
    "RESULTS": "results",
    "CONCLUSIONS": "results",
}


@dataclass
class SegmentConfig:
    db_path: str = "data/neurolink.db"
    pubmedbert_model: str = (
        "ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section"
    )


def _split_sentences(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", s).strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]


def _load_pubmedbert(model_name: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model, model.config.id2label


def _bert_bucket(sentence: str, tokenizer, model, id2label: dict[int, str]) -> Bucket | None:
    import torch

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    label = id2label[int(logits.argmax(-1))]
    return BERT_TO_BUCKET.get(label)


def segment_abstract(
    text: str,
    tokenizer,
    model,
    id2label: dict[int, str],
) -> tuple[str, str]:
    """Rules split IMRaD sections; BERT classifies unstructured sentences."""
    question_parts: list[str] = []
    results_parts: list[str] = []

    for content, hint in structure_abstract(text):
        if hint is not None:
            (question_parts if hint == "question" else results_parts).append(content)
            continue
        for sent in _split_sentences(content):
            if is_junk_sentence(sent):
                continue
            bucket = _bert_bucket(sent, tokenizer, model, id2label)
            if bucket == "question":
                question_parts.append(sent)
            elif bucket == "results":
                results_parts.append(sent)

    if not question_parts and not results_parts and text.strip():
        for sent in _split_sentences(text):
            if is_junk_sentence(sent):
                continue
            bucket = _bert_bucket(sent, tokenizer, model, id2label)
            if bucket == "question":
                question_parts.append(sent)
            elif bucket == "results":
                results_parts.append(sent)

    return polish_segment_field(" ".join(question_parts)), polish_segment_field(" ".join(results_parts))


def run_segment(config_path: str | SegmentConfig, run_id: str | None = None) -> int:
    cfg = load_config(config_path, SegmentConfig)
    db = Database(resolve_path(cfg.db_path))
    db.init_schema()
    run_id = run_id or make_run_id("segment")
    now = datetime.now(timezone.utc).isoformat()

    try:
        pubmedbert = _load_pubmedbert(cfg.pubmedbert_model)
        logger.info("PubMedBERT loaded: %s", cfg.pubmedbert_model)
    except ImportError as e:
        raise ImportError(
            "segment requires transformers and torch — pip install -e ."
        ) from e

    n = 0
    with db.connect() as conn:
        articles = conn.execute(
            "SELECT pmid, abstract, text_work FROM articles"
        ).fetchall()
        for row in articles:
            text = (row["text_work"] or row["abstract"] or "").strip()
            question, results = segment_abstract(text, *pubmedbert)
            conn.execute(
                """
                INSERT INTO article_segments (pmid, question, results, segmentation_method, qc_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pmid) DO UPDATE SET
                    question=excluded.question, results=excluded.results,
                    segmentation_method=excluded.segmentation_method, updated_at=excluded.updated_at
                """,
                (row["pmid"], question, results, "hybrid", None, now),
            )
            n += 1

    db.record_run(run_id, "segment", notes=f"{n} articles")
    logger.info("Segmentation: %d articles", n)
    return n
