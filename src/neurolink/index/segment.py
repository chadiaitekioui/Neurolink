"""Segment abstracts: rules (structure) + PubMedBERT + subject extraction.

Stores a short research-direction *subject* in ``article_segments.question``
with ``qc_score`` = subjectness in [0, 1].
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..db import Database
from ..utils.pubmed_clean import (
    Bucket,
    is_junk_sentence,
    polish_segment_field,
    structure_abstract_sections,
)
from ..utils.config import load_config, make_run_id, resolve_path
from ..utils.torch_device import resolve_torch_device
from .subject import SubjectClassifier, SubjectConfig, extract_subject, get_subject_classifier
from .subject_llm import results_from_sections

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
    device: str = "cpu"  # cpu | cuda | auto (auto: use CUDA when available)
    # Commit every N newly segmented articles (resume-safe on interrupt).
    commit_every: int = 100
    # Extract research-direction subjects (rules + prototype classifier).
    subject: SubjectConfig = field(default_factory=SubjectConfig)


def load_pmids_file(path: str | Path) -> list[str]:
    """Load PMIDs from JSON (list, {\"pmids\": [...]}, or [{\"pmid\": ...}]) or plain text."""
    p = resolve_path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("{"):
        data = json.loads(text)
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return [
                    str(row["pmid"])
                    for row in data
                    if isinstance(row, dict) and row.get("pmid")
                ]
            return [str(x) for x in data if str(x).strip()]
        if isinstance(data, dict):
            if "pmids" in data:
                return [str(x) for x in data["pmids"] if str(x).strip()]
            if "samples" in data:
                return [
                    str(row["pmid"])
                    for row in data["samples"]
                    if isinstance(row, dict) and row.get("pmid")
                ]
        raise ValueError(f"Unsupported PMID JSON in {p}")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _fetch_articles_for_segment(
    conn,
    *,
    pmids: list[str] | None,
    force: bool,
    limit: int | None,
) -> list:
    if pmids:
        placeholders = ",".join("?" * len(pmids))
        if force:
            sql = f"""
                SELECT a.pmid, a.title, a.abstract, a.text_work
                FROM articles a
                WHERE a.pmid IN ({placeholders})
                ORDER BY a.pmid
            """
        else:
            sql = f"""
                SELECT a.pmid, a.title, a.abstract, a.text_work
                FROM articles a
                LEFT JOIN article_segments s ON a.pmid = s.pmid
                WHERE a.pmid IN ({placeholders}) AND s.pmid IS NULL
                ORDER BY a.pmid
            """
        rows = conn.execute(sql, pmids).fetchall()
    else:
        sql = """
            SELECT a.pmid, a.title, a.abstract, a.text_work
            FROM articles a
            LEFT JOIN article_segments s ON a.pmid = s.pmid
            WHERE s.pmid IS NULL
            ORDER BY a.pmid
        """
        if limit is not None:
            sql += " LIMIT ?"
            rows = conn.execute(sql, (limit,)).fetchall()
        else:
            rows = conn.execute(sql).fetchall()
    if pmids and limit is not None:
        rows = rows[:limit]
    return rows


def _split_sentences(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", s).strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]


def _load_pubmedbert(model_name: str, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, model.config.id2label


def _bert_bucket(
    sentence: str,
    tokenizer,
    model,
    id2label: dict[int, str],
    device: str,
) -> Bucket | None:
    import torch

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    label = id2label[int(logits.argmax(-1))]
    return BERT_TO_BUCKET.get(label)


def _bucket_parts(
    text: str,
    tokenizer,
    model,
    id2label: dict[int, str],
    device: str,
) -> tuple[str, str]:
    """Split abstract into question-bucket + results-bucket text (pre-subject)."""
    question_parts: list[str] = []
    results_parts: list[str] = []

    for content, hint, _section in structure_abstract_sections(text):
        if hint is not None:
            (question_parts if hint == "question" else results_parts).append(content)
            continue
        for sent in _split_sentences(content):
            if is_junk_sentence(sent):
                continue
            bucket = _bert_bucket(sent, tokenizer, model, id2label, device)
            if bucket == "question":
                question_parts.append(sent)
            elif bucket == "results":
                results_parts.append(sent)

    if not question_parts and not results_parts and text.strip():
        for sent in _split_sentences(text):
            if is_junk_sentence(sent):
                continue
            bucket = _bert_bucket(sent, tokenizer, model, id2label, device)
            if bucket == "question":
                question_parts.append(sent)
            elif bucket == "results":
                results_parts.append(sent)

    return (
        polish_segment_field(" ".join(question_parts)),
        polish_segment_field(" ".join(results_parts)),
    )


def segment_abstract_llm(
    text: str,
    *,
    title: str | None = None,
    subject_cfg: SubjectConfig | None = None,
    classifier: SubjectClassifier | None = None,
) -> tuple[str, str, float | None]:
    """LLM subject extraction + rule-based results bucket (no PubMedBERT)."""
    cfg = subject_cfg or SubjectConfig()
    results = results_from_sections(text)

    if not cfg.enabled:
        return "", results, None

    extracted = extract_subject(
        text,
        cfg=cfg,
        title=title,
        classifier=classifier,
    )
    if extracted is None:
        return "", results, 0.0
    return extracted.text, results, extracted.subjectness


def segment_abstract(
    text: str,
    tokenizer,
    model,
    id2label: dict[int, str],
    device: str = "cpu",
    *,
    title: str | None = None,
    subject_cfg: SubjectConfig | None = None,
    classifier: SubjectClassifier | None = None,
) -> tuple[str, str, float | None]:
    """Rules + BERT buckets, then subject extraction into a short research direction.

    Returns ``(subject_or_question, results, qc_score)``.
    ``qc_score`` is subjectness when extraction succeeds, else None.
    """
    question_bucket, results = _bucket_parts(text, tokenizer, model, id2label, device)
    cfg = subject_cfg or SubjectConfig()

    if not cfg.enabled:
        return question_bucket, results, None

    # Prefer extracting from full abstract (section labels) when available.
    source = text if text.strip() else question_bucket
    extracted = extract_subject(
        source,
        cfg=cfg,
        title=title,
        classifier=classifier,
    )
    if extracted is None and question_bucket:
        extracted = extract_subject(
            question_bucket,
            cfg=cfg,
            title=title,
            classifier=classifier,
        )
    if extracted is None:
        # Keep cleaned bucket as fallback (may be filtered later at impact).
        return question_bucket, results, 0.0
    return extracted.text, results, extracted.subjectness


def run_segment(
    config_path: str | SegmentConfig,
    run_id: str | None = None,
    *,
    limit: int | None = None,
    pmids: list[str] | None = None,
    force: bool = False,
) -> int:
    cfg = load_config(config_path, SegmentConfig)
    db = Database(resolve_path(cfg.db_path))
    db.init_schema()
    run_id = run_id or make_run_id("segment")
    now = datetime.now(timezone.utc).isoformat()
    commit_every = max(1, cfg.commit_every)

    with db.connect(readonly=True) as conn:
        articles = _fetch_articles_for_segment(
            conn, pmids=pmids, force=force, limit=limit
        )
        total_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        already = total_articles - conn.execute(
            """
            SELECT COUNT(*) FROM articles a
            INNER JOIN article_segments s ON a.pmid = s.pmid
            """
        ).fetchone()[0]

    scope = ""
    if pmids:
        scope = f" pmids={len(pmids)}"
    if limit is not None:
        scope += f" limit={limit}"
    if force:
        scope += " force"

    if not articles:
        logger.info(
            "Segmentation already complete: %d/%d articles (nothing to do%s)",
            already,
            total_articles,
            scope,
        )
        return 0

    if already and not (pmids or limit is not None):
        logger.info(
            "Resuming segmentation: %d/%d already done (%d remaining%s)",
            already,
            total_articles,
            len(articles),
            scope,
        )
    else:
        logger.info("Segmentation: %d articles%s", len(articles), scope)

    device = resolve_torch_device(cfg.device)
    use_llm = cfg.subject.extraction_mode in ("llm", "hybrid")

    tokenizer = model = id2label = None
    if not use_llm:
        try:
            tokenizer, model, id2label = _load_pubmedbert(cfg.pubmedbert_model, device)
            logger.info("PubMedBERT loaded: %s (device=%s)", cfg.pubmedbert_model, device)
        except ImportError as e:
            raise ImportError(
                "segment requires transformers and torch — pip install -e ."
            ) from e
    else:
        logger.info(
            "Segmentation mode=%s — skipping PubMedBERT; LLM=%s",
            cfg.subject.extraction_mode,
            cfg.subject.llm.base_model,
        )

    classifier = None
    if cfg.subject.enabled and cfg.subject.use_level2_classifier:
        classifier = get_subject_classifier(cfg.subject.classifier_model)

    n = 0
    n_subjects = 0
    with db.connect() as conn:
        for row in articles:
            text = (row["text_work"] or row["abstract"] or "").strip()
            if use_llm:
                subject, results, qc = segment_abstract_llm(
                    text,
                    title=row["title"],
                    subject_cfg=cfg.subject,
                    classifier=classifier,
                )
                method = "llm+subject"
            else:
                subject, results, qc = segment_abstract(
                    text,
                    tokenizer,
                    model,
                    id2label,
                    device,
                    title=row["title"],
                    subject_cfg=cfg.subject,
                    classifier=classifier,
                )
                method = "hybrid+subject" if cfg.subject.enabled else "hybrid"
            if qc is not None and qc > 0:
                n_subjects += 1
            conn.execute(
                """
                INSERT INTO article_segments (pmid, question, results, segmentation_method, qc_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pmid) DO UPDATE SET
                    question=excluded.question, results=excluded.results,
                    segmentation_method=excluded.segmentation_method,
                    qc_score=excluded.qc_score, updated_at=excluded.updated_at
                """,
                (row["pmid"], subject, results, method, qc, now),
            )
            n += 1
            if n % commit_every == 0:
                conn.commit()
                done = already + n
                logger.info(
                    "Segmentation: %d/%d (%.1f%%)",
                    done,
                    total_articles,
                    100.0 * done / max(1, total_articles),
                )

    if use_llm:
        try:
            from ..forecast.predict.llm_core import release_gpu_memory

            release_gpu_memory()
            logger.info("Released LLM VRAM after segment stage")
        except ImportError:
            pass

    db.record_run(
        run_id,
        "segment",
        notes=f"{n} new / {already + n} total; subjects={n_subjects}",
    )
    logger.info(
        "Segmentation: %d new articles (%d total); subjects extracted=%d",
        n,
        already + n,
        n_subjects,
    )
    return n
