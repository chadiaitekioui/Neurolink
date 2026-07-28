"""Fetch citation counts (OpenAlex) and label high-impact research directions.

Impact scores are normalized as citations per year since publication so articles
from different vintages remain comparable.

Propagates ``article_segments`` → ``questions`` as short *subjects*,
with year clamp, subjectness filter, and optional impact reweighting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import requests

from ..db import Database
from ..utils.config import load_config, make_run_id, resolve_path
from .subject import (
    SubjectConfig,
    extract_subject,
    get_subject_classifier,
    weighted_impact,
    year_in_range,
)

logger = logging.getLogger(__name__)

OPENALEX = "https://api.openalex.org/works"


@dataclass
class ImpactConfig:
    db_path: str = "data/neurolink.db"
    critical_percentile: float = 0.90
    delay_seconds: float = 0.1
    mailto: str | None = None
    # Commit + log every N newly fetched citations (resume-safe on interrupt).
    commit_every: int = 100
    # Subject extraction / filtering when rebuilding questions.
    subject: SubjectConfig = field(default_factory=SubjectConfig)


def years_since_publication(pub_year: int, ref_year: int | None = None) -> int:
    ref_year = ref_year or datetime.now().year
    return max(1, ref_year - pub_year)


def citation_rate(
    citation_count: int, pub_year: int, ref_year: int | None = None
) -> float:
    return citation_count / years_since_publication(pub_year, ref_year)


def fetch_citation_count(pmid: str, doi: str | None, mailto: str | None) -> int | None:
    headers = {"User-Agent": "neurolink/0.1"}
    params = {}
    if mailto:
        params["mailto"] = mailto
    for url in (
        f"{OPENALEX}/pmid:{pmid}",
        f"{OPENALEX}/{doi}" if doi and doi.startswith("http") else None,
        f"https://api.openalex.org/works/https://doi.org/{doi}" if doi else None,
    ):
        if not url:
            continue
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return int(data.get("cited_by_count", 0))
        except requests.RequestException as e:
            logger.debug("OpenAlex %s: %s", url, e)
        time.sleep(0.05)
    return None


def _fetch_missing_citations(cfg: ImpactConfig, db: Database) -> int:
    """Fetch OpenAlex counts for articles not yet in citations. Returns newly fetched count."""
    now = datetime.now(timezone.utc).isoformat()
    commit_every = max(1, cfg.commit_every)

    with db.connect(readonly=True) as conn:
        articles = conn.execute(
            """
            SELECT a.pmid, a.doi, a.year
            FROM articles a
            LEFT JOIN citations c ON a.pmid = c.pmid
            WHERE a.year IS NOT NULL AND c.pmid IS NULL
            """
        ).fetchall()
        total_articles = conn.execute(
            "SELECT COUNT(*) FROM articles WHERE year IS NOT NULL"
        ).fetchone()[0]
        already = total_articles - len(articles)

    if already:
        logger.info(
            "Resuming impact: %d/%d articles already have citations (%d remaining)",
            already,
            total_articles,
            len(articles),
        )
    else:
        logger.info("Impact: fetching OpenAlex citations for %d articles", len(articles))

    if not articles:
        return 0

    fetched = 0
    pending: list[tuple[str, int, str]] = []

    def flush(conn) -> None:
        nonlocal pending
        if not pending:
            return
        conn.executemany(
            """
            INSERT INTO citations (pmid, citation_count, source, fetched_at)
            VALUES (?, ?, 'openalex', ?)
            ON CONFLICT(pmid) DO UPDATE SET
                citation_count=excluded.citation_count, fetched_at=excluded.fetched_at
            """,
            pending,
        )
        conn.commit()
        pending = []

    with db.connect() as conn:
        for row in articles:
            cc = fetch_citation_count(row["pmid"], row["doi"], cfg.mailto)
            if cc is None:
                cc = 0
            pending.append((row["pmid"], cc, now))
            fetched += 1
            if len(pending) >= commit_every:
                flush(conn)
                done = already + fetched
                logger.info(
                    "Impact citations: %d/%d (%.1f%%)",
                    done,
                    total_articles,
                    100.0 * done / max(1, total_articles),
                )
            time.sleep(cfg.delay_seconds)
        flush(conn)

    logger.info("Impact citations: fetched %d new rows (total target %d)", fetched, total_articles)
    return fetched


def _resolve_subject_text(
    *,
    segment_question: str,
    abstract: str | None,
    title: str | None,
    qc_score: float | None,
    cfg: SubjectConfig,
    classifier,
    segmentation_method: str | None = None,
) -> tuple[str | None, float]:
    """Return (subject_text, subjectness) or (None, 0) if rejected."""
    if not cfg.enabled:
        text = (segment_question or "").strip()
        return (text or None, 1.0 if text else 0.0)

    seg_q = (segment_question or "").strip()
    method = (segmentation_method or "").lower()
    qc = float(qc_score or 0.0)
    # Keep failed LLM drafts in article_segments; do not rules-reextract or propagate.
    if "llm" in method and qc < cfg.min_subjectness and seg_q:
        return None, qc

    # Re-extract when qc missing/low or segment still looks like a long abstract blob.
    words = len(seg_q.split())
    needs_extract = (
        qc_score is None
        or qc < cfg.min_subjectness
        or words > cfg.max_words + 5
        or words < cfg.absolute_min_words
    )
    if not needs_extract and seg_q:
        return seg_q, qc

    source = (abstract or segment_question or "").strip()
    # Impact runs on CPU after segment; never reload the index LLM here.
    reextract_cfg = cfg if cfg.extraction_mode == "rules" else replace(cfg, extraction_mode="rules")
    extracted = extract_subject(
        source,
        cfg=reextract_cfg,
        title=title,
        classifier=classifier,
    )
    if extracted is None:
        return None, 0.0
    if extracted.subjectness < cfg.min_subjectness:
        return None, extracted.subjectness
    return extracted.text, extracted.subjectness


def _score_and_propagate(cfg: ImpactConfig, db: Database) -> int:
    """Compute impact scores from citations and rebuild questions from subjects."""
    ref_year = datetime.now().year
    subject_cfg = cfg.subject

    classifier = None
    if subject_cfg.enabled and subject_cfg.use_level2_classifier:
        classifier = get_subject_classifier(subject_cfg.classifier_model)

    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT a.pmid, a.year, COALESCE(c.citation_count, 0) AS citation_count
            FROM articles a
            LEFT JOIN citations c ON a.pmid = c.pmid
            WHERE a.year IS NOT NULL
            """
        ).fetchall()

        counts_by_year: dict[int, list[tuple[str, float]]] = {}
        for row in rows:
            year = int(row["year"])
            if not year_in_range(year, subject_cfg):
                continue
            rate = citation_rate(int(row["citation_count"]), year, ref_year)
            counts_by_year.setdefault(year, []).append((row["pmid"], rate))

        for year, items in counts_by_year.items():
            if not items:
                continue
            scores = sorted(rate for _, rate in items)
            idx = max(0, int(len(scores) * cfg.critical_percentile) - 1)
            threshold = scores[idx]
            for pmid, rate in items:
                is_critical = 1 if rate >= threshold and rate > 0 else 0
                conn.execute(
                    """
                    INSERT INTO article_impact (pmid, impact_score, is_critical)
                    VALUES (?, ?, ?)
                    ON CONFLICT(pmid) DO UPDATE SET
                        impact_score=excluded.impact_score,
                        is_critical=excluded.is_critical
                    """,
                    (pmid, rate, is_critical),
                )

        conn.execute("DELETE FROM questions")
        segments = conn.execute(
            """
            SELECT s.pmid, s.question, s.qc_score, s.results,
                   a.year, a.title, a.abstract, a.text_work,
                   i.impact_score, i.is_critical
            FROM article_segments s
            JOIN articles a ON s.pmid = a.pmid
            LEFT JOIN article_impact i ON s.pmid = i.pmid
            WHERE s.question IS NOT NULL AND TRIM(s.question) != ''
            """
        ).fetchall()

        n_kept = 0
        n_dropped_year = 0
        n_dropped_subject = 0
        for seg in segments:
            year = seg["year"]
            if year is None or not year_in_range(int(year), subject_cfg):
                n_dropped_year += 1
                continue

            abstract = (seg["text_work"] or seg["abstract"] or "").strip()
            subject_text, subjectness = _resolve_subject_text(
                segment_question=seg["question"] or "",
                abstract=abstract,
                title=seg["title"],
                qc_score=seg["qc_score"],
                cfg=subject_cfg,
                classifier=classifier,
                segmentation_method=seg["segmentation_method"],
            )
            if not subject_text:
                n_dropped_subject += 1
                continue

            # Persist refined subject + qc back onto the segment for resume/debug.
            conn.execute(
                """
                UPDATE article_segments
                SET question = ?, qc_score = ?,
                    segmentation_method = CASE
                        WHEN COALESCE(segmentation_method, '') LIKE '%+subject' THEN segmentation_method
                        ELSE COALESCE(segmentation_method, 'hybrid') || '+subject'
                    END,
                    updated_at = ?
                WHERE pmid = ?
                """,
                (
                    subject_text,
                    subjectness,
                    datetime.now(timezone.utc).isoformat(),
                    seg["pmid"],
                ),
            )

            impact = weighted_impact(seg["impact_score"], subjectness, subject_cfg)
            conn.execute(
                """
                INSERT INTO questions (pmid, question_text, year, impact_score, is_critical)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    seg["pmid"],
                    subject_text,
                    int(year),
                    impact,
                    seg["is_critical"] or 0,
                ),
            )
            n_kept += 1

        logger.info(
            "Impact subjects: kept=%d dropped_year=%d dropped_subject=%d "
            "(year_range=%d-%d, min_subjectness=%.2f)",
            n_kept,
            n_dropped_year,
            n_dropped_subject,
            subject_cfg.year_min,
            subject_cfg.year_max,
            subject_cfg.min_subjectness,
        )

    return n_kept


def run_impact(config_path: str | ImpactConfig, run_id: str | None = None) -> int:
    cfg = load_config(config_path, ImpactConfig)
    db = Database(resolve_path(cfg.db_path))
    db.init_schema()
    run_id = run_id or make_run_id("impact")

    _fetch_missing_citations(cfg, db)
    n = _score_and_propagate(cfg, db)

    db.record_run(run_id, "impact", notes=f"{n} subjects")
    logger.info("Impact: %d subjects propagated", n)
    return n
