"""Fetch citation counts (OpenAlex) and label high-impact questions.

Impact scores are normalized as citations per year since publication so articles
from different vintages remain comparable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from ..db import Database
from ..utils.config import load_config, make_run_id, resolve_path

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

    with db.connect() as conn:
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


def _score_and_propagate(cfg: ImpactConfig, db: Database) -> int:
    """Compute impact scores from citations and rebuild questions from segments."""
    ref_year = datetime.now().year

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
            SELECT s.pmid, s.question, a.year, i.impact_score, i.is_critical
            FROM article_segments s
            JOIN articles a ON s.pmid = a.pmid
            LEFT JOIN article_impact i ON s.pmid = i.pmid
            WHERE s.question IS NOT NULL AND TRIM(s.question) != ''
            """
        ).fetchall()
        for seg in segments:
            conn.execute(
                """
                INSERT INTO questions (pmid, question_text, year, impact_score, is_critical)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    seg["pmid"],
                    seg["question"],
                    seg["year"],
                    seg["impact_score"],
                    seg["is_critical"] or 0,
                ),
            )

    return len(segments)


def run_impact(config_path: str | ImpactConfig, run_id: str | None = None) -> int:
    cfg = load_config(config_path, ImpactConfig)
    db = Database(resolve_path(cfg.db_path))
    db.init_schema()
    run_id = run_id or make_run_id("impact")

    _fetch_missing_citations(cfg, db)
    n = _score_and_propagate(cfg, db)

    db.record_run(run_id, "impact", notes=f"{n} questions")
    logger.info("Impact: %d questions propagated", n)
    return n
