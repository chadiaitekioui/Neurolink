"""Indexing"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..db import Database
from ..utils.config import load_config, make_run_id, resolve_path
from .collect import run_collect
from .embed import run_embed
from .impact import run_impact
from .segment import run_segment

logger = logging.getLogger(__name__)

INDEX_STAGES = ("collect", "segment", "impact", "embed")


@dataclass
class IndexPipelineConfig:
    db_path: str = "data/neurolink.db"
    collect_config: str = "config/index/collect.yaml"
    segment_config: str = "config/index/segment.yaml"
    impact_config: str = "config/index/impact.yaml"
    embed_config: str = "config/index/embed.yaml"
    stages: list[str] = field(default_factory=lambda: list(INDEX_STAGES))
    # Skip the whole index when questions + embeddings are already present.
    skip_if_ready: bool = True


@dataclass(frozen=True)
class IndexCounts:
    articles: int
    segments: int
    segments_missing: int
    citations: int
    questions: int
    questions_unembedded: int

    @property
    def ready(self) -> bool:
        return self.questions > 0 and self.questions_unembedded == 0


def get_index_counts(db_path: str | Path) -> IndexCounts:
    path = resolve_path(db_path)
    if not path.exists():
        return IndexCounts(0, 0, 0, 0, 0, 0)
    db = Database(path)
    with db.connect(readonly=True) as conn:
        articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        segments = conn.execute("SELECT COUNT(*) FROM article_segments").fetchone()[0]
        segments_missing = conn.execute(
            """
            SELECT COUNT(*) FROM articles a
            LEFT JOIN article_segments s ON a.pmid = s.pmid
            WHERE s.pmid IS NULL
            """
        ).fetchone()[0]
        try:
            citations = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
        except Exception:
            citations = 0
        questions = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        questions_unembedded = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE embedding IS NULL"
        ).fetchone()[0]
    return IndexCounts(
        articles=articles,
        segments=segments,
        segments_missing=segments_missing,
        citations=citations,
        questions=questions,
        questions_unembedded=questions_unembedded,
    )


def is_index_ready(db_path: str | Path) -> bool:
    return get_index_counts(db_path).ready


def check_index_ready(db_path: str | Path) -> None:
    counts = get_index_counts(db_path)
    if counts.questions == 0:
        raise RuntimeError("Index incomplete: no questions in database. Run index pipeline.")
    if counts.questions_unembedded > 0:
        raise RuntimeError(
            f"Index incomplete: {counts.questions_unembedded} questions lack embeddings. "
            "Run embed stage."
        )


def run_index(config_path: str | Path | IndexPipelineConfig, run_id: str | None = None) -> str:
    cfg = load_config(config_path, IndexPipelineConfig)
    run_id = run_id or make_run_id("index")
    logger.info("Index layer run_id=%s", run_id)

    counts = get_index_counts(cfg.db_path)
    logger.info(
        "Index status: articles=%d segments=%d(+%d missing) questions=%d "
        "unembedded=%d citations=%d",
        counts.articles,
        counts.segments,
        counts.segments_missing,
        counts.questions,
        counts.questions_unembedded,
        counts.citations,
    )

    if cfg.skip_if_ready and counts.ready:
        logger.info("Index already complete — skipping all stages")
        return run_id

    if "collect" in cfg.stages:
        run_collect(cfg.collect_config)

    if "segment" in cfg.stages:
        # run_segment itself skips / resumes incomplete articles.
        run_segment(cfg.segment_config, run_id)

    if "impact" in cfg.stages:
        # Cheap + idempotent: fills missing citations and rebuilds questions.
        run_impact(cfg.impact_config, run_id)

    if "embed" in cfg.stages:
        # run_embed itself skips questions that already have embeddings.
        run_embed(cfg.embed_config, run_id)

    check_index_ready(cfg.db_path)
    logger.info("Index layer finished.")
    return run_id
