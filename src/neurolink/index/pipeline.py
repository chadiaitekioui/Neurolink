"""Indexing pipeline: collect → direction → impact → embed."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..db import Database
from ..utils.config import load_config, make_run_id, resolve_path
from .collect import run_collect
from .embed import run_embed
from .impact import run_impact
from .subject import run_directions

logger = logging.getLogger(__name__)

INDEX_STAGES = ("collect", "direction", "impact", "embed")


@dataclass
class IndexPipelineConfig:
    db_path: str = "data/neurolink.db"
    collect_config: str = "config/index/collect.yaml"
    direction_config: str = "config/index/direction.yaml"
    impact_config: str = "config/index/impact.yaml"
    embed_config: str = "config/index/embed.yaml"
    stages: list[str] = field(default_factory=lambda: list(INDEX_STAGES))
    skip_if_ready: bool = True


@dataclass(frozen=True)
class IndexCounts:
    articles: int
    directions: int
    directions_missing: int
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
        directions = conn.execute("SELECT COUNT(*) FROM article_segments").fetchone()[0]
        directions_missing = conn.execute(
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
        directions=directions,
        directions_missing=directions_missing,
        citations=citations,
        questions=questions,
        questions_unembedded=questions_unembedded,
    )


def is_index_ready(db_path: str | Path) -> bool:
    return get_index_counts(db_path).ready


def check_index_ready(db_path: str | Path) -> tuple[bool, str]:
    counts = get_index_counts(db_path)
    if counts.questions == 0:
        return False, "no questions — run login-index + direction-embed"
    if counts.questions_unembedded > 0:
        return False, f"{counts.questions_unembedded} questions lack embeddings — run embed"
    return True, "ready"


def run_index(config_path: str | Path | IndexPipelineConfig, run_id: str | None = None) -> str:
    cfg = load_config(config_path, IndexPipelineConfig)
    run_id = run_id or make_run_id("index")
    logger.info("Index layer run_id=%s", run_id)

    counts = get_index_counts(cfg.db_path)
    logger.info(
        "Index status: articles=%d directions=%d(+%d missing) questions=%d "
        "unembedded=%d citations=%d",
        counts.articles,
        counts.directions,
        counts.directions_missing,
        counts.questions,
        counts.questions_unembedded,
        counts.citations,
    )

    if cfg.skip_if_ready and counts.ready:
        logger.info("Index already complete — skipping all stages")
        return run_id

    if "collect" in cfg.stages:
        run_collect(cfg.collect_config)

    if "direction" in cfg.stages:
        run_directions(cfg.direction_config, run_id)

    if "impact" in cfg.stages:
        run_impact(cfg.impact_config, run_id)

    if "embed" in cfg.stages:
        run_embed(cfg.embed_config, run_id)

    ok, msg = check_index_ready(cfg.db_path)
    if not ok:
        raise RuntimeError(f"Index incomplete: {msg}")
    logger.info("Index layer finished.")
    return run_id
