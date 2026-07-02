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


def check_index_ready(db_path: str | Path) -> None:
    db = Database(resolve_path(db_path))
    with db.connect() as conn:
        n_questions = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        n_embedded = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE embedding IS NOT NULL"
        ).fetchone()[0]
    if n_questions == 0:
        raise RuntimeError("Index incomplete: no questions in database. Run index pipeline.")
    if n_embedded == 0:
        raise RuntimeError("Index incomplete: questions have no embeddings. Run embed stage.")


def run_index(config_path: str | Path | IndexPipelineConfig, run_id: str | None = None) -> str:
    cfg = load_config(config_path, IndexPipelineConfig)
    run_id = run_id or make_run_id("index")
    logger.info("Index layer run_id=%s", run_id)

    if "collect" in cfg.stages:
        run_collect(cfg.collect_config)

    if "segment" in cfg.stages:
        run_segment(cfg.segment_config, run_id)

    if "impact" in cfg.stages:
        run_impact(cfg.impact_config, run_id)

    if "embed" in cfg.stages:
        run_embed(cfg.embed_config, run_id)

    check_index_ready(cfg.db_path)
    logger.info("Index layer finished.")
    return run_id
