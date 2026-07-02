"""Forecast layer orchestration (centroid or literature track)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..db import Database
from ..index.pipeline import check_index_ready
from ..utils.config import load_config, make_run_id, resolve_path
from .predict.models import (
    MODEL_CENTROID_TRAJECTORY,
    MODEL_LITERATURE_LORA,
    PredictConfig,
    run_predict,
)
from .topics import run_topics
from .train import (
    check_literature_adapter,
    infer_year_max,
    run_train_literature,
    run_train_literature_errors,
)

logger = logging.getLogger(__name__)

CENTROID_STAGES = ("topics", "predict")
LITERATURE_STAGES = ("train_literature", "predict")


@dataclass
class ForecastPipelineConfig:
    track: str = "centroid"  # centroid | literature
    db_path: str = "data/neurolink.db"
    topics_config: str = "config/forecast/topics.yaml"
    predict_config: str = "config/forecast/predict_literature.yaml"
    year_max: int | None = None
    error_target_year: int | None = None
    pred_run_id: str | None = None
    stages: list[str] = field(default_factory=list)


def _resolve_stages(cfg: ForecastPipelineConfig) -> list[str]:
    if cfg.stages:
        return cfg.stages
    return (
        list(CENTROID_STAGES)
        if cfg.track == "centroid"
        else list(LITERATURE_STAGES)
    )


def check_centroid_ready(db_path: str | Path) -> None:
    db = Database(resolve_path(db_path))
    with db.connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM topic_centroid_snapshots").fetchone()[0]
    if n == 0:
        raise RuntimeError(
            "Centroid forecast not ready: no topic snapshots. Run topics stage first."
        )


def run_centroid_forecast(config_path: str | Path, run_id: str | None = None) -> str:
    cfg = load_config(config_path, ForecastPipelineConfig)
    stages = _resolve_stages(cfg)
    if cfg.track != "centroid":
        raise ValueError(f"Expected track=centroid, got {cfg.track!r}")
    check_index_ready(cfg.db_path)
    run_id = run_id or make_run_id("forecast_centroid")
    logger.info("Forecast centroid run_id=%s", run_id)

    if "topics" in stages:
        run_topics(cfg.topics_config, run_id=run_id)

    if "predict" in stages:
        check_centroid_ready(cfg.db_path)
        predict_cfg = load_config(cfg.predict_config, PredictConfig)
        predict_cfg.models = [MODEL_CENTROID_TRAJECTORY]
        run_predict(predict_cfg, run_id)

    logger.info("Forecast centroid finished.")
    return run_id


def run_literature_forecast(config_path: str | Path, run_id: str | None = None) -> str:
    cfg = load_config(config_path, ForecastPipelineConfig)
    stages = _resolve_stages(cfg)
    if cfg.track != "literature":
        raise ValueError(f"Expected track=literature, got {cfg.track!r}")
    check_index_ready(cfg.db_path)
    run_id = run_id or make_run_id("forecast_literature")
    logger.info("Forecast literature run_id=%s", run_id)

    year_max = cfg.year_max

    if "train_literature_errors" in stages:
        if cfg.error_target_year is None:
            raise ValueError("error_target_year required for train_literature_errors stage")
        run_train_literature_errors(
            cfg.predict_config,
            target_year=cfg.error_target_year,
            pred_run_id=cfg.pred_run_id,
            run_id=run_id,
        )

    if "train_literature" in stages:
        year_max = run_train_literature(cfg.predict_config, year_max, run_id)

    if "predict" in stages:
        predict_cfg = load_config(cfg.predict_config, PredictConfig)
        if year_max is None:
            year_max = infer_year_max(cfg.db_path, predict_cfg)
        if year_max is None or year_max <= 0:
            logger.warning("Cannot predict literature_lora without a valid year_max")
            return run_id
        check_literature_adapter(predict_cfg.literature, year_max)
        predict_cfg.models = [MODEL_LITERATURE_LORA]
        run_predict(predict_cfg, run_id)

    logger.info("Forecast literature finished.")
    return run_id
