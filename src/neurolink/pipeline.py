"""Orchestrator across the 3 layers"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .eval.pipeline import run_eval_layer
from .forecast.pipeline import run_centroid_forecast, run_literature_forecast
from .index.pipeline import run_index
from .utils.config import load_config, make_run_id

logger = logging.getLogger(__name__)

TRACK_INDEX = "index"
TRACK_CENTROID = "centroid"
TRACK_LITERATURE = "literature"
TRACK_FULL = "full"


@dataclass
class PipelineConfig:
    track: str = TRACK_FULL
    db_path: str = "data/neurolink.db"
    index_config: str = "config/index/pipeline.yaml"
    forecast_centroid_config: str = "config/forecast/pipeline_centroid.yaml"
    forecast_literature_config: str = "config/forecast/pipeline_literature.yaml"
    eval_config: str = "config/eval/pipeline.yaml"


def run_pipeline(config_path: str | Path) -> None:
    cfg = load_config(config_path, PipelineConfig)
    run_id = make_run_id("pipeline")
    logger.info("Pipeline track=%s run_id=%s", cfg.track, run_id)

    if cfg.track in (TRACK_INDEX, TRACK_FULL):
        run_index(cfg.index_config, run_id)

    if cfg.track in (TRACK_CENTROID, TRACK_FULL):
        run_centroid_forecast(cfg.forecast_centroid_config, run_id)

    if cfg.track in (TRACK_LITERATURE, TRACK_FULL):
        run_literature_forecast(cfg.forecast_literature_config, run_id)

    if cfg.track in (TRACK_CENTROID, TRACK_LITERATURE, TRACK_FULL):
        run_eval_layer(cfg.eval_config, predict_run_id=run_id)

    logger.info("Pipeline finished.")
