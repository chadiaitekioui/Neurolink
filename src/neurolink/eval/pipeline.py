"""Eval layer orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.config import load_config, make_run_id
from .metrics import EvalConfig, run_eval

logger = logging.getLogger(__name__)


@dataclass
class EvalPipelineConfig:
    db_path: str = "data/neurolink.db"
    eval_config: str = "config/eval/scenarios/eval_compare_2022.yaml"
    predict_run_id: str | None = None
    stages: list[str] = field(default_factory=lambda: ["eval"])


def run_eval_layer(
    config_path: str | Path,
    run_id: str | None = None,
    predict_run_id: str | None = None,
) -> str:
    cfg = load_config(config_path, EvalPipelineConfig)
    eval_run = run_id or make_run_id("eval")
    logger.info("Eval layer run_id=%s", eval_run)

    if "eval" in cfg.stages:
        eval_cfg = load_config(cfg.eval_config, EvalConfig)
        if predict_run_id or cfg.predict_run_id:
            eval_cfg.run_id = predict_run_id or cfg.predict_run_id
        run_eval(eval_cfg, eval_run)

    logger.info("Eval layer finished.")
    return eval_run
