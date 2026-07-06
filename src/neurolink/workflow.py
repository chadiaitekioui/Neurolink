"""End-to-end benchmark workflow (index → dual LoRA anchors → eval → forecast)."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, replace

from .eval import EvalConfig, run_eval
from .forecast import run_benchmark, run_predict, run_train_literature
from .index import IndexPipelineConfig, run_index
from .index.segment import SegmentConfig
from .utils.config import load_config, make_run_id, resolve_path
from .utils.torch_device import resolve_torch_device

logger = logging.getLogger(__name__)

PREDICT_LITERATURE_CONFIG = "config/forecast/predict_literature.yaml"
PREDICT_COMPARE_CONFIG = "config/forecast/predict_compare.yaml"
INDEX_PIPELINE_CONFIG = "config/index/pipeline.yaml"
SEGMENT_CONFIG = "config/index/segment.yaml"
EVAL_COMPARE_2022_CONFIG = "config/eval/scenarios/eval_compare_2022.yaml"
EVAL_COMPARE_2025_CONFIG = "config/eval/scenarios/eval_compare_2025.yaml"
PREDICT_2027_CONFIG = "config/forecast/scenarios/predict_2027.yaml"


@dataclass
class CompleteWorkflowConfig:
    db_path: str = "data/neurolink.db"
    skip_index: bool = False
    segment_device: str = "auto"
    lora_anchor_first: int = 2022
    lora_anchor_second: int = 2025
    forecast_year: int = 2027
    predict_config: str = PREDICT_LITERATURE_CONFIG
    compare_config: str = PREDICT_COMPARE_CONFIG
    eval_first_config: str = EVAL_COMPARE_2022_CONFIG
    eval_second_config: str = EVAL_COMPARE_2025_CONFIG
    forecast_config: str = PREDICT_2027_CONFIG


def _run_index(cfg: CompleteWorkflowConfig, run_id: str) -> None:
    index_cfg = load_config(INDEX_PIPELINE_CONFIG, IndexPipelineConfig)
    segment_cfg = replace(
        load_config(SEGMENT_CONFIG, SegmentConfig),
        device=resolve_torch_device(cfg.segment_device),
    )
    run_index(replace(index_cfg, segment_config=segment_cfg), run_id)


def _benchmark_and_eval(
    *,
    compare_config: str,
    eval_config: str,
    lora_year_max: int,
) -> str:
    pred_run_id, _, _ = run_benchmark(
        compare_config,
        lora_year_max=lora_year_max,
        run_id=make_run_id(f"benchmark_{lora_year_max}"),
    )
    eval_cfg = load_config(eval_config, EvalConfig)
    eval_cfg = replace(eval_cfg, run_id=pred_run_id)
    run_eval(eval_cfg, run_id=make_run_id(f"eval_{lora_year_max}"))
    return pred_run_id


def run_complete_workflow(
    config: CompleteWorkflowConfig | None = None,
    *,
    run_id: str | None = None,
) -> str:
    """
    Full protocol: index → LoRA@T1 → benchmark+eval → LoRA@T2 → benchmark+eval → forecast.

    GPU required for segment, LoRA training, and LLM benchmark stages.
    """
    cfg = config or CompleteWorkflowConfig()
    run_id = run_id or make_run_id("workflow")

    if not cfg.skip_index:
        logger.info("Running full index pipeline")
        _run_index(cfg, run_id)
    else:
        logger.info("Skipping index")

    logger.info("Train LoRA year_max=%d", cfg.lora_anchor_first)
    run_train_literature(cfg.predict_config, year_max=cfg.lora_anchor_first, run_id=run_id)

    logger.info("Benchmark + eval (anchor %d)", cfg.lora_anchor_first)
    _benchmark_and_eval(
        compare_config=cfg.compare_config,
        eval_config=cfg.eval_first_config,
        lora_year_max=cfg.lora_anchor_first,
    )
    summary = resolve_path("eval/summary.md")
    if summary.is_file():
        shutil.copy(summary, resolve_path("eval/summary_job2.md"))

    logger.info("Train LoRA year_max=%d", cfg.lora_anchor_second)
    run_train_literature(cfg.predict_config, year_max=cfg.lora_anchor_second, run_id=run_id)

    logger.info("Benchmark + eval (anchor %d)", cfg.lora_anchor_second)
    _benchmark_and_eval(
        compare_config=cfg.compare_config,
        eval_config=cfg.eval_second_config,
        lora_year_max=cfg.lora_anchor_second,
    )

    logger.info("Forecast year %d (no ground-truth eval)", cfg.forecast_year)
    run_predict(cfg.forecast_config, run_id=make_run_id(f"predict_{cfg.forecast_year}"))

    logger.info("Complete workflow finished (run_id=%s)", run_id)
    return run_id
