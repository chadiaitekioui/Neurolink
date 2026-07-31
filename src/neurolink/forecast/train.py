"""Literature LoRA training (fit step, separate from predict)."""

from __future__ import annotations

import logging
from pathlib import Path

from ..db import Database
from ..utils.config import infer_test_years, load_config, make_run_id, resolve_path
from .predict.literature_lora import (
    LiteratureLoraConfig,
    _adapter_path,
    adapter_exists,
    train_literature_lora,
)
from .predict.models import PredictConfig

logger = logging.getLogger(__name__)


def infer_year_max(db_path: str | Path, predict_cfg: PredictConfig) -> int | None:
    db = Database(resolve_path(db_path))
    with db.connect() as conn:
        test_years = infer_test_years(conn, predict_cfg.test_years)
    return max(test_years) - 1 if test_years else None


def calibration_years(
    available: list[int],
    target_year: int,
    calibration_start: int | None = None,
) -> list[int]:
    """Years N before forecasting target_year (helper for tooling)."""
    if target_year <= 1 or not available:
        return []
    end = target_year - 1
    start = calibration_start if calibration_start is not None else available[0] + 1
    return sorted(y for y in available if start <= y <= end)


def run_train_literature(
    config_path: str | Path | PredictConfig,
    year_max: int | None = None,
    run_id: str | None = None,
    *,
    skip_if_exists: bool = False,
) -> int:
    cfg = (
        load_config(config_path, PredictConfig)
        if isinstance(config_path, (str, Path))
        else config_path
    )
    db = Database(resolve_path(cfg.db_path))
    run_id = run_id or make_run_id("train_literature")

    with db.connect() as conn:
        if year_max is None:
            test_years = infer_test_years(conn, cfg.test_years)
            if not test_years:
                logger.warning("No test years — cannot infer year_max for training")
                return 0
            year_max = max(test_years) - 1
        if year_max <= 0:
            logger.warning("year_max=%d too small for LoRA training", year_max)
            return 0

        if skip_if_exists and adapter_exists(cfg.literature, year_max):
            adapter = _adapter_path(cfg.literature, year_max) / "lora"
            logger.info(
                "Skipping LoRA training — adapter already exists: %s",
                adapter,
            )
            db.record_run(
                run_id,
                "train_literature",
                notes=f"year_max={year_max}, skipped (adapter exists)",
            )
            return year_max

        n_examples = train_literature_lora(conn, year_max, cfg.literature)

    db.record_run(run_id, "train_literature", notes=f"year_max={year_max}, examples={n_examples}")
    logger.info("Literature LoRA training finished (year_max=%d)", year_max)
    return year_max


def check_literature_adapter(cfg: LiteratureLoraConfig, year_max: int) -> None:
    adapter = _adapter_path(cfg, year_max) / "lora"
    if not adapter.exists():
        raise RuntimeError(
            f"Literature LoRA adapter missing at {adapter}. "
            f"Train first (year_max={year_max})."
        )
