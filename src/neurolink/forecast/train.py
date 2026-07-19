"""Literature LoRA training (fit step, separate from predict)."""

from __future__ import annotations

import logging
from pathlib import Path

from ..db import Database
from ..utils.config import available_question_years, infer_test_years, load_config, make_run_id, resolve_path
from .predict.literature_lora import (
    LiteratureLoraConfig,
    _adapter_path,
    adapter_exists,
    train_literature_lora,
    train_literature_lora_on_errors,
)
from .predict.models import MODEL_LITERATURE_LORA, PredictConfig, run_predict

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
    """Years N to predict / eval / error-train before forecasting target_year."""
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


def run_train_literature_errors(
    config_path: str | Path | PredictConfig,
    target_year: int,
    *,
    pred_run_id: str | None = None,
    eval_k: int | None = None,
    run_id: str | None = None,
) -> int:
    """Fine-tune LoRA on ground-truth questions missed for target_year."""
    cfg = (
        load_config(config_path, PredictConfig)
        if isinstance(config_path, (str, Path))
        else config_path
    )
    db = Database(resolve_path(cfg.db_path))
    run_id = run_id or make_run_id("train_literature_errors")
    eval_k = eval_k or (max(cfg.top_k) if cfg.top_k else 50)

    with db.connect() as conn:
        n_examples = train_literature_lora_on_errors(
            conn,
            target_year,
            cfg.literature,
            pred_run_id=pred_run_id,
            model=MODEL_LITERATURE_LORA,
            eval_k=eval_k,
        )

    db.record_run(
        run_id,
        "train_literature_errors",
        notes=f"target_year={target_year}, examples={n_examples}",
    )
    logger.info(
        "Error-correction LoRA finished (target_year=%d, %d examples)",
        target_year,
        n_examples,
    )
    return n_examples


def _predict_year(cfg: PredictConfig, year: int, run_id: str) -> None:
    predict_cfg = PredictConfig(
        db_path=cfg.db_path,
        test_years=[year],
        top_k=cfg.top_k,
        models=[MODEL_LITERATURE_LORA],
        seed=cfg.seed,
        literature=cfg.literature,
    )
    run_predict(predict_cfg, run_id)


def _eval_year(cfg: PredictConfig, year: int, pred_run_id: str) -> None:
    from ..eval import EvalConfig, run_eval

    eval_cfg = EvalConfig(
        db_path=cfg.db_path,
        test_years=[year],
        top_k=cfg.top_k,
        models=[MODEL_LITERATURE_LORA],
        run_id=pred_run_id,
    )
    run_eval(eval_cfg, run_id=make_run_id("eval"))


def run_lora_forecast(
    config_path: str | Path | PredictConfig,
    target_year: int,
    *,
    calibrate_errors: bool = False,
    calibration_start: int | None = None,
    train_initial: bool = True,
    run_eval: bool = True,
    run_id: str | None = None,
) -> str:
    """
    Forecast target_year with optional error-train calibration.

    Simple (calibrate_errors=False):
      train on ≤ target_year−1 → predict target_year → optional eval

    Calibrated (calibrate_errors=True):
      train on ≤ first_cal−1
      for each calibration year N: predict N → eval N → error-train N
      predict target_year → optional eval
    """
    cfg = (
        load_config(config_path, PredictConfig)
        if isinstance(config_path, (str, Path))
        else config_path
    )
    db = Database(resolve_path(cfg.db_path))
    run_id = run_id or make_run_id("literature_forecast")
    eval_k = max(cfg.top_k) if cfg.top_k else 50

    with db.connect() as conn:
        available = available_question_years(conn)

    if not calibrate_errors:
        year_max = target_year - 1
        if train_initial:
            if not adapter_exists(cfg.literature, year_max):
                run_train_literature(cfg, year_max=year_max, run_id=run_id)
            else:
                logger.info("Reusing adapter year_max_%d", year_max)
        _predict_year(cfg, target_year, run_id)
        if run_eval:
            _eval_year(cfg, target_year, run_id)
        db.record_run(
            run_id,
            "literature_forecast",
            notes=f"target={target_year}, calibrated=0",
        )
        return run_id

    cal_years = calibration_years(available, target_year, calibration_start)
    if not cal_years:
        logger.warning(
            "No calibration years before %d — falling back to simple forecast",
            target_year,
        )
        return run_lora_forecast(
            cfg,
            target_year,
            calibrate_errors=False,
            train_initial=train_initial,
            run_eval=run_eval,
            run_id=run_id,
        )

    initial_year_max = min(cal_years) - 1
    if initial_year_max <= 0:
        logger.warning("initial_year_max=%d too small", initial_year_max)
        return run_id

    logger.info(
        "Calibrated forecast → %d: calibration years %s, initial year_max=%d",
        target_year,
        cal_years,
        initial_year_max,
    )

    if train_initial and not adapter_exists(cfg.literature, initial_year_max):
        run_train_literature(cfg, year_max=initial_year_max, run_id=run_id)
    elif train_initial:
        logger.info("Reusing adapter year_max_%d", initial_year_max)

    for N in cal_years:
        logger.info("Calibration: predict %d", N)
        _predict_year(cfg, N, run_id)
        _eval_year(cfg, N, run_id)
        logger.info("Calibration: error-train on %d", N)
        run_train_literature_errors(
            cfg,
            target_year=N,
            pred_run_id=run_id,
            eval_k=eval_k,
            run_id=run_id,
        )

    logger.info("Final forecast: predict %d", target_year)
    _predict_year(cfg, target_year, run_id)
    if run_eval:
        _eval_year(cfg, target_year, run_id)

    db.record_run(
        run_id,
        "literature_forecast",
        notes=f"target={target_year}, calibration={cal_years}",
    )
    logger.info("Literature forecast finished (run_id=%s)", run_id)
    return run_id


def check_literature_adapter(cfg: LiteratureLoraConfig, year_max: int) -> None:
    adapter = _adapter_path(cfg, year_max) / "lora"
    if not adapter.exists():
        raise RuntimeError(
            f"Literature LoRA adapter missing at {adapter}. "
            f"Run train_literature stage first (year_max={year_max})."
        )
