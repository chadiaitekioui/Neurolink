"""Temporal prediction — literature LLMs + baselines."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ...db import Database
from ...utils.config import infer_test_years, load_config, make_run_id, resolve_path
from .literature_lora import LiteratureLoraConfig, make_literature_predictor
from .llm_core import release_gpu_memory

logger = logging.getLogger(__name__)

MODEL_RANDOM = "random"
MODEL_FREQUENCY = "frequency"
MODEL_LITERATURE_LORA = "literature_lora"
MODEL_MISTRAL_BASE = "mistral_base"
MODEL_BRAINGPT = "braingpt"

LLM_LITERATURE_MODELS = frozenset({
    MODEL_LITERATURE_LORA,
    MODEL_MISTRAL_BASE,
    MODEL_BRAINGPT,
})


@dataclass
class PredictConfig:
    db_path: str = "data/neurolink.db"
    test_years: list[int] = field(default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023, 2024])
    top_k: list[int] = field(default_factory=lambda: [10, 50])
    models: list[str] = field(
        default_factory=lambda: [
            MODEL_RANDOM,
            MODEL_FREQUENCY,
            MODEL_LITERATURE_LORA,
        ]
    )
    seed: int = 42
    literature: LiteratureLoraConfig = field(default_factory=LiteratureLoraConfig)


def _context_questions(conn, year_max: int) -> list[str]:
    rows = conn.execute(
        "SELECT question_text FROM questions WHERE year <= ?",
        (year_max,),
    ).fetchall()
    return [r["question_text"] for r in rows]


def predict_random(candidates: list[str], k: int, rng: random.Random) -> list[tuple[str, float]]:
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    return [(t, 0.0) for t in shuffled[:k]]


def predict_frequency(conn, N: int, k: int) -> list[tuple[str, float]]:
    rows = conn.execute(
        """
        SELECT question_text, COUNT(*) as c FROM questions
        WHERE year = ? GROUP BY question_text ORDER BY c DESC LIMIT ?
        """,
        (N - 1, k),
    ).fetchall()
    return [(r["question_text"], float(r["c"])) for r in rows]


def _build_predictors(cfg: PredictConfig) -> dict:
    return {
        MODEL_RANDOM: lambda conn, N, k, rng: predict_random(
            _context_questions(conn, N - 1), k, rng
        ),
        MODEL_FREQUENCY: lambda conn, N, k, rng: predict_frequency(conn, N, k),
        MODEL_LITERATURE_LORA: make_literature_predictor(cfg.literature, MODEL_LITERATURE_LORA),
        MODEL_MISTRAL_BASE: make_literature_predictor(cfg.literature, MODEL_MISTRAL_BASE),
        MODEL_BRAINGPT: make_literature_predictor(cfg.literature, MODEL_BRAINGPT),
    }


def run_predict(config_path: str | PredictConfig, run_id: str | None = None) -> int:
    cfg = (
        load_config(config_path, PredictConfig)
        if isinstance(config_path, str)
        else config_path
    )
    db = Database(resolve_path(cfg.db_path))
    run_id = run_id or make_run_id("predict")
    rng = random.Random(cfg.seed)
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    max_k = max(cfg.top_k) if cfg.top_k else 50
    predictors = _build_predictors(cfg)

    with db.connect() as conn:
        test_years = infer_test_years(conn, cfg.test_years)
        if not test_years:
            logger.warning("No test years available")
            return 0

        for model in cfg.models:
            if model not in predictors:
                logger.warning("Unknown model: %s", model)
                continue
            model_total = 0
            for N in test_years:
                preds = predictors[model](conn, N, max_k, rng)
                model_total += len(preds)
                conn.execute(
                    "DELETE FROM predictions WHERE target_year=? AND model=? AND run_id=?",
                    (N, model, run_id),
                )
                for rank, (text, score) in enumerate(preds, start=1):
                    conn.execute(
                        """
                        INSERT INTO predictions
                        (target_year, model, rank, question_predicted, score, run_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (N, model, rank, text, score, run_id, now),
                    )
                    n += 1
            if model in LLM_LITERATURE_MODELS:
                logger.info(
                    "Predict model=%s: %d rows across years %s (max_k=%d) — see GEN_AUDIT lines",
                    model,
                    model_total,
                    test_years,
                    max_k,
                )
                release_gpu_memory()

    db.record_run(run_id, "predict", notes=f"{n} predictions")
    logger.info("Predictions: %d rows (max_k=%d)", n, max_k)
    return n
