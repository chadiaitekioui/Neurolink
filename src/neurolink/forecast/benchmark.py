"""LLM benchmark: compare literature_lora, mistral_base, braingpt on post-training years."""

from __future__ import annotations

from dataclasses import replace

from ..db import Database
from ..utils.config import resolve_path
from .predict.literature_lora import (
    LiteratureLoraConfig,
    infer_benchmark_years,
    list_saved_lora_year_max,
)
from .predict.models import (
    MODEL_BRAINGPT,
    MODEL_LITERATURE_LORA,
    MODEL_MISTRAL_BASE,
    PredictConfig,
)

BENCHMARK_MODELS = [MODEL_LITERATURE_LORA, MODEL_MISTRAL_BASE, MODEL_BRAINGPT]


def resolve_benchmark(
    cfg: PredictConfig,
    *,
    lora_year_max: int | None = None,
    db_path: str | None = None,
) -> tuple[PredictConfig, int, list[int]]:
    """
    Build a compare/benchmark PredictConfig.

    Uses a fixed LoRA adapter (year_max=T) and forecast years with questions where year > T.
    """
    db_path = db_path or cfg.db_path
    lit = cfg.literature

    saved = list_saved_lora_year_max(lit)
    if not saved:
        raise RuntimeError(
            "No saved LoRA adapters found. Train one first (menu LoRA or train-literature)."
        )

    anchor = lora_year_max if lora_year_max is not None else saved[-1]
    if anchor not in saved:
        raise RuntimeError(
            f"LoRA year_max_{anchor} not found. Available: {', '.join(map(str, saved))}"
        )

    db = Database(resolve_path(db_path))
    with db.connect() as conn:
        available = infer_benchmark_years(conn, anchor)

    if not available:
        raise RuntimeError(
            f"No forecast years after year_max={anchor} in the database. "
            "Index more literature or pick another adapter."
        )

    # Respect explicit test_years from config when set (e.g. Job 2: 2023–2025 only).
    if cfg.test_years:
        allowed = set(available)
        test_years = [y for y in cfg.test_years if y in allowed]
        if not test_years:
            raise RuntimeError(
                f"Configured test_years={cfg.test_years} have no overlap with "
                f"available years after year_max={anchor}: {available}"
            )
    else:
        test_years = available

    literature = replace(lit, benchmark_lora_year_max=anchor)
    benchmark_cfg = replace(
        cfg,
        db_path=db_path,
        test_years=test_years,
        models=list(cfg.models) if cfg.models else list(BENCHMARK_MODELS),
        literature=literature,
    )
    return benchmark_cfg, anchor, test_years


def run_benchmark(
    config_path: str | PredictConfig,
    *,
    lora_year_max: int | None = None,
    run_id: str | None = None,
) -> tuple[str, int, list[int]]:
    """Resolve benchmark years from saved LoRA and run predict for all three LLMs."""
    from ..utils.config import load_config, make_run_id
    from .predict.models import run_predict

    cfg = load_config(config_path, PredictConfig) if isinstance(config_path, str) else config_path
    benchmark_cfg, anchor, test_years = resolve_benchmark(cfg, lora_year_max=lora_year_max)
    run = run_id or make_run_id("benchmark")
    run_predict(benchmark_cfg, run)
    return run, anchor, test_years
