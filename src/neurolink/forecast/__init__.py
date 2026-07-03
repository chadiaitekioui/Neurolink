"""Forecast: literature LoRA predict + benchmark."""

from .benchmark import BENCHMARK_MODELS, resolve_benchmark, run_benchmark
from .pipeline import LITERATURE_STAGES, ForecastPipelineConfig, run_literature_forecast
from .predict.models import (
    LLM_LITERATURE_MODELS,
    MODEL_BRAINGPT,
    MODEL_FREQUENCY,
    MODEL_LITERATURE_LORA,
    MODEL_MISTRAL_BASE,
    MODEL_RANDOM,
    PredictConfig,
    run_predict,
)
from .train import (
    calibration_years,
    run_lora_forecast,
    run_train_literature,
    run_train_literature_errors,
)

__all__ = [
    "BENCHMARK_MODELS",
    "ForecastPipelineConfig",
    "LITERATURE_STAGES",
    "LLM_LITERATURE_MODELS",
    "MODEL_BRAINGPT",
    "MODEL_FREQUENCY",
    "MODEL_LITERATURE_LORA",
    "MODEL_MISTRAL_BASE",
    "MODEL_RANDOM",
    "PredictConfig",
    "calibration_years",
    "resolve_benchmark",
    "run_benchmark",
    "run_literature_forecast",
    "run_lora_forecast",
    "run_predict",
    "run_train_literature",
    "run_train_literature_errors",
]
