"""Forecast: literature LoRA predict + benchmark.

Submodules are imported lazily via ``forecast.train``, ``forecast.benchmark``,
``forecast.predict.*`` to avoid circular imports with eval/index.
"""

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
from .train import calibration_years, run_train_literature

__all__ = [
    "LLM_LITERATURE_MODELS",
    "MODEL_BRAINGPT",
    "MODEL_FREQUENCY",
    "MODEL_LITERATURE_LORA",
    "MODEL_MISTRAL_BASE",
    "MODEL_RANDOM",
    "PredictConfig",
    "calibration_years",
    "run_benchmark",
    "run_predict",
    "run_train_literature",
    "resolve_benchmark",
    "BENCHMARK_MODELS",
]


def __getattr__(name: str):
    if name in {"run_benchmark", "resolve_benchmark", "BENCHMARK_MODELS"}:
        from . import benchmark as _benchmark

        return getattr(_benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
