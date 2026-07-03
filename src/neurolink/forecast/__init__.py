"""Forecast: topics + predict (centroid or literature)."""

from .pipeline import (
    CENTROID_STAGES,
    LITERATURE_STAGES,
    ForecastPipelineConfig,
    check_centroid_ready,
    run_centroid_forecast,
    run_literature_forecast,
)
from .predict.models import (
    LLM_LITERATURE_MODELS,
    MODEL_BRAINGPT,
    MODEL_CENTROID_TRAJECTORY,
    MODEL_FREQUENCY,
    MODEL_LITERATURE_LORA,
    MODEL_MISTRAL_BASE,
    MODEL_RANDOM,
    PredictConfig,
    run_predict,
)
from .topics import TopicsConfig, compute_centroids, match_clusters_to_tracks, run_topics
from .benchmark import BENCHMARK_MODELS, resolve_benchmark, run_benchmark
from .train import (
    calibration_years,
    run_lora_forecast,
    run_train_literature,
    run_train_literature_errors,
)

__all__ = [
    "CENTROID_STAGES",
    "LITERATURE_STAGES",
    "ForecastPipelineConfig",
    "LLM_LITERATURE_MODELS",
    "MODEL_BRAINGPT",
    "MODEL_CENTROID_TRAJECTORY",
    "MODEL_FREQUENCY",
    "MODEL_LITERATURE_LORA",
    "MODEL_MISTRAL_BASE",
    "MODEL_RANDOM",
    "PredictConfig",
    "BENCHMARK_MODELS",
    "TopicsConfig",
    "calibration_years",
    "check_centroid_ready",
    "compute_centroids",
    "match_clusters_to_tracks",
    "resolve_benchmark",
    "run_benchmark",
    "run_centroid_forecast",
    "run_literature_forecast",
    "run_lora_forecast",
    "run_predict",
    "run_topics",
    "run_train_literature",
    "run_train_literature_errors",
]
