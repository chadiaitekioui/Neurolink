"""Layer 3 — prediction evaluation."""

from .brainbench import BrainBenchResult, paired_perplexity_accuracy, run_brainbench_year
from .contamination import ContaminationReport, run_contamination_audit
from .matching import EmbeddingMatcher, TfidfMatcher, make_matcher
from .metrics import EvalConfig, run_eval, write_summary
from .perplexity import ZlibPplStats, zlib_perplexity_ratio
from .pipeline import EvalPipelineConfig, run_eval_layer
from .stress import StressConfig, evaluate_cell, flatten_stress_metrics

__all__ = [
    "BrainBenchResult",
    "ContaminationReport",
    "EmbeddingMatcher",
    "EvalConfig",
    "EvalPipelineConfig",
    "StressConfig",
    "TfidfMatcher",
    "ZlibPplStats",
    "evaluate_cell",
    "flatten_stress_metrics",
    "make_matcher",
    "paired_perplexity_accuracy",
    "run_brainbench_year",
    "run_contamination_audit",
    "run_eval",
    "run_eval_layer",
    "write_summary",
    "zlib_perplexity_ratio",
]
