"""Layer 3 — prediction evaluation."""

from .brainbench import BrainBenchResult, paired_perplexity_accuracy, run_brainbench_year
from .contamination import ContaminationReport, run_contamination_audit
from .matching import TfidfMatcher
from .metrics import EvalConfig, run_eval, write_summary
from .perplexity import ZlibPplStats, zlib_perplexity_ratio
from .pipeline import EvalPipelineConfig, run_eval_layer

# Backward-compatible alias
_TfidfMatcher = TfidfMatcher

__all__ = [
    "BrainBenchResult",
    "ContaminationReport",
    "EvalConfig",
    "EvalPipelineConfig",
    "TfidfMatcher",
    "_TfidfMatcher",
    "ZlibPplStats",
    "paired_perplexity_accuracy",
    "run_brainbench_year",
    "run_contamination_audit",
    "run_eval",
    "run_eval_layer",
    "write_summary",
    "zlib_perplexity_ratio",
]
