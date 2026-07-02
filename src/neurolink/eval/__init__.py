"""Layer 3 — prediction evaluation."""

from .matching import TfidfMatcher
from .metrics import EvalConfig, run_eval, write_summary
from .pipeline import EvalPipelineConfig, run_eval_layer

# Backward-compatible alias
_TfidfMatcher = TfidfMatcher

__all__ = [
    "EvalConfig",
    "EvalPipelineConfig",
    "TfidfMatcher",
    "_TfidfMatcher",
    "run_eval",
    "run_eval_layer",
    "write_summary",
]
