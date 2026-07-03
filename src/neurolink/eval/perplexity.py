"""Perplexity and zlib–perplexity ratio (Luo et al. 2025, BrainBench Methods eq. 1 & 3)."""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass

from ..forecast.predict.llm_core import CausalLMConfig, sequence_perplexity

logger = logging.getLogger(__name__)


def zlib_compressed_length(text: str) -> int:
    """Bytes after zlib compression — data-agnostic compressibility proxy."""
    if not text.strip():
        return 0
    return len(zlib.compress(text.encode("utf-8")))


def zlib_perplexity_ratio(text: str, llm_cfg: CausalLMConfig) -> float | None:
    """ratio = zlib(X) / PPL(X). High values suggest LLM memorization (Carlini et al. 2021)."""
    ppl = sequence_perplexity(text, llm_cfg)
    if ppl <= 0:
        return None
    return zlib_compressed_length(text) / ppl


@dataclass
class ZlibPplStats:
    n: int
    mean_ratio: float
    median_ratio: float
    high_ratio_rate: float  # fraction above reference percentile


def collect_zlib_ppl_ratios(texts: list[str], llm_cfg: CausalLMConfig) -> list[float]:
    ratios: list[float] = []
    for text in texts:
        if not text.strip():
            continue
        try:
            ratio = zlib_perplexity_ratio(text, llm_cfg)
        except Exception as exc:
            logger.debug("zlib-PPL failed: %s", exc)
            continue
        if ratio is not None:
            ratios.append(ratio)
    return ratios


def summarize_zlib_ppl_ratios(
    texts: list[str],
    llm_cfg: CausalLMConfig,
    *,
    reference_ratios: list[float] | None = None,
    high_percentile: float = 90.0,
) -> ZlibPplStats | None:
    """Summarize zlib–perplexity ratios for a text set."""
    import numpy as np

    ratios = collect_zlib_ppl_ratios(texts, llm_cfg)
    if not ratios:
        return None

    ref = reference_ratios or ratios
    threshold = float(np.percentile(ref, high_percentile))
    arr = np.asarray(ratios, dtype=np.float64)
    return ZlibPplStats(
        n=len(ratios),
        mean_ratio=float(arr.mean()),
        median_ratio=float(np.median(arr)),
        high_ratio_rate=float((arr >= threshold).sum()) / len(arr),
    )
