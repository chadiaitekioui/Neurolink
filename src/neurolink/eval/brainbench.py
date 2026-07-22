"""BrainBench-style paired perplexity discrimination (Luo et al. 2025, Methods)."""

from __future__ import annotations

import logging
import random
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..forecast.predict.llm_core import CausalLMConfig, sequence_perplexity

if TYPE_CHECKING:
    from ..forecast.predict.literature_lora import LiteratureLoraConfig

logger = logging.getLogger(__name__)

BRAINBENCH_PREFIX = (
    "You are a neuroscientist with deep knowledge in neuroscience. "
    "Here is a research direction from a neuroscience publication: "
)


@dataclass
class BrainBenchResult:
    accuracy: float
    mean_confidence: float
    n_pairs: int


def _distractor_pool(conn: sqlite3.Connection, target_year: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT question_text FROM questions
        WHERE year IS NOT NULL AND year != ? AND year < ?
        ORDER BY COALESCE(impact_score, 0) DESC
        """,
        (target_year, target_year),
    ).fetchall()
    return [(r["question_text"] or "").strip() for r in rows if (r["question_text"] or "").strip()]


def _format_passage(question: str) -> str:
    return f"{BRAINBENCH_PREFIX}{question.strip()}"


def paired_perplexity_accuracy(
    real_texts: list[str],
    distractor_texts: list[str],
    llm_cfg: CausalLMConfig,
    *,
    max_pairs: int,
    seed: int = 42,
) -> BrainBenchResult | None:
    """
    BrainBench decision rule (eq. 2): choose passage with lower perplexity.
    Accuracy = fraction where PPL(real) < PPL(distractor).
    Confidence = |PPL(distractor) - PPL(real)|.
    """
    if not real_texts or not distractor_texts:
        return None

    rng = random.Random(seed)
    n_pairs = min(max_pairs, len(real_texts))
    sampled_reals = rng.sample(real_texts, n_pairs) if len(real_texts) > n_pairs else list(real_texts)

    correct = 0
    confidences: list[float] = []
    for real in sampled_reals:
        distractor = rng.choice(distractor_texts)
        try:
            ppl_real = sequence_perplexity(_format_passage(real), llm_cfg)
            ppl_alt = sequence_perplexity(_format_passage(distractor), llm_cfg)
        except Exception as exc:
            logger.debug("BrainBench PPL failed: %s", exc)
            continue
        if ppl_real < ppl_alt:
            correct += 1
        confidences.append(abs(ppl_alt - ppl_real))

    if not confidences:
        return None

    return BrainBenchResult(
        accuracy=correct / len(confidences),
        mean_confidence=sum(confidences) / len(confidences),
        n_pairs=len(confidences),
    )


def run_brainbench_year(
    conn: sqlite3.Connection,
    target_year: int,
    lit_cfg: LiteratureLoraConfig,
    llm_cfg: CausalLMConfig,
    *,
    max_pairs: int = 50,
    seed: int = 42,
) -> BrainBenchResult | None:
    """Discriminate real year-N questions from distractors drawn from prior years."""
    rows = conn.execute(
        """
        SELECT question_text FROM questions
        WHERE year = ?
        ORDER BY COALESCE(impact_score, 0) DESC
        """,
        (target_year,),
    ).fetchall()
    real = [(r["question_text"] or "").strip() for r in rows if (r["question_text"] or "").strip()]
    distractors = _distractor_pool(conn, target_year)
    if not real or not distractors:
        logger.warning("BrainBench year %d: insufficient questions (real=%d, alt=%d)", target_year, len(real), len(distractors))
        return None

    return paired_perplexity_accuracy(real, distractors, llm_cfg, max_pairs=max_pairs, seed=seed)
