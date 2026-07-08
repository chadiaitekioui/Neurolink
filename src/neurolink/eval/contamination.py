"""LoRA contamination audit vs indexed corpus (BrainBench memorization framework)."""

from __future__ import annotations

import logging
import random
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..forecast.predict.llm_core import CausalLMConfig
from .matching import TfidfMatcher
from .perplexity import collect_zlib_ppl_ratios, summarize_zlib_ppl_ratios

if TYPE_CHECKING:
    from ..forecast.predict.literature_lora import LiteratureLoraConfig

logger = logging.getLogger(__name__)


@dataclass
class ContaminationReport:
    corpus_recycling_rate: float
    context_recycling_rate: float
    context_verbatim_recycling_rate: float
    train_eval_overlap_rate: float
    verbatim_recycling_rate: float
    zlib_ppl_train_mean: float | None
    zlib_ppl_pred_mean: float | None
    zlib_ppl_corpus_mean: float | None
    zlib_ppl_pred_high_rate: float | None
    n_predictions: int
    n_train_completions: int
    n_context_questions: int


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _corpus_questions_before_year(conn: sqlite3.Connection, year: int) -> list[str]:
    rows = conn.execute(
        "SELECT question_text FROM questions WHERE year IS NOT NULL AND year < ?",
        (year,),
    ).fetchall()
    return [(r["question_text"] or "").strip() for r in rows if (r["question_text"] or "").strip()]


def _verbatim_rate(predictions: list[str], corpus: list[str]) -> float:
    if not predictions:
        return 0.0
    corpus_norm = {_normalize(c) for c in corpus}
    hits = sum(1 for p in predictions if _normalize(p) in corpus_norm)
    return hits / len(predictions)


def _semantic_recycling_rate(
    predictions: list[str],
    corpus: list[str],
    threshold: float,
) -> float:
    if not predictions or not corpus:
        return 0.0
    matcher = TfidfMatcher(corpus, threshold)
    sim_threshold = matcher._sim_threshold
    recycled = 0
    for pred in predictions:
        sims = matcher.similarity_matrix([pred])
        if sims is not None and float(sims.max()) >= sim_threshold:
            recycled += 1
    return recycled / len(predictions)


def _train_completions(
    conn: sqlite3.Connection,
    year_max: int,
    lit_cfg: LiteratureLoraConfig,
) -> list[str]:
    from ..forecast.predict.literature_lora import build_temporal_examples

    examples = build_temporal_examples(conn, year_max, lit_cfg)
    return [completion for _, completion in examples]


def _eval_questions(conn: sqlite3.Connection, year: int) -> list[str]:
    rows = conn.execute(
        "SELECT question_text FROM questions WHERE year = ?",
        (year,),
    ).fetchall()
    return [(r["question_text"] or "").strip() for r in rows if (r["question_text"] or "").strip()]


def _train_eval_overlap(train_completions: list[str], eval_questions: list[str], threshold: float) -> float:
    if not eval_questions or not train_completions:
        return 0.0
    matcher = TfidfMatcher(train_completions, threshold)
    sim_threshold = matcher._sim_threshold
    matched = 0
    for q in eval_questions:
        sims = matcher.similarity_matrix([q])
        if sims is not None and float(sims.max()) >= sim_threshold:
            matched += 1
    return matched / len(eval_questions)


def run_contamination_audit(
    conn: sqlite3.Connection,
    *,
    target_year: int,
    year_max: int,
    predictions: list[str],
    lit_cfg: LiteratureLoraConfig,
    llm_cfg: CausalLMConfig,
    semantic_threshold: float = 0.55,
    corpus_sample_size: int = 200,
    seed: int = 42,
    include_train_overlap: bool = True,
) -> ContaminationReport | None:
    """Measure LoRA contamination relative to the indexed question corpus."""
    if not predictions:
        return None

    corpus = _corpus_questions_before_year(conn, target_year)
    train_completions = (
        _train_completions(conn, year_max, lit_cfg) if include_train_overlap else []
    )
    eval_questions = _eval_questions(conn, target_year)

    from ..forecast.predict.literature_lora import list_context_questions

    context_questions = list_context_questions(conn, target_year, lit_cfg)

    rng = random.Random(seed)
    corpus_sample = corpus if len(corpus) <= corpus_sample_size else rng.sample(corpus, corpus_sample_size)

    ref_stats = summarize_zlib_ppl_ratios(corpus_sample, llm_cfg)
    corpus_ratios = collect_zlib_ppl_ratios(corpus_sample, llm_cfg)
    train_stats = summarize_zlib_ppl_ratios(
        train_completions, llm_cfg, reference_ratios=corpus_ratios or None
    )
    pred_stats = summarize_zlib_ppl_ratios(
        predictions, llm_cfg, reference_ratios=corpus_ratios or None
    )

    return ContaminationReport(
        corpus_recycling_rate=_semantic_recycling_rate(predictions, corpus, semantic_threshold),
        context_recycling_rate=_semantic_recycling_rate(
            predictions, context_questions, semantic_threshold
        ),
        context_verbatim_recycling_rate=_verbatim_rate(predictions, context_questions),
        train_eval_overlap_rate=(
            _train_eval_overlap(train_completions, eval_questions, semantic_threshold)
            if include_train_overlap
            else 0.0
        ),
        verbatim_recycling_rate=_verbatim_rate(predictions, corpus),
        zlib_ppl_train_mean=train_stats.mean_ratio if train_stats else None,
        zlib_ppl_pred_mean=pred_stats.mean_ratio if pred_stats else None,
        zlib_ppl_corpus_mean=ref_stats.mean_ratio if ref_stats else None,
        zlib_ppl_pred_high_rate=pred_stats.high_ratio_rate if pred_stats else None,
        n_predictions=len(predictions),
        n_train_completions=len(train_completions),
        n_context_questions=len(context_questions),
    )