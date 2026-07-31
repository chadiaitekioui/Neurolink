"""Evaluation: MiniLM P@k/R@k, stress/beyond-retrieval, BrainBench, LoRA contamination."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..db import Database
from ..utils.config import infer_test_years, load_config, make_run_id, resolve_path
from .brainbench import run_brainbench_year
from .contamination import run_contamination_audit
from .matching import make_matcher

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    db_path: str = "data/neurolink.db"
    test_years: list[int] = field(default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023, 2024])
    top_k: list[int] = field(default_factory=lambda: [10, 50])
    models: list[str] = field(
        default_factory=lambda: ["random", "frequency", "literature_lora"]
    )
    # MiniLM cosine threshold (as-is). TF-IDF ablation still applies offset inside TfidfMatcher.
    semantic_threshold: float = 0.50
    matcher_backend: str = "minilm"  # minilm | tfidf
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    critical_only: bool = True
    # Drop GT lines that look like metadata / abstract junk.
    filter_gt_noise: bool = True
    # If n_preds < min_pred_coverage * k, skip publishing P@k/R@k (still emit coverage + format).
    min_pred_coverage: float = 0.5
    run_id: str | None = None
    predict_config: str = "config/forecast/predict_literature.yaml"
    # Freeze literature_lora adapter at this year_max for all eval years (Job 2/3).
    # When set, BrainBench / contamination use year_max_{lora_year_max}/lora for every N.
    lora_year_max: int | None = None
    brainbench_enabled: bool = True
    contamination_enabled: bool = True
    brainbench_max_pairs: int = 50
    contamination_corpus_sample: int = 200
    # First-class stress / beyond-retrieval block (MiniLM offline; no LM regen).
    stress_enabled: bool = True
    stress_corpus_sample: int = 4000
    stress_near_threshold_band: float = 0.05
    stress_length_control_words: int = 12
    stress_max_context_questions: int = 30


def _llm_cfg_for_model(lit_cfg, year_max: int, model: str):
    from ..forecast.predict.literature_lora import resolve_literature_llm_cfg

    return resolve_literature_llm_cfg(lit_cfg, year_max, model)


def _append_metric(
    rows: list[tuple],
    *,
    year: int,
    model: str,
    metric: str,
    value: float,
    eval_run: str,
    k: int = 0,
) -> None:
    rows.append((year, model, metric, k, value, eval_run))


def run_eval(config_path: str | EvalConfig, run_id: str | None = None) -> int:
    from dataclasses import replace

    from ..forecast.predict.models import LLM_LITERATURE_MODELS, MODEL_LITERATURE_LORA, PredictConfig

    cfg = load_config(config_path, EvalConfig) if isinstance(config_path, str) else config_path
    db = Database(resolve_path(cfg.db_path))
    eval_run = run_id or make_run_id("eval")
    rows_to_insert: list[tuple] = []

    predict_cfg = load_config(cfg.predict_config, PredictConfig)
    lit_cfg = predict_cfg.literature
    if cfg.lora_year_max is not None:
        lit_cfg = replace(lit_cfg, benchmark_lora_year_max=cfg.lora_year_max)
        logger.info(
            "Eval: freezing literature_lora adapter at year_max=%d for all test years",
            cfg.lora_year_max,
        )
    elif lit_cfg.benchmark_lora_year_max is not None:
        logger.info(
            "Eval: using literature.benchmark_lora_year_max=%d from predict config",
            lit_cfg.benchmark_lora_year_max,
        )

    stress_cfg = None
    stress_baselines_cache: dict[tuple[int, int], dict[str, Any]] = {}
    if cfg.stress_enabled:
        from .stress import stress_config_from_eval

        stress_cfg = stress_config_from_eval(
            top_k=cfg.top_k,
            semantic_threshold=cfg.semantic_threshold,
            embed_model=cfg.embed_model,
            critical_only=cfg.critical_only,
            filter_gt_noise=cfg.filter_gt_noise,
            corpus_sample=cfg.stress_corpus_sample,
            near_threshold_band=cfg.stress_near_threshold_band,
            length_control_words=cfg.stress_length_control_words,
            max_context_questions=cfg.stress_max_context_questions,
        )
        logger.info(
            "Stress metrics enabled (corpus_sample=%d, beyond-retrieval + baselines)",
            cfg.stress_corpus_sample,
        )

    with db.connect() as conn:
        test_years = infer_test_years(conn, cfg.test_years)
        pred_run_id = cfg.run_id
        if not pred_run_id:
            pred_row = conn.execute(
                "SELECT run_id FROM runs WHERE stage='predict' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            pred_run_id = pred_row["run_id"] if pred_row else None

        from ..forecast.predict.direction_filter import format_compliance, is_clean_gt_text

        for model in cfg.models:
            for N in test_years:
                ref_rows = conn.execute(
                    "SELECT question_text, embedding, is_critical FROM questions WHERE year = ?",
                    (N,),
                ).fetchall()
                refs: list[dict] = []
                for r in ref_rows:
                    if cfg.critical_only and not r["is_critical"]:
                        continue
                    text = r["question_text"] or ""
                    if cfg.filter_gt_noise and not is_clean_gt_text(text):
                        continue
                    refs.append({"text": text})
                if not refs:
                    # Fallback: critical_only / noise filter emptied the pool.
                    refs = [{"text": r["question_text"]} for r in ref_rows]
                    if cfg.filter_gt_noise:
                        cleaned = [x for x in refs if is_clean_gt_text(x["text"] or "")]
                        if cleaned:
                            refs = cleaned

                pred_rows = conn.execute(
                    """
                    SELECT question_predicted FROM predictions
                    WHERE target_year=? AND model=? AND run_id=?
                    ORDER BY rank
                    """,
                    (N, model, pred_run_id),
                ).fetchall()
                preds = [r["question_predicted"] for r in pred_rows]
                if not preds or not refs:
                    continue

                fc = format_compliance(preds)
                _append_metric(
                    rows_to_insert, year=N, model=model, metric="format_frac_valid",
                    value=fc.frac_valid, eval_run=eval_run,
                )
                _append_metric(
                    rows_to_insert, year=N, model=model, metric="format_frac_noise",
                    value=fc.frac_noise, eval_run=eval_run,
                )
                _append_metric(
                    rows_to_insert, year=N, model=model, metric="format_frac_word_len_ok",
                    value=fc.frac_word_len_ok, eval_run=eval_run,
                )
                _append_metric(
                    rows_to_insert, year=N, model=model, metric="format_frac_with_doi",
                    value=fc.frac_with_doi, eval_run=eval_run,
                )
                _append_metric(
                    rows_to_insert, year=N, model=model, metric="format_mean_words",
                    value=fc.mean_words, eval_run=eval_run,
                )
                _append_metric(
                    rows_to_insert, year=N, model=model, metric="n_predictions",
                    value=float(len(preds)), eval_run=eval_run,
                )

                matcher = make_matcher(
                    [r["text"] for r in refs],
                    cfg.semantic_threshold,
                    backend=cfg.matcher_backend,
                    model_name=cfg.embed_model,
                )
                precision_by_k: dict[int, float] = {}
                for k in cfg.top_k:
                    coverage = min(1.0, len(preds) / float(k)) if k > 0 else 0.0
                    _append_metric(
                        rows_to_insert,
                        year=N,
                        model=model,
                        metric="pred_coverage@k",
                        value=coverage,
                        eval_run=eval_run,
                        k=k,
                    )
                    if coverage < cfg.min_pred_coverage:
                        logger.warning(
                            "Skip P@k/R@k for %s year=%d k=%d: pred_coverage=%.2f < %.2f "
                            "(n_preds=%d)",
                            model,
                            N,
                            k,
                            coverage,
                            cfg.min_pred_coverage,
                            len(preds),
                        )
                        _append_metric(
                            rows_to_insert,
                            year=N,
                            model=model,
                            metric="precision@k_skipped",
                            value=1.0,
                            eval_run=eval_run,
                            k=k,
                        )
                        continue

                    p, r, r_norm = matcher.precision_recall_at_k(preds, k)
                    precision_by_k[k] = p
                    _append_metric(rows_to_insert, year=N, model=model, metric="precision@k", value=p, eval_run=eval_run, k=k)
                    _append_metric(rows_to_insert, year=N, model=model, metric="recall@k", value=r, eval_run=eval_run, k=k)
                    _append_metric(
                        rows_to_insert,
                        year=N,
                        model=model,
                        metric="recall@k_normalized",
                        value=r_norm,
                        eval_run=eval_run,
                        k=k,
                    )

                if stress_cfg is not None and pred_run_id:
                    from .stress import evaluate_cell, flatten_stress_metrics

                    lit_stress = replace(
                        lit_cfg,
                        max_context_questions=stress_cfg.max_context_questions,
                    )
                    cell = evaluate_cell(
                        conn,
                        year=N,
                        model=model,
                        predict_run_id=pred_run_id,
                        cfg=stress_cfg,
                        lit_cfg=lit_stress,
                        baselines_cache=stress_baselines_cache,
                        preds=preds,
                        refs=[r["text"] for r in refs],
                    )
                    if cell:
                        n_stress = 0
                        for metric, k_s, val in flatten_stress_metrics(cell):
                            _append_metric(
                                rows_to_insert,
                                year=N,
                                model=model,
                                metric=metric,
                                value=val,
                                eval_run=eval_run,
                                k=k_s,
                            )
                            n_stress += 1
                        cond = (
                            (cell.get("beyond_retrieval") or {})
                            .get(str(max(cfg.top_k) if cfg.top_k else 50), {})
                            .get("corpus_minilm", {})
                            .get("conditional_beyond")
                        )
                        logger.info(
                            "Stress %s year=%d: %d metrics%s",
                            model,
                            N,
                            n_stress,
                            f", conditional_beyond@max_k={cond:.3f}" if cond is not None else "",
                        )

                if model not in LLM_LITERATURE_MODELS:
                    continue

                year_max = N - 1
                if lit_cfg.benchmark_lora_year_max is not None:
                    year_max = lit_cfg.benchmark_lora_year_max
                llm_cfg = _llm_cfg_for_model(lit_cfg, year_max, model)

                if cfg.brainbench_enabled:
                    bb = run_brainbench_year(
                        conn, N, lit_cfg, llm_cfg, max_pairs=cfg.brainbench_max_pairs
                    )
                    if bb:
                        _append_metric(
                            rows_to_insert, year=N, model=model, metric="brainbench_accuracy",
                            value=bb.accuracy, eval_run=eval_run,
                        )
                        _append_metric(
                            rows_to_insert, year=N, model=model, metric="brainbench_confidence",
                            value=bb.mean_confidence, eval_run=eval_run,
                        )
                        logger.info(
                            "BrainBench %d: accuracy=%.3f confidence=%.3f (n=%d)",
                            N, bb.accuracy, bb.mean_confidence, bb.n_pairs,
                        )

                if cfg.contamination_enabled:
                    report = run_contamination_audit(
                        conn,
                        target_year=N,
                        year_max=year_max,
                        predictions=preds,
                        lit_cfg=lit_cfg,
                        llm_cfg=llm_cfg,
                        semantic_threshold=cfg.semantic_threshold,
                        corpus_sample_size=cfg.contamination_corpus_sample,
                        include_train_overlap=(model == MODEL_LITERATURE_LORA),
                        matcher_backend=cfg.matcher_backend,
                        embed_model=cfg.embed_model,
                    )
                    if report:
                        _append_metric(
                            rows_to_insert, year=N, model=model, metric="contamination_corpus_recycling",
                            value=report.corpus_recycling_rate, eval_run=eval_run,
                        )
                        _append_metric(
                            rows_to_insert, year=N, model=model, metric="contamination_context_recycling",
                            value=report.context_recycling_rate, eval_run=eval_run,
                        )
                        _append_metric(
                            rows_to_insert, year=N, model=model, metric="contamination_context_verbatim_recycling",
                            value=report.context_verbatim_recycling_rate, eval_run=eval_run,
                        )
                        _append_metric(
                            rows_to_insert, year=N, model=model, metric="contamination_train_eval_overlap",
                            value=report.train_eval_overlap_rate, eval_run=eval_run,
                        )
                        _append_metric(
                            rows_to_insert, year=N, model=model, metric="contamination_verbatim_recycling",
                            value=report.verbatim_recycling_rate, eval_run=eval_run,
                        )
                        primary_k = max(cfg.top_k) if cfg.top_k else 50
                        p_primary = precision_by_k.get(primary_k)
                        if p_primary is not None:
                            extension_vs_context = p_primary - report.context_recycling_rate
                            extension_vs_corpus = p_primary - report.corpus_recycling_rate
                            _append_metric(
                                rows_to_insert,
                                year=N,
                                model=model,
                                metric="extension_vs_context",
                                value=extension_vs_context,
                                eval_run=eval_run,
                                k=primary_k,
                            )
                            _append_metric(
                                rows_to_insert,
                                year=N,
                                model=model,
                                metric="extension_vs_corpus",
                                value=extension_vs_corpus,
                                eval_run=eval_run,
                                k=primary_k,
                            )
                        if report.zlib_ppl_pred_mean is not None:
                            _append_metric(
                                rows_to_insert, year=N, model=model, metric="contamination_zlib_ppl_pred_mean",
                                value=report.zlib_ppl_pred_mean, eval_run=eval_run,
                            )
                        if report.zlib_ppl_pred_high_rate is not None:
                            _append_metric(
                                rows_to_insert, year=N, model=model, metric="contamination_zlib_ppl_pred_high_rate",
                                value=report.zlib_ppl_pred_high_rate, eval_run=eval_run,
                            )
                        if report.zlib_ppl_train_mean is not None:
                            _append_metric(
                                rows_to_insert, year=N, model=model, metric="contamination_zlib_ppl_train_mean",
                                value=report.zlib_ppl_train_mean, eval_run=eval_run,
                            )
                        logger.info(
                            "Contamination %d: corpus_recycling=%.3f context_recycling=%.3f "
                            "train_eval_overlap=%.3f verbatim=%.3f extension_vs_context=%s "
                            "zlib_ppl_pred_high=%s",
                            N,
                            report.corpus_recycling_rate,
                            report.context_recycling_rate,
                            report.train_eval_overlap_rate,
                            report.verbatim_recycling_rate,
                            f"{(p_primary - report.context_recycling_rate):.3f}"
                            if p_primary is not None
                            else "n/a",
                            f"{report.zlib_ppl_pred_high_rate:.3f}" if report.zlib_ppl_pred_high_rate is not None else "n/a",
                        )

    with db.connect() as conn:
        for N, model, metric, k, val, er in rows_to_insert:
            conn.execute(
                """
                INSERT INTO evaluations
                (target_year, model, metric, k, value, ci_low, ci_high, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (N, model, metric, k, val, None, None, er),
            )

    n = len(rows_to_insert)
    db.record_run(eval_run, "eval", notes=f"{n} metrics")
    write_summary(db, eval_run, resolve_path("eval/summary.md"))
    logger.info("Evaluation: %d metrics", n)
    return n


def write_summary(db: Database, run_id: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT target_year, model, metric, k, value
            FROM evaluations WHERE run_id = ?
            ORDER BY model, target_year, metric, k
            """,
            (run_id,),
        ).fetchall()
    lines = [
        "# Pipeline evaluation\n",
        f"run_id: `{run_id}`\n\n",
        "Metrics follow Luo et al. (2025) BrainBench where noted: "
        "paired perplexity discrimination (`brainbench_*`) and "
        "zlib–perplexity memorization ratios (`contamination_zlib_ppl_*`). "
        "P@k/R@k use MiniLM cosine by default (`matcher_backend: minilm`). "
        "`extension_vs_context` = P@k − context_recycling (legitimate extension proxy); "
        "`extension_vs_corpus` = P@k − corpus_recycling. "
        "`pred_coverage@k` = min(1, n_preds/k); if below `min_pred_coverage`, P@k/R@k are skipped "
        "(`precision@k_skipped=1`). "
        "`format_*` metrics track research-direction validity (noise, DOI, word length). "
        "GT noise can be dropped via `filter_gt_noise`. "
        "`recall@k_normalized` = R@k / min(k, N_GT) × N_GT = (# GT matchées) / min(k, N_GT) "
        "(fraction du plafond théorique k/N_GT, bornée par 1). "
        "Stress block (`stress_enabled`): `baseline_*_precision`, "
        "`beyond_conditional_corpus_minilm` (speaking success scale), "
        "`beyond_incremental_*`, `stress_novelty`, diversity / near-threshold / length-control.\n\n",
        "| Year | Model | Metric | k | Value |\n",
        "|------|-------|--------|---|-------|\n",
    ]
    for r in rows:
        k_display = r["k"] if r["k"] else "—"
        lines.append(f"| {r['target_year']} | {r['model']} | {r['metric']} | {k_display} | {r['value']:.3f} |\n")
    path.write_text("".join(lines), encoding="utf-8")
    logger.info("Report: %s", path)
