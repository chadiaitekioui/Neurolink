"""Evaluation: TF-IDF P@k/R@k, BrainBench perplexity discrimination, LoRA contamination."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..db import Database
from ..utils.config import infer_test_years, load_config, make_run_id, resolve_path
from .brainbench import run_brainbench_year
from .contamination import run_contamination_audit
from .matching import TfidfMatcher

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    db_path: str = "data/neurolink.db"
    test_years: list[int] = field(default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023, 2024])
    top_k: list[int] = field(default_factory=lambda: [10, 50])
    models: list[str] = field(
        default_factory=lambda: ["random", "frequency", "literature_lora"]
    )
    semantic_threshold: float = 0.55
    critical_only: bool = True
    run_id: str | None = None
    predict_config: str = "config/forecast/predict_literature.yaml"
    # Freeze literature_lora adapter at this year_max for all eval years (Job 2/3).
    # When set, BrainBench / contamination use year_max_{lora_year_max}/lora for every N.
    lora_year_max: int | None = None
    brainbench_enabled: bool = True
    contamination_enabled: bool = True
    brainbench_max_pairs: int = 50
    contamination_corpus_sample: int = 200


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

    with db.connect() as conn:
        test_years = infer_test_years(conn, cfg.test_years)
        pred_row = cfg.run_id or conn.execute(
            "SELECT run_id FROM runs WHERE stage='predict' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        pred_run_id = pred_row["run_id"] if pred_row else None

        for model in cfg.models:
            for N in test_years:
                ref_rows = conn.execute(
                    "SELECT question_text, embedding, is_critical FROM questions WHERE year = ?",
                    (N,),
                ).fetchall()
                refs = []
                for r in ref_rows:
                    if cfg.critical_only and not r["is_critical"]:
                        continue
                    refs.append({"text": r["question_text"]})
                if not refs:
                    refs = [{"text": r["question_text"]} for r in ref_rows]

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

                matcher = TfidfMatcher([r["text"] for r in refs], cfg.semantic_threshold)
                precision_by_k: dict[int, float] = {}
                for k in cfg.top_k:
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
                        p_primary = precision_by_k.get(primary_k, 0.0)
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
                            "train_eval_overlap=%.3f verbatim=%.3f extension_vs_context=%.3f "
                            "zlib_ppl_pred_high=%s",
                            N,
                            report.corpus_recycling_rate,
                            report.context_recycling_rate,
                            report.train_eval_overlap_rate,
                            report.verbatim_recycling_rate,
                            extension_vs_context,
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
        "`extension_vs_context` = P@k − context_recycling (legitimate extension proxy); "
        "`extension_vs_corpus` = P@k − corpus_recycling. "
        "`recall@k_normalized` = R@k / min(k, N_GT) × N_GT = (# GT matchées) / min(k, N_GT) "
        "(fraction du plafond théorique k/N_GT, bornée par 1).\n\n",
        "| Year | Model | Metric | k | Value |\n",
        "|------|-------|--------|---|-------|\n",
    ]
    for r in rows:
        k_display = r["k"] if r["k"] else "—"
        lines.append(f"| {r['target_year']} | {r['model']} | {r['metric']} | {k_display} | {r['value']:.3f} |\n")
    path.write_text("".join(lines), encoding="utf-8")
    logger.info("Report: %s", path)
