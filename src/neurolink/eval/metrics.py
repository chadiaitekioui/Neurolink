"""Evaluation: P@k, R@k, semantic similarity, bootstrap."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..db import Database
from ..utils.config import infer_test_years, load_config, make_run_id, resolve_path
from .matching import TfidfMatcher

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    db_path: str = "data/neurolink.db"
    test_years: list[int] = field(default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023, 2024])
    top_k: list[int] = field(default_factory=lambda: [10, 50])
    models: list[str] = field(
        default_factory=lambda: ["random", "frequency", "literature_lora", "centroid_trajectory"]
    )
    semantic_threshold: float = 0.55
    critical_only: bool = True
    run_id: str | None = None


def run_eval(config_path: str | EvalConfig, run_id: str | None = None) -> int:
    cfg = load_config(config_path, EvalConfig) if isinstance(config_path, str) else config_path
    db = Database(resolve_path(cfg.db_path))
    eval_run = run_id or make_run_id("eval")
    rows_to_insert: list[tuple] = []

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
                for k in cfg.top_k:
                    p, r = matcher.precision_recall_at_k(preds, k)
                    rows_to_insert.append((N, model, "precision@k", k, p, eval_run))
                    rows_to_insert.append((N, model, "recall@k", k, r, eval_run))

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


def write_summary(db: Database, run_id: str, path) -> None:
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
    lines = ["# Pipeline evaluation\n", f"run_id: `{run_id}`\n\n", "| Year | Model | Metric | k | Value |\n", "|------|-------|--------|---|-------|\n"]
    for r in rows:
        lines.append(f"| {r['target_year']} | {r['model']} | {r['metric']} | {r['k']} | {r['value']:.3f} |\n")
    path.write_text("".join(lines), encoding="utf-8")
    logger.info("Report: %s", path)
