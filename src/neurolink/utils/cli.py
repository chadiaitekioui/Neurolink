"""CLI — python -m neurolink <command>."""

from __future__ import annotations

import argparse
import logging

from ..db import Database
from ..eval import run_eval
from ..forecast import (
    PredictConfig,
    run_centroid_forecast,
    run_literature_forecast,
    run_predict,
    run_rolling_literature,
    run_topics,
    run_train_literature,
    run_train_literature_errors,
)
from ..index import (
    CollectConfig,
    import_pubmed_text_file,
    run_collect,
    run_embed,
    run_impact,
    run_index,
    run_segment,
)
from ..menu import run_menu
from ..pipeline import run_pipeline
from .config import load_config, make_run_id, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_init_db(args: argparse.Namespace) -> None:
    db = Database(resolve_path(args.db))
    db.init_schema()
    logger.info("Database initialized: %s", db.path)


def cmd_import(args: argparse.Namespace) -> None:
    cfg = CollectConfig(db_path=args.db)
    import_pubmed_text_file(cfg, resolve_path(args.file), make_run_id("import"))


def cmd_collect(args: argparse.Namespace) -> None:
    run_collect(args.config)


def cmd_segment(args: argparse.Namespace) -> None:
    run_segment(args.config)


def cmd_impact(args: argparse.Namespace) -> None:
    run_impact(args.config)


def cmd_embed(args: argparse.Namespace) -> None:
    run_embed(args.config)


def cmd_index(args: argparse.Namespace) -> None:
    run_index(args.config)


def cmd_topics(args: argparse.Namespace) -> None:
    run_topics(args.config, target_year=args.year)


def cmd_predict(args: argparse.Namespace) -> None:
    run_predict(args.config)


def cmd_train_literature(args: argparse.Namespace) -> None:
    run_train_literature(args.config, year_max=args.year_max)


def cmd_train_literature_errors(args: argparse.Namespace) -> None:
    run_train_literature_errors(
        args.config,
        target_year=args.target_year,
        pred_run_id=args.pred_run_id,
        eval_k=args.eval_k,
    )


def cmd_rolling_literature(args: argparse.Namespace) -> None:
    run_rolling_literature(args.config, year_max_start=args.year_max_start)


def cmd_centroid(args: argparse.Namespace) -> None:
    run_centroid_forecast(args.config)


def cmd_literature(args: argparse.Namespace) -> None:
    run_literature_forecast(args.config)


def cmd_eval(args: argparse.Namespace) -> None:
    run_eval(args.config)


def cmd_run(args: argparse.Namespace) -> None:
    run_pipeline(args.config)


def cmd_menu(_args: argparse.Namespace) -> None:
    run_menu()


def cmd_status(args: argparse.Namespace) -> None:
    db = Database(resolve_path(args.db))
    with db.connect() as conn:
        tables = [
            "articles",
            "article_segments",
            "questions",
            "topic_centroid_snapshots",
            "topic_dynamics",
            "predictions",
            "evaluations",
        ]
        for t in tables:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"{t}: {n}")
            except Exception:
                print(f"{t}: (missing)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="neurolink", description="neurolink pipeline")
    p.add_argument("--db", default="data/neurolink.db")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Create SQLite schema").set_defaults(func=cmd_init_db)

    imp = sub.add_parser("import", help="Import a PubMed abstracts file")
    imp.add_argument("file", help="Path to abstracts.txt")
    imp.set_defaults(func=cmd_import)

    col = sub.add_parser("collect", help="Collect via PubMed API")
    col.add_argument("--config", default="config/index/collect.yaml")
    col.set_defaults(func=cmd_collect)

    seg = sub.add_parser("segment", help="Segment question / results")
    seg.add_argument("--config", default="config/index/segment.yaml")
    seg.set_defaults(func=cmd_segment)

    impc = sub.add_parser("impact", help="OpenAlex citations + impact labels")
    impc.add_argument("--config", default="config/index/impact.yaml")
    impc.set_defaults(func=cmd_impact)

    emb = sub.add_parser("embed", help="Embed research questions")
    emb.add_argument("--config", default="config/index/embed.yaml")
    emb.set_defaults(func=cmd_embed)

    idx = sub.add_parser("index", help="Run index layer (collect → embed)")
    idx.add_argument("--config", default="config/index/pipeline.yaml")
    idx.set_defaults(func=cmd_index)

    top = sub.add_parser("topics", help="Dynamic topics and emergence scores")
    top.add_argument("--config", default="config/forecast/topics.yaml")
    top.add_argument("--year", type=int, default=None)
    top.set_defaults(func=cmd_topics)

    pr = sub.add_parser("predict", help="Generate predictions")
    pr.add_argument("--config", default="config/forecast/predict.yaml")
    pr.set_defaults(func=cmd_predict)

    tr = sub.add_parser("train-literature", help="Train literature LoRA up to year_max")
    tr.add_argument("--config", default="config/forecast/predict_literature.yaml")
    tr.add_argument("--year-max", type=int, default=None)
    tr.set_defaults(func=cmd_train_literature)

    tre = sub.add_parser(
        "train-literature-errors",
        help="Fine-tune LoRA on missed ground-truth for a target year",
    )
    tre.add_argument("--config", default="config/forecast/predict_literature.yaml")
    tre.add_argument("--target-year", type=int, required=True)
    tre.add_argument("--pred-run-id", default=None)
    tre.add_argument("--eval-k", type=int, default=None)
    tre.set_defaults(func=cmd_train_literature_errors)

    roll = sub.add_parser(
        "rolling-literature",
        help="Rolling forecast: predict N, error-train on N, repeat",
    )
    roll.add_argument("--config", default="config/forecast/predict_literature.yaml")
    roll.add_argument("--year-max-start", type=int, default=None)
    roll.set_defaults(func=cmd_rolling_literature)

    cent = sub.add_parser("centroid", help="Run centroid forecast track")
    cent.add_argument("--config", default="config/forecast/pipeline_centroid.yaml")
    cent.set_defaults(func=cmd_centroid)

    lit = sub.add_parser("literature", help="Run literature forecast track")
    lit.add_argument("--config", default="config/forecast/pipeline_literature.yaml")
    lit.set_defaults(func=cmd_literature)

    ev = sub.add_parser("eval", help="Evaluate predictions")
    ev.add_argument("--config", default="config/eval/eval.yaml")
    ev.set_defaults(func=cmd_eval)

    rn = sub.add_parser("run", help="Run full or partial pipeline")
    rn.add_argument("--config", default="config/pipeline.yaml")
    rn.set_defaults(func=cmd_run)

    sub.add_parser("status", help="SQLite table row counts").set_defaults(func=cmd_status)
    sub.add_parser("menu", help="Interactive menu (index / literature / centroid)").set_defaults(func=cmd_menu)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
