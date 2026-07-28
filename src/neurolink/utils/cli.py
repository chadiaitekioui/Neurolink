"""CLI — python -m neurolink <command>."""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from ..db import Database
from ..eval import run_eval
from ..forecast import (
    PredictConfig,
    run_literature_forecast,
    run_predict,
    run_train_literature,
    run_train_literature_errors,
)
from ..index import (
    CollectConfig,
    SegmentConfig,
    import_pubmed_text_file,
    run_collect,
    run_embed,
    run_impact,
    run_index,
    run_segment,
)
from ..menu import run_menu
from ..pipeline import run_pipeline
from ..workflow import CompleteWorkflowConfig, run_complete_workflow
from .config import load_config, make_run_id, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _segment_config_for_cli(config_path: str, device: str | None = None) -> SegmentConfig:
    """Load segment config; CLI defaults to auto CUDA detection unless overridden."""
    cfg = load_config(config_path, SegmentConfig)
    if device is not None:
        return replace(cfg, device=device)
    return replace(cfg, device="auto")


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
    device = "cpu" if args.cpu else args.device
    pmids = None
    if args.pmids_file:
        from ..index.segment import load_pmids_file

        pmids = load_pmids_file(args.pmids_file)
        if not pmids:
            raise SystemExit(f"No PMIDs in {args.pmids_file}")
    run_segment(
        _segment_config_for_cli(args.config, device=device),
        limit=args.limit,
        pmids=pmids,
        force=args.force,
    )


def cmd_impact(args: argparse.Namespace) -> None:
    run_impact(args.config)


def cmd_embed(args: argparse.Namespace) -> None:
    run_embed(args.config)


def cmd_index(args: argparse.Namespace) -> None:
    from ..index.pipeline import IndexPipelineConfig

    device = "cpu" if args.cpu else args.device
    cfg = load_config(args.config, IndexPipelineConfig)
    if "segment" in cfg.stages:
        segment_cfg = _segment_config_for_cli(cfg.segment_config, device=device)
        run_index(replace(cfg, segment_config=segment_cfg))
    else:
        run_index(cfg)


def cmd_predict(args: argparse.Namespace) -> None:
    run_predict(args.config)


def cmd_train_literature(args: argparse.Namespace) -> None:
    run_train_literature(
        args.config,
        year_max=args.year_max,
        skip_if_exists=args.skip_if_exists,
    )


def cmd_train_literature_errors(args: argparse.Namespace) -> None:
    run_train_literature_errors(
        args.config,
        target_year=args.target_year,
        pred_run_id=args.pred_run_id,
        eval_k=args.eval_k,
    )


def cmd_literature(args: argparse.Namespace) -> None:
    run_literature_forecast(args.config)


def cmd_compare(args: argparse.Namespace) -> None:
    """Run LLM benchmark (literature_lora vs mistral_base vs braingpt)."""
    from ..forecast import run_benchmark

    run_id, anchor, years = run_benchmark(
        args.config,
        lora_year_max=args.lora_year_max,
    )
    logger.info(
        "Benchmark finished run_id=%s anchor_year_max=%d years=%s",
        run_id,
        anchor,
        years,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    run_eval(args.config)


def cmd_run(args: argparse.Namespace) -> None:
    run_pipeline(args.config)


def cmd_workflow(args: argparse.Namespace) -> None:
    cfg = CompleteWorkflowConfig(
        skip_index=args.skip_index,
        segment_device="cpu" if args.cpu else args.device,
        lora_anchor_first=args.lora_first,
        lora_anchor_second=args.lora_second,
        forecast_year=args.forecast_year,
    )
    run_complete_workflow(cfg)


def cmd_menu(_args: argparse.Namespace) -> None:
    run_menu()


def cmd_status(args: argparse.Namespace) -> None:
    from ..index.pipeline import get_index_counts

    db = Database(resolve_path(args.db))
    with db.connect(readonly=True) as conn:
        tables = [
            "articles",
            "article_segments",
            "citations",
            "article_impact",
            "questions",
            "predictions",
            "evaluations",
        ]
        for t in tables:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"{t}: {n}")
            except Exception:
                print(f"{t}: (missing)")
    counts = get_index_counts(args.db)
    print(
        f"segments_missing: {counts.segments_missing}\n"
        f"questions_unembedded: {counts.questions_unembedded}\n"
        f"index_ready: {counts.ready}"
    )


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
    seg.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Torch device (default: auto-detect CUDA)",
    )
    seg.add_argument("--cpu", action="store_true", help="Force CPU (overrides --device)")
    seg.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max articles to segment (missing segments first, ordered by pmid)",
    )
    seg.add_argument(
        "--pmids-file",
        default=None,
        help="JSON or text file of PMIDs to segment (optional with --force to re-run)",
    )
    seg.add_argument(
        "--force",
        action="store_true",
        help="Re-segment PMIDs even if article_segments already exists",
    )
    seg.set_defaults(func=cmd_segment)

    impc = sub.add_parser("impact", help="OpenAlex citations + impact labels")
    impc.add_argument("--config", default="config/index/impact.yaml")
    impc.set_defaults(func=cmd_impact)

    emb = sub.add_parser("embed", help="Embed research questions")
    emb.add_argument("--config", default="config/index/embed.yaml")
    emb.set_defaults(func=cmd_embed)

    idx = sub.add_parser("index", help="Run index layer (collect → embed)")
    idx.add_argument("--config", default="config/index/pipeline.yaml")
    idx.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Torch device for segment stage (default: auto-detect CUDA)",
    )
    idx.add_argument("--cpu", action="store_true", help="Force CPU for segment (overrides --device)")
    idx.set_defaults(func=cmd_index)

    pr = sub.add_parser("predict", help="Generate predictions")
    pr.add_argument("--config", default="config/forecast/predict.yaml")
    pr.set_defaults(func=cmd_predict)

    tr = sub.add_parser("train-literature", help="Train literature LoRA up to year_max")
    tr.add_argument("--config", default="config/forecast/predict_literature.yaml")
    tr.add_argument("--year-max", type=int, default=None)
    tr.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip training when data/models/literature/year_max_{Y}/lora/ already exists",
    )
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

    lit = sub.add_parser("literature", help="Run literature forecast track")
    lit.add_argument("--config", default="config/forecast/pipeline_literature.yaml")
    lit.set_defaults(func=cmd_literature)

    cmp = sub.add_parser(
        "compare",
        help="Benchmark literature_lora, mistral_base, braingpt after a saved LoRA year_max",
    )
    cmp.add_argument("--config", default="config/forecast/predict_compare.yaml")
    cmp.add_argument(
        "--lora-year-max",
        type=int,
        default=None,
        help="LoRA anchor year_max (default: latest saved adapter)",
    )
    cmp.set_defaults(func=cmd_compare)

    wf = sub.add_parser(
        "workflow",
        help="Complete benchmark workflow: index → LoRA×2 → benchmark×2 → forecast (GPU)",
    )
    wf.add_argument("--skip-index", action="store_true", help="Skip index (already built)")
    wf.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Torch device for segment stage",
    )
    wf.add_argument("--cpu", action="store_true", help="Force CPU for segment")
    wf.add_argument("--lora-first", type=int, default=2022, help="First LoRA year_max")
    wf.add_argument("--lora-second", type=int, default=2025, help="Second LoRA year_max")
    wf.add_argument("--forecast-year", type=int, default=2027, help="Final forecast year")
    wf.set_defaults(func=cmd_workflow)

    ev = sub.add_parser("eval", help="Evaluate predictions")
    ev.add_argument("--config", default="config/eval/eval.yaml")
    ev.set_defaults(func=cmd_eval)

    rn = sub.add_parser("run", help="Run full or partial pipeline")
    rn.add_argument("--config", default="config/pipeline.yaml")
    rn.set_defaults(func=cmd_run)

    sub.add_parser("status", help="SQLite table row counts").set_defaults(func=cmd_status)
    sub.add_parser("menu", help="Interactive menu (index / LoRA / benchmark)").set_defaults(func=cmd_menu)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
