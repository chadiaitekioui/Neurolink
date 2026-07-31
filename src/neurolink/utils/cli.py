"""CLI — python -m neurolink <command>.

Production path mirrors the README cluster jobs. Local stage commands remain
for scripts invoked inside sbatch / login shells.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from ..cluster_jobs import JOB_ORDER, list_jobs, submit_job
from ..db import Database
from ..eval import run_eval
from ..forecast import run_predict, run_train_literature
from ..index import (
    DirectionConfig,
    run_collect,
    run_directions,
    run_embed,
    run_impact,
    run_index,
)
from ..menu import run_menu
from .config import load_config, resolve_path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _direction_config_for_cli(config_path: str, device: str | None = None) -> DirectionConfig:
    cfg = load_config(config_path, DirectionConfig)
    if device is not None:
        return replace(cfg, device=device)
    return replace(cfg, device="auto")


def cmd_init_db(args: argparse.Namespace) -> None:
    db = Database(resolve_path(args.db))
    db.init_schema()
    logger.info("Database initialized: %s", db.path)


def cmd_collect(args: argparse.Namespace) -> None:
    run_collect(args.config)


def cmd_direction(args: argparse.Namespace) -> None:
    device = "cpu" if args.cpu else args.device
    pmids = None
    if args.pmids_file:
        from ..index.subject import load_pmids_file

        pmids = load_pmids_file(args.pmids_file)
        if not pmids:
            raise SystemExit(f"No PMIDs in {args.pmids_file}")
    run_directions(
        _direction_config_for_cli(args.config, device=device),
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
    cfg = (
        load_config(args.config, IndexPipelineConfig)
        if args.config
        else IndexPipelineConfig()
    )
    if "direction" in cfg.stages:
        direction_cfg = _direction_config_for_cli(cfg.direction_config, device=device)
        from .config import make_run_id

        run_id = make_run_id("index")
        if "collect" in cfg.stages:
            run_collect(cfg.collect_config)
        run_directions(direction_cfg, run_id)
        if "impact" in cfg.stages:
            run_impact(cfg.impact_config, run_id)
        if "embed" in cfg.stages:
            run_embed(cfg.embed_config, run_id)
    else:
        run_index(cfg)


def cmd_predict(args: argparse.Namespace) -> None:
    run_predict(args.config)


def cmd_train(args: argparse.Namespace) -> None:
    run_train_literature(
        args.config,
        year_max=args.year_max,
        skip_if_exists=args.skip_if_exists,
    )


def cmd_compare(args: argparse.Namespace) -> None:
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


def cmd_jobs(_args: argparse.Namespace) -> None:
    for spec in list_jobs():
        print(f"{spec.key:16} [{spec.kind:6}]  {spec.title}")
        print(f"{'':16}  {spec.description}")
        print(f"{'':16}  script: {spec.script}")


def cmd_submit(args: argparse.Namespace) -> None:
    try:
        code = submit_job(
            args.job,
            dry_run=args.dry_run,
            account=args.account,
            time=args.time,
            qos=args.qos,
            constraint=args.constraint,
            job_name=args.job_name,
            extra=args.sbatch_arg or None,
        )
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    if code != 0:
        raise SystemExit(code)


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
        f"directions_missing: {counts.directions_missing}\n"
        f"questions_unembedded: {counts.questions_unembedded}\n"
        f"index_ready: {counts.ready}"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="neurolink",
        description=(
            "Neurolink: index PubMed → LoRA train → LLM benchmark. "
            "Use `submit` / `menu` for cluster jobs (see README)."
        ),
    )
    p.add_argument("--db", default="data/neurolink.db")
    sub = p.add_subparsers(dest="command", required=True)

    # --- Cluster ---
    sub.add_parser("menu", help="Interactive launcher for cluster jobs").set_defaults(
        func=cmd_menu
    )
    sub.add_parser("jobs", help="List cluster jobs").set_defaults(func=cmd_jobs)

    sj = sub.add_parser(
        "submit",
        help="Launch a cluster job (sbatch or login bash script)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Jobs: " + ", ".join(JOB_ORDER),
    )
    sj.add_argument("job", choices=list(JOB_ORDER), help="Job key (see `neurolink jobs`)")
    sj.add_argument("--dry-run", action="store_true", help="Print command only")
    sj.add_argument("--account", default=None, help="Override #SBATCH --account")
    sj.add_argument("--time", default=None, help="Override #SBATCH --time (HH:MM:SS)")
    sj.add_argument("--qos", default=None, help="Override #SBATCH --qos")
    sj.add_argument(
        "--constraint",
        default=None,
        help="Override #SBATCH -C / --constraint (e.g. v100-32g)",
    )
    sj.add_argument("--job-name", default=None, help="Override #SBATCH --job-name")
    sj.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Extra sbatch flag (repeatable), e.g. --sbatch-arg=--nice=10000",
    )
    sj.set_defaults(func=cmd_submit)

    # --- Local / in-job stages ---
    sub.add_parser("init-db", help="Create SQLite schema").set_defaults(func=cmd_init_db)

    col = sub.add_parser("collect", help="Collect via PubMed API")
    col.add_argument("--config", default="config/index/collect.yaml")
    col.set_defaults(func=cmd_collect)

    direction = sub.add_parser(
        "direction", help="LLM research-direction extraction (Mistral-Instruct)"
    )
    direction.add_argument("--config", default="config/index/direction.yaml")
    direction.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Torch device (default: auto-detect CUDA)",
    )
    direction.add_argument("--cpu", action="store_true", help="Force CPU")
    direction.add_argument("--limit", type=int, default=None, help="Max articles to process")
    direction.add_argument("--pmids-file", default=None, help="JSON/text PMID list")
    direction.add_argument("--force", action="store_true", help="Re-extract even if present")
    direction.set_defaults(func=cmd_direction)

    impc = sub.add_parser("impact", help="OpenAlex citations + rebuild questions")
    impc.add_argument("--config", default="config/index/impact.yaml")
    impc.set_defaults(func=cmd_impact)

    emb = sub.add_parser("embed", help="Embed research directions (MiniLM)")
    emb.add_argument("--config", default="config/index/embed.yaml")
    emb.set_defaults(func=cmd_embed)

    idx = sub.add_parser("index", help="Local index pipeline (collect → embed)")
    idx.add_argument(
        "--config",
        default=None,
        help="Optional IndexPipelineConfig YAML (defaults: config/index/*.yaml stages)",
    )
    idx.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    idx.add_argument("--cpu", action="store_true")
    idx.set_defaults(func=cmd_index)

    tr = sub.add_parser("train", help="Train literature LoRA up to year_max")
    tr.add_argument("--config", default="config/forecast/train_lora_base.yaml")
    tr.add_argument("--year-max", type=int, default=None)
    tr.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip when adapter already exists",
    )
    tr.set_defaults(func=cmd_train)
    # Alias kept for existing slurm scripts
    tr_alias = sub.add_parser("train-literature", help=argparse.SUPPRESS)
    tr_alias.add_argument("--config", default="config/forecast/train_lora_base.yaml")
    tr_alias.add_argument("--year-max", type=int, default=None)
    tr_alias.add_argument("--skip-if-exists", action="store_true")
    tr_alias.set_defaults(func=cmd_train)

    cmp = sub.add_parser(
        "compare",
        help="Benchmark literature_lora vs mistral_base vs braingpt",
    )
    cmp.add_argument("--config", default="config/forecast/predict_compare.yaml")
    cmp.add_argument("--lora-year-max", type=int, default=None)
    cmp.set_defaults(func=cmd_compare)

    pr = sub.add_parser("predict", help="Generate predictions")
    pr.add_argument("--config", default="config/forecast/predict_compare.yaml")
    pr.set_defaults(func=cmd_predict)

    ev = sub.add_parser("eval", help="Evaluate predictions (incl. stress metrics)")
    ev.add_argument("--config", default="config/eval/scenarios/eval_compare_2022.yaml")
    ev.set_defaults(func=cmd_eval)

    sub.add_parser("status", help="SQLite table row counts").set_defaults(func=cmd_status)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
