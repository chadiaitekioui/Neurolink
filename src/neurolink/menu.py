"""Interactive terminal menu"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, replace

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text
import shutil

from .db import Database
from .eval import EvalConfig, run_eval
from .forecast import (
    MODEL_CENTROID_TRAJECTORY,
    MODEL_LITERATURE_LORA,
    PredictConfig,
    calibration_years,
    run_lora_forecast,
    run_predict,
    run_topics,
)
from .utils.hf_auth import hf_token_available, set_hf_token
from .index import (
    CollectConfig,
    EmbedConfig,
    ImpactConfig,
    IndexPipelineConfig,
    SegmentConfig,
    check_index_ready,
    run_collect,
    run_embed,
    run_impact,
    run_index,
    run_segment,
)
from .utils.config import available_question_years, load_config, make_run_id, resolve_path

logger = logging.getLogger(__name__)

console = Console()
PREDICT_LITERATURE_CONFIG = "config/forecast/predict_literature.yaml"
PREDICT_CENTROID_CONFIG = "config/forecast/predict_centroid.yaml"
TOPICS_CONFIG = "config/forecast/topics.yaml"
EVAL_CENTROID_CONFIG = "config/eval/eval_centroid.yaml"
EVAL_LITERATURE_CONFIG = "config/eval/eval_literature.yaml"
INDEX_PIPELINE_CONFIG = "config/index/pipeline.yaml"
COLLECT_CONFIG = "config/index/collect.yaml"
SEGMENT_CONFIG = "config/index/segment.yaml"
IMPACT_CONFIG = "config/index/impact.yaml"
EMBED_CONFIG = "config/index/embed.yaml"

BANNER = r"""
 _   _ _____ _   _ ____   ___  _     ___ _   _ _  __
| \ | | ____| | | |  _ \ / _ \| |   |_ _| \ | | |/ /
|  \| |  _| | | | | |_) | | | | |    | ||  \| | ' /
| |\  | |___| |_| |  _ <| |_| | |___ | || |\  | . \
|_| \_|_____|\___/|_| \_\\___/|_____|___|_| \_|_|\_\
"""



def _banner() -> None:
    console_width = shutil.get_terminal_size().columns
    for line in BANNER.splitlines():
        console.print(Text.from_markup(line.center(console_width), style="bold white"))
    console.print(
        Panel(
            """Modular pipeline to forecast emergent neuroscience research directions from PubMed literature using two approaches:\n
            1 - Centroid trajectory: neuroscience topics are centroids; research questions are points in their clusters.\n
            2 - Literature LoRA: fine-tune Mistral-7B on temporal question pairs.\n""",
            border_style="green",
        )
    )


def _parse_years(raw: str, default: list[int]) -> list[int]:
    raw = raw.strip()
    if not raw:
        return default
    try:
        return [int(y.strip()) for y in raw.split(",") if y.strip()]
    except ValueError:
        console.print("[yellow]Invalid format — using default years.[/yellow]")
        return default


def _show_params_table(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAVY, border_style="green")
    table.add_column("Parameter", style="bold")
    table.add_column("Value")
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)


def _common_predict_params(cfg: PredictConfig) -> PredictConfig:
    console.print("\n[bold]Common parameters[/bold]")
    db = Prompt.ask("SQLite database", default=cfg.db_path)
    years_raw = Prompt.ask(
        "Test years (comma-separated)",
        default=",".join(str(y) for y in cfg.test_years),
    )
    top_k = IntPrompt.ask("Max top-k predictions", default=max(cfg.top_k) if cfg.top_k else 50)
    test_years = _parse_years(years_raw, cfg.test_years)
    return replace(cfg, db_path=db, test_years=test_years, top_k=[top_k])


@dataclass
class LiteratureSession:
    cfg: PredictConfig
    target_year: int
    train_first: bool
    calibrate_errors: bool
    calibration_start: int | None

    @property
    def year_max(self) -> int:
        return self.target_year - 1


def _ensure_hf_token() -> bool:
    """Check HF token; prompt once if missing (needed for Mistral-7B download)."""
    if hf_token_available():
        return True
    console.print(
        Panel(
            "No Hugging Face token found (HF_TOKEN / huggingface-cli login).\n"
            "Mistral-7B is a gated model — accept the license at\n"
            "https://huggingface.co/mistralai/Mistral-7B-v0.1 then paste your token.",
            title="Hugging Face",
            border_style="yellow",
        )
    )
    if not Confirm.ask("Enter a Hugging Face token now?", default=True):
        return False
    token = Prompt.ask("HF token", password=True)
    if not token.strip():
        console.print("[red]Token required for LoRA train/predict.[/red]")
        return False
    set_hf_token(token)
    console.print("[green]Token set for this session (HF_TOKEN).[/green]")
    return True


def _configure_literature(cfg: PredictConfig) -> LiteratureSession | None:
    console.print("\n[bold magenta]LoRA[/bold magenta]")
    console.print(
        "[dim]Forecast target year N using literature through N−1 "
        "(optional: calibrate with error-train on each prior year).[/dim]"
    )
    db_path = Prompt.ask("SQLite database", default=cfg.db_path)
    default_year = max(cfg.test_years) if cfg.test_years else 2020
    target_year = IntPrompt.ask("Target year to predict", default=default_year)
    if target_year <= 1:
        console.print("[red]Target year must be ≥ 2 (need at least one prior year).[/red]")
        return None

    calibrate_errors = False
    calibration_start: int | None = None
    if target_year > 2:
        calibrate_errors = Confirm.ask(
            f"Calibrate with error-train until {target_year - 1}? "
            "(each year: predict → eval → correct errors)",
            default=False,
        )
        if calibrate_errors:
            db = Database(resolve_path(db_path))
            with db.connect() as conn:
                available = available_question_years(conn)
            cal_years = calibration_years(available, target_year)
            if len(cal_years) < 1:
                console.print(
                    "[yellow]Not enough years in DB for calibration — simple mode.[/yellow]"
                )
                calibrate_errors = False
            else:
                default_start = cal_years[0]
                calibration_start = IntPrompt.ask(
                    "First calibration year",
                    default=default_start,
                )
                cal_years = calibration_years(available, target_year, calibration_start)
                console.print(
                    f"[dim]Calibration years: {', '.join(map(str, cal_years))} "
                    f"→ then forecast {target_year}[/dim]"
                )
    else:
        console.print(
            f"[dim]Simple mode: train on ≤ {target_year - 1}, predict {target_year}.[/dim]"
        )

    top_k = IntPrompt.ask("Max top-k predictions", default=max(cfg.top_k) if cfg.top_k else 50)
    train_first = Confirm.ask("Train LoRA before predict?", default=True)
    max_ctx = IntPrompt.ask("Context questions", default=cfg.literature.max_context_questions)
    temp = FloatPrompt.ask("Generation temperature", default=cfg.literature.llm.temperature)
    tokens = IntPrompt.ask("Max tokens per question", default=cfg.literature.llm.max_new_tokens)
    use_4bit = Confirm.ask("4-bit quantization?", default=cfg.literature.use_4bit)

    literature = replace(
        cfg.literature,
        max_context_questions=max_ctx,
        use_4bit=use_4bit,
        backend="lora",
        llm=replace(cfg.literature.llm, temperature=temp, max_new_tokens=tokens),
    )
    cfg = replace(
        cfg,
        db_path=db_path,
        test_years=[target_year],
        top_k=[top_k],
        literature=literature,
        models=[MODEL_LITERATURE_LORA],
    )

    return LiteratureSession(
        cfg=cfg,
        target_year=target_year,
        train_first=train_first,
        calibrate_errors=calibrate_errors,
        calibration_start=calibration_start,
    )


def _confirm_literature(session: LiteratureSession) -> bool:
    cfg = session.cfg
    cal_label = "—"
    if session.calibrate_errors:
        db = Database(resolve_path(cfg.db_path))
        with db.connect() as conn:
            years = calibration_years(
                available_question_years(conn),
                session.target_year,
                session.calibration_start,
            )
        cal_label = ", ".join(map(str, years)) + f" → {session.target_year}"

    _show_params_table(
        "literature_lora summary",
        [
            ("Model", MODEL_LITERATURE_LORA),
            ("Database", cfg.db_path),
            ("Target year", str(session.target_year)),
            ("Error-train calibration", str(session.calibrate_errors)),
            ("Calibration path", cal_label),
            ("Literature through (simple)", str(session.year_max)),
            ("Top-k", str(max(cfg.top_k))),
            ("Context questions", str(cfg.literature.max_context_questions)),
            ("Temperature", f"{cfg.literature.llm.temperature:.2f}"),
            ("Train before predict", str(session.train_first)),
        ],
    )
    return Confirm.ask("\nRun LoRA?", default=True)


def _configure_centroid(cfg: PredictConfig) -> PredictConfig:
    cfg = _common_predict_params(cfg)
    console.print("\n[bold magenta]Centroid trajectory[/bold magenta]")
    top_tracks = IntPrompt.ask("Number of tracks", default=cfg.centroid.top_tracks)
    temp = FloatPrompt.ask("Generation temperature", default=cfg.centroid.llm.temperature)
    tokens = IntPrompt.ask("Max tokens per question", default=cfg.centroid.llm.max_new_tokens)
    use_4bit = Confirm.ask("4-bit quantization?", default=cfg.centroid.llm.use_4bit)

    centroid = replace(
        cfg.centroid,
        top_tracks=top_tracks,
        llm=replace(cfg.centroid.llm, temperature=temp, max_new_tokens=tokens, use_4bit=use_4bit),
    )
    return replace(
        cfg,
        centroid=centroid,
        models=[MODEL_CENTROID_TRAJECTORY],
    )


def _confirm_centroid(cfg: PredictConfig, run_topics: bool) -> bool:
    _show_params_table(
        "centroid_trajectory summary",
        [
            ("Model", MODEL_CENTROID_TRAJECTORY),
            ("Database", cfg.db_path),
            ("Test years", ", ".join(map(str, cfg.test_years))),
            ("Top-k", str(max(cfg.top_k))),
            ("Tracks", str(cfg.centroid.top_tracks)),
            ("Temperature", f"{cfg.centroid.llm.temperature:.2f}"),
            ("Run topics first", str(run_topics)),
        ],
    )
    return Confirm.ask("\nRun Centroid trajectory?", default=True)


def _optional_str(prompt: str, default: str | None) -> str | None:
    raw = Prompt.ask(prompt, default=default or "")
    return raw.strip() or None


@dataclass
class IndexSession:
    db_path: str
    collect: CollectConfig
    segment: SegmentConfig
    impact: ImpactConfig
    embed: EmbedConfig


def _load_index_session(db_path: str | None = None) -> IndexSession:
    base = load_config(INDEX_PIPELINE_CONFIG, IndexPipelineConfig)
    path = db_path or base.db_path
    return IndexSession(
        db_path=path,
        collect=_with_db(path, load_config(COLLECT_CONFIG, CollectConfig)),
        segment=_with_db(path, load_config(SEGMENT_CONFIG, SegmentConfig)),
        impact=_with_db(path, load_config(IMPACT_CONFIG, ImpactConfig)),
        embed=_with_db(path, load_config(EMBED_CONFIG, EmbedConfig)),
    )


def _sync_index_db(session: IndexSession) -> IndexSession:
    return IndexSession(
        db_path=session.db_path,
        collect=_with_db(session.db_path, session.collect),
        segment=_with_db(session.db_path, session.segment),
        impact=_with_db(session.db_path, session.impact),
        embed=_with_db(session.db_path, session.embed),
    )


def _index_status(db_path: str) -> tuple[bool, str]:
    path = resolve_path(db_path)
    if not path.exists():
        return False, "database file does not exist — run Init database first"
    try:
        check_index_ready(db_path)
        return True, "ready (questions + embeddings)"
    except RuntimeError as e:
        return False, str(e)
    except Exception as e:
        return False, f"cannot read database: {e}"


def _ensure_index_or_continue(db_path: str) -> bool:
    ready, msg = _index_status(db_path)
    if ready:
        return True
    console.print(f"[yellow]Index not ready:[/yellow] {msg}")
    if Confirm.ask("Open index menu to build the database?", default=True):
        _run_index_menu(db_path)
        ready, msg = _index_status(db_path)
        if ready:
            return True
        console.print(f"[yellow]Index still not ready:[/yellow] {msg}")
    return Confirm.ask("Continue forecast anyway?", default=False)


def _with_db(db_path: str, cfg):
    return replace(cfg, db_path=db_path)


def _init_database(db_path: str) -> None:
    db = Database(resolve_path(db_path))
    db.init_schema()
    console.print(f"[green]Schema initialized:[/green] {db_path}")


def _configure_collect(cfg: CollectConfig) -> CollectConfig:
    console.print("\n[bold]Collect (PubMed API)[/bold]")
    mesh = Prompt.ask("MeSH term", default=cfg.mesh)
    term = _optional_str("Custom PubMed query (empty = MeSH)", cfg.term)
    year_from = IntPrompt.ask("Year from", default=cfg.year_from)
    year_to = IntPrompt.ask("Year to", default=cfg.year_to)
    exclude_reviews = Confirm.ask("Exclude reviews?", default=cfg.exclude_reviews)
    retmax = IntPrompt.ask("Max articles (retmax)", default=cfg.retmax)
    batch_size = IntPrompt.ask("Batch size (efetch)", default=cfg.batch_size)
    delay = FloatPrompt.ask("Delay between API calls (s)", default=cfg.delay_seconds)
    email = _optional_str("NCBI contact email (optional)", cfg.email)
    return replace(
        cfg,
        mesh=mesh,
        term=term,
        year_from=year_from,
        year_to=year_to,
        exclude_reviews=exclude_reviews,
        retmax=retmax,
        batch_size=batch_size,
        delay_seconds=delay,
        email=email,
    )


def _configure_segment(cfg: SegmentConfig) -> SegmentConfig:
    console.print("\n[bold]Segment[/bold] (rules structure + PubMedBERT)")
    model = Prompt.ask("PubMedBERT model", default=cfg.pubmedbert_model)
    return replace(cfg, pubmedbert_model=model)


def _configure_impact(cfg: ImpactConfig) -> ImpactConfig:
    console.print("\n[bold]Impact[/bold]")
    percentile = FloatPrompt.ask("Critical percentile", default=cfg.critical_percentile)
    delay = FloatPrompt.ask("Delay between OpenAlex calls (s)", default=cfg.delay_seconds)
    mailto = _optional_str("OpenAlex mailto (optional)", cfg.mailto)
    return replace(
        cfg,
        critical_percentile=percentile,
        delay_seconds=delay,
        mailto=mailto,
    )


def _configure_embed(cfg: EmbedConfig) -> EmbedConfig:
    console.print("\n[bold]Embed[/bold]")
    model = Prompt.ask("Embedding model", default=cfg.model_name)
    batch = IntPrompt.ask("Batch size", default=cfg.batch_size)
    force_tfidf = Confirm.ask(
        "Force TF-IDF (skip sentence-transformers)?",
        default=cfg.fallback == "tfidf",
    )
    return replace(
        cfg,
        model_name=model,
        batch_size=batch,
        fallback="tfidf" if force_tfidf else "model",
    )


def _show_collect_summary(cfg: CollectConfig) -> None:
    _show_params_table(
        "collect",
        [
            ("Database", cfg.db_path),
            ("MeSH", cfg.mesh),
            ("Custom query", cfg.term or "(MeSH)"),
            ("Years", f"{cfg.year_from}–{cfg.year_to}"),
            ("Exclude reviews", str(cfg.exclude_reviews)),
            ("retmax", str(cfg.retmax)),
            ("batch_size", str(cfg.batch_size)),
            ("delay_seconds", f"{cfg.delay_seconds:.2f}"),
            ("email", cfg.email or "—"),
        ],
    )


def _show_segment_summary(cfg: SegmentConfig) -> None:
    _show_params_table(
        "segment",
        [
            ("Database", cfg.db_path),
            ("Method", "rules + PubMedBERT"),
            ("PubMedBERT model", cfg.pubmedbert_model),
        ],
    )


def _show_impact_summary(cfg: ImpactConfig) -> None:
    _show_params_table(
        "impact",
        [
            ("Database", cfg.db_path),
            ("critical_percentile", f"{cfg.critical_percentile:.2f}"),
            ("delay_seconds", f"{cfg.delay_seconds:.2f}"),
            ("mailto", cfg.mailto or "—"),
        ],
    )


def _show_embed_summary(cfg: EmbedConfig) -> None:
    _show_params_table(
        "embed",
        [
            ("Database", cfg.db_path),
            ("model_name", cfg.model_name),
            ("batch_size", str(cfg.batch_size)),
            ("fallback", "tfidf" if cfg.fallback == "tfidf" else "sentence-transformers"),
        ],
    )


def _configure_index_params(session: IndexSession) -> IndexSession:
    console.print("\n[bold]Configure index parameters[/bold]")
    console.print(
        "  [cyan]1[/cyan] Collect   [cyan]2[/cyan] Segment   "
        "[cyan]3[/cyan] Impact   [cyan]4[/cyan] Embed   [cyan]5[/cyan] All   [cyan]0[/cyan] Back"
    )
    choice = Prompt.ask(
        "Stage to configure",
        choices=["1", "2", "3", "4", "5", "0"],
        default="5",
        show_choices=False,
    )
    if choice == "0":
        return session
    if choice in ("1", "5"):
        session = replace(session, collect=_configure_collect(session.collect))
    if choice in ("2", "5"):
        session = replace(session, segment=_configure_segment(session.segment))
    if choice in ("3", "5"):
        session = replace(session, impact=_configure_impact(session.impact))
    if choice in ("4", "5"):
        session = replace(session, embed=_configure_embed(session.embed))
    return session


def _run_index_stage(label: str, fn) -> None:
    try:
        with console.status(f"[bold green]{label}…", spinner="dots"):
            n = fn()
        console.print(Panel(f"[bold green]✓ {label}[/bold green] ({n} rows)", border_style="green"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red] {e}", border_style="red"))
        logger.exception("%s failed", label)


def _run_full_index(session: IndexSession) -> None:
    _show_collect_summary(session.collect)
    _show_segment_summary(session.segment)
    _show_impact_summary(session.impact)
    _show_embed_summary(session.embed)
    if not Confirm.ask("\nRun full index pipeline (collect → segment → impact → embed)?", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    cfg = replace(
        load_config(INDEX_PIPELINE_CONFIG, IndexPipelineConfig),
        db_path=session.db_path,
        collect_config=session.collect,
        segment_config=session.segment,
        impact_config=session.impact,
        embed_config=session.embed,
        stages=["collect", "segment", "impact", "embed"],
    )
    try:
        with console.status("[bold green]Running index pipeline…", spinner="dots"):
            run_index(cfg)
        console.print(Panel("[bold green]✓ Index pipeline done[/bold green]", border_style="green"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red] {e}", border_style="red"))
        logger.exception("Index pipeline failed")


def _run_index_step_menu(session: IndexSession) -> IndexSession:
    while True:
        console.print(
            "\n[bold]Index step[/bold] — "
            "[cyan]1[/cyan] Collect   [cyan]2[/cyan] Segment   "
            "[cyan]3[/cyan] Impact   [cyan]4[/cyan] Embed   [cyan]0[/cyan] Back"
        )
        choice = Prompt.ask(
            "[bold]Step[/bold]",
            choices=["1", "2", "3", "4", "0"],
            default="1",
            show_choices=False,
        )
        if choice == "0":
            return session
        if choice == "1":
            if Confirm.ask("Edit collect parameters first?", default=False):
                session = replace(session, collect=_configure_collect(session.collect))
            _show_collect_summary(session.collect)
            if Confirm.ask("Run collect?", default=True):
                _run_index_stage("Collect", lambda: run_collect(session.collect))
            continue
        if choice == "2":
            if Confirm.ask("Edit segment parameters first?", default=False):
                session = replace(session, segment=_configure_segment(session.segment))
            _show_segment_summary(session.segment)
            if Confirm.ask("Run segment?", default=True):
                _run_index_stage("Segment", lambda: run_segment(session.segment))
            continue
        if choice == "3":
            if Confirm.ask("Edit impact parameters first?", default=False):
                session = replace(session, impact=_configure_impact(session.impact))
            _show_impact_summary(session.impact)
            if Confirm.ask("Run impact?", default=True):
                _run_index_stage("Impact", lambda: run_impact(session.impact))
            continue
        if choice == "4":
            if Confirm.ask("Edit embed parameters first?", default=False):
                session = replace(session, embed=_configure_embed(session.embed))
            _show_embed_summary(session.embed)
            if Confirm.ask("Run embed?", default=True):
                _run_index_stage("Embed", lambda: run_embed(session.embed))


def _run_index_menu(initial_db: str | None = None) -> IndexSession:
    session = _load_index_session(initial_db)
    if initial_db is None:
        session = replace(
            session,
            db_path=Prompt.ask("SQLite database", default=session.db_path),
        )
        session = _sync_index_db(session)

    while True:
        ready, status_msg = _index_status(session.db_path)
        status_style = "green" if ready else "yellow"
        console.print(f"\n[bold]Index[/bold] — [dim]{session.db_path}[/dim]")
        console.print(f"  Status: [{status_style}]{status_msg}[/{status_style}]")
        console.print(
            "  [cyan]1[/cyan] Initialize DB   [cyan]2[/cyan] Run a step   "
            "[cyan]3[/cyan] Full pipeline   [cyan]4[/cyan] Change database   "
            "[cyan]0[/cyan] Back"
        )
        choice = Prompt.ask(
            "[bold]Index action[/bold]",
            choices=["1", "2", "3", "4", "0"],
            default="3",
            show_choices=False,
        )

        if choice == "0":
            return session
        if choice == "4":
            session = replace(
                session,
                db_path=Prompt.ask("SQLite database", default=session.db_path),
            )
            session = _sync_index_db(session)
            continue
        if choice == "1":
            _init_database(session.db_path)
            continue
        if choice == "2":
            session = _run_index_step_menu(session)
            continue
        if choice == "3":
            if Confirm.ask("Edit parameters before full pipeline?", default=True):
                session = _configure_index_params(session)
            _run_full_index(session)


def _run_literature(session: LiteratureSession) -> None:
    cfg = session.cfg
    if not _ensure_index_or_continue(cfg.db_path):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    if not _ensure_hf_token():
        console.print("[yellow]Cancelled — Hugging Face token required.[/yellow]")
        return
    if not _confirm_literature(session):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    run_eval_after = Confirm.ask(
        "Run evaluation on final forecast?"
        if session.calibrate_errors
        else "Run evaluation after predict?",
        default=True,
    )
    run_id = make_run_id("menu_literature")
    label = (
        "Calibrating LoRA (predict → eval → error-train)…"
        if session.calibrate_errors
        else "Running LoRA…"
    )
    try:
        with console.status(f"[bold green]{label}", spinner="dots"):
            run_lora_forecast(
                cfg,
                session.target_year,
                calibrate_errors=session.calibrate_errors,
                calibration_start=session.calibration_start,
                train_initial=session.train_first,
                run_eval=run_eval_after,
                run_id=run_id,
            )
        console.print(Panel("[bold green]✓ LoRA done[/bold green]", border_style="green"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red] {e}", border_style="red"))
        logger.exception("LoRA failed")


def _run_centroid(cfg: PredictConfig) -> None:
    if not _ensure_index_or_continue(cfg.db_path):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    run_topics_first = Confirm.ask("Run [bold]topics[/bold] before predict?", default=True)
    if not _confirm_centroid(cfg, run_topics_first):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    run_eval_after = Confirm.ask("Run evaluation after predict?", default=True)
    try:
        with console.status("[bold green]Running Centroid trajectory…", spinner="dots"):
            if run_topics_first:
                run_topics(TOPICS_CONFIG)
            run_predict(cfg)
        if run_eval_after:
            with console.status("[bold green]Evaluating…", spinner="dots"):
                eval_cfg = load_config(EVAL_CENTROID_CONFIG, EvalConfig)
                eval_cfg = replace(
                    eval_cfg,
                    db_path=cfg.db_path,
                    test_years=cfg.test_years,
                    top_k=cfg.top_k,
                    models=[MODEL_CENTROID_TRAJECTORY],
                )
                run_eval(eval_cfg)
        console.print(Panel("[bold green]✓ Centroid trajectory done[/bold green]", border_style="green"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red] {e}", border_style="red"))
        logger.exception("Centroid trajectory failed")


def _show_status() -> None:
    base = load_config(INDEX_PIPELINE_CONFIG, IndexPipelineConfig)
    db_path = Prompt.ask("SQLite database", default=base.db_path)
    ready, status_msg = _index_status(db_path)
    status_style = "green" if ready else "yellow"
    console.print(f"Index: [{status_style}]{status_msg}[/{status_style}]")

    if not resolve_path(db_path).exists():
        console.print("[yellow]Database file not found.[/yellow]")
        return

    db = Database(resolve_path(db_path))
    table = Table(title=f"Database status — {db_path}", box=box.ROUNDED)
    table.add_column("Table")
    table.add_column("Rows", justify="right")
    names = [
        "articles",
        "article_segments",
        "questions",
        "topic_centroid_snapshots",
        "topic_dynamics",
        "predictions",
        "evaluations",
    ]
    with db.connect() as conn:
        for name in names:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            except Exception:
                n = "—"
            table.add_row(name, str(n))
        try:
            embedded = conn.execute(
                "SELECT COUNT(*) FROM questions WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            table.add_row("questions (embedded)", str(embedded))
        except Exception:
            pass
    console.print(table)


def run_menu() -> None:
    _banner()
    base_lit = load_config(PREDICT_LITERATURE_CONFIG, PredictConfig)
    base_cent = load_config(PREDICT_CENTROID_CONFIG, PredictConfig)

    while True:
        console.print()
        console.print(
            "  [cyan]1[/cyan] Index   [cyan]2[/cyan] LoRA   "
            "[cyan]3[/cyan] Centroid trajectory   [cyan]4[/cyan] Database status   "
            "[cyan]0[/cyan] Quit"
        )
        choice = Prompt.ask(
            "[bold]Choose an action[/bold]",
            choices=["1", "2", "3", "4", "0"],
            default="1",
            show_choices=False,
        )

        if choice == "0":
            console.print("[dim]Goodbye.[/dim]")
            break
        if choice == "1":
            _run_index_menu()
        elif choice == "2":
            session = _configure_literature(base_lit)
            if session is not None:
                _run_literature(session)
        elif choice == "3":
            _run_centroid(_configure_centroid(base_cent))
        elif choice == "4":
            _show_status()


def main() -> None:
    try:
        run_menu()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(0)
