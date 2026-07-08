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
    MODEL_BRAINGPT,
    MODEL_LITERATURE_LORA,
    MODEL_MISTRAL_BASE,
    PredictConfig,
    calibration_years,
    resolve_benchmark,
    run_lora_forecast,
    run_predict,
    run_train_literature,
)
from .forecast.predict.literature_lora import adapter_exists, list_saved_lora_year_max
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
from .utils.torch_device import cuda_available, cuda_device_name
from .workflow import CompleteWorkflowConfig, run_complete_workflow

logger = logging.getLogger(__name__)

console = Console()
PREDICT_LITERATURE_CONFIG = "config/forecast/predict_literature.yaml"
PREDICT_COMPARE_CONFIG = "config/forecast/predict_compare.yaml"
EVAL_LITERATURE_CONFIG = "config/eval/eval_literature.yaml"
EVAL_COMPARE_CONFIG = "config/eval/eval_compare.yaml"
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
            """Modular pipeline to forecast emergent neuroscience research directions from PubMed literature.\n
            1 - Index: collect PubMed → segment → impact → embed.\n
            2 - Literature LoRA: fine-tune Mistral-7B on temporal question pairs.\n
            3 - Benchmark: compare LoRA vs Mistral-7B vs BrainGPT on years after a saved LoRA year_max.\n
            5 - Complete workflow (GPU): full cluster run see Protocol.md.\n""",
            border_style="green",
        )
    )


def _show_params_table(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title, box=box.SIMPLE_HEAVY, border_style="green")
    table.add_column("Parameter", style="bold")
    table.add_column("Value")
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)


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

    literature_defaults = cfg.literature
    saved = list_saved_lora_year_max(literature_defaults)
    year_max = target_year - 1
    has_adapter = adapter_exists(literature_defaults, year_max)

    if saved:
        console.print(
            f"[dim]Saved LoRA adapters (year_max): {', '.join(map(str, saved))}[/dim]"
        )
    else:
        console.print("[yellow]No saved LoRA adapters in data/models/literature/.[/yellow]")

    if has_adapter:
        console.print(
            f"[green]Local adapter found:[/green] "
            f"data/models/literature/year_max_{year_max}/lora/"
        )
        train_first = Confirm.ask(
            "Train LoRA before predict? (No = inference with saved weights)",
            default=False,
        )
    else:
        console.print(
            f"[yellow]No adapter for year_max_{year_max} — training required before predict.[/yellow]"
        )
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

    year_max = session.target_year - 1
    has_adapter = adapter_exists(cfg.literature, year_max)

    _show_params_table(
        "literature_lora summary",
        [
            ("Model", MODEL_LITERATURE_LORA),
            ("Database", cfg.db_path),
            ("Target year", str(session.target_year)),
            ("LoRA adapter", f"year_max_{year_max} ({'found' if has_adapter else 'missing'})"),
            ("Mode", "train + predict" if session.train_first else "inference only"),
            ("Error-train calibration", str(session.calibrate_errors)),
            ("Calibration path", cal_label),
            ("Literature through (simple)", str(session.year_max)),
            ("Top-k", str(max(cfg.top_k))),
            ("Context questions", str(cfg.literature.max_context_questions)),
            ("Temperature", f"{cfg.literature.llm.temperature:.2f}"),
        ],
    )
    return Confirm.ask("\nRun LoRA?", default=True)


@dataclass
class CompareSession:
    cfg: PredictConfig
    lora_anchor_year_max: int
    benchmark_years: list[int]


def _configure_compare(base_cfg: PredictConfig) -> CompareSession | None:
    console.print("\n[bold magenta]LLM benchmark[/bold magenta]")
    console.print(
        "[dim]Compare literature_lora · mistral_base · braingpt on years "
        "strictly after a saved LoRA year_max.[/dim]"
    )

    db_path = Prompt.ask("SQLite database", default=base_cfg.db_path)
    cfg = replace(base_cfg, db_path=db_path)

    saved = list_saved_lora_year_max(cfg.literature)
    if not saved:
        console.print("[yellow]No saved LoRA adapters found.[/yellow]")
        if Confirm.ask("Train a new LoRA adapter now?", default=True):
            train_year_max = IntPrompt.ask(
                "Literature through year (year_max)",
                default=max(cfg.test_years) - 1 if cfg.test_years else 2023,
            )
            if not _ensure_hf_token():
                return None
            try:
                with console.status("[bold green]Training LoRA…", spinner="dots"):
                    run_train_literature(
                        replace(cfg, test_years=[train_year_max + 1]),
                        year_max=train_year_max,
                    )
            except Exception as e:
                console.print(f"[red]Training failed: {e}[/red]")
                return None
            saved = list_saved_lora_year_max(cfg.literature)
        if not saved:
            console.print("[red]A saved LoRA adapter is required for the benchmark.[/red]")
            return None

    console.print(f"[dim]Saved LoRA adapters (year_max): {', '.join(map(str, saved))}[/dim]")
    default_anchor = saved[-1]
    lora_anchor = IntPrompt.ask(
        "LoRA anchor year_max (literature trained through)",
        default=default_anchor,
    )
    if lora_anchor not in saved:
        console.print(
            f"[red]year_max_{lora_anchor} not found. Available: {', '.join(map(str, saved))}[/red]"
        )
        return None

    try:
        benchmark_cfg, anchor, benchmark_years = resolve_benchmark(
            cfg, lora_year_max=lora_anchor, db_path=db_path
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        return None

    console.print(
        f"[green]Benchmark years (>{anchor}):[/green] {', '.join(map(str, benchmark_years))}"
    )

    include_lora = Confirm.ask("Include literature_lora?", default=True)
    include_mistral = Confirm.ask("Include mistral_base?", default=True)
    include_braingpt = Confirm.ask("Include braingpt?", default=True)
    models: list[str] = []
    if include_lora:
        models.append(MODEL_LITERATURE_LORA)
    if include_mistral:
        models.append(MODEL_MISTRAL_BASE)
    if include_braingpt:
        models.append(MODEL_BRAINGPT)
    if not models:
        console.print("[red]Select at least one model.[/red]")
        return None

    top_k = IntPrompt.ask("Max top-k predictions", default=max(cfg.top_k) if cfg.top_k else 50)
    max_ctx = IntPrompt.ask("Context questions", default=cfg.literature.max_context_questions)
    temp = FloatPrompt.ask("Generation temperature", default=cfg.literature.llm.temperature)
    tokens = IntPrompt.ask("Max tokens per question", default=cfg.literature.llm.max_new_tokens)
    use_4bit = Confirm.ask("4-bit quantization?", default=cfg.literature.use_4bit)

    literature = replace(
        benchmark_cfg.literature,
        max_context_questions=max_ctx,
        use_4bit=use_4bit,
        llm=replace(benchmark_cfg.literature.llm, temperature=temp, max_new_tokens=tokens),
    )
    final_cfg = replace(
        benchmark_cfg,
        top_k=[top_k],
        models=models,
        literature=literature,
    )
    return CompareSession(
        cfg=final_cfg,
        lora_anchor_year_max=anchor,
        benchmark_years=benchmark_years,
    )


def _confirm_compare(session: CompareSession) -> bool:
    cfg = session.cfg
    _show_params_table(
        "benchmark summary",
        [
            ("Models", ", ".join(cfg.models)),
            ("Database", cfg.db_path),
            ("LoRA anchor (year_max)", str(session.lora_anchor_year_max)),
            ("Benchmark years", ", ".join(map(str, session.benchmark_years))),
            ("Top-k", str(max(cfg.top_k))),
            ("Context questions", str(cfg.literature.max_context_questions)),
            ("Temperature", f"{cfg.literature.llm.temperature:.2f}"),
            ("4-bit", str(cfg.literature.use_4bit)),
            ("BrainGPT model", cfg.literature.braingpt_model),
        ],
    )
    return Confirm.ask("\nRun benchmark?", default=True)


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
    term = _optional_str(
        "PubMed query (empty = MeSH)",
        cfg.term or cfg.mesh,
    )
    mesh = Prompt.ask("MeSH fallback (if query empty)", default=cfg.mesh)
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
        term=term if term else None,
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
    device = cfg.device
    if cuda_available():
        gpu_name = cuda_device_name()
        label = f" ({gpu_name})" if gpu_name else ""
        use_gpu = Confirm.ask(
            f"Use CUDA GPU for PubMedBERT{label}?",
            default=cfg.device == "cuda",
        )
        device = "cuda" if use_gpu else "cpu"
    else:
        console.print("[dim]No CUDA GPU detected — PubMedBERT will run on CPU.[/dim]")
        device = "cpu"
    return replace(cfg, pubmedbert_model=model, device=device)


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
    query = cfg.term if cfg.term else f"{cfg.mesh}[MeSH]"
    _show_params_table(
        "collect",
        [
            ("Database", cfg.db_path),
            ("Query", query),
            ("Years", f"{cfg.year_from}–{cfg.year_to}"),
            ("Exclude reviews", str(cfg.exclude_reviews)),
            ("retmax", str(cfg.retmax)),
            ("batch_size", str(cfg.batch_size)),
            ("delay_seconds", f"{cfg.delay_seconds:.2f}"),
            ("email", cfg.email or "—"),
        ],
    )


def _show_segment_summary(cfg: SegmentConfig) -> None:
    device_label = cfg.device
    if cfg.device == "cuda" and cuda_available():
        gpu_name = cuda_device_name()
        if gpu_name:
            device_label = f"cuda ({gpu_name})"
    elif cfg.device == "auto":
        device_label = "auto (CUDA if available)"
    _show_params_table(
        "segment",
        [
            ("Database", cfg.db_path),
            ("Method", "rules + PubMedBERT"),
            ("PubMedBERT model", cfg.pubmedbert_model),
            ("Device", device_label),
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
    year_max = session.target_year - 1
    if not _ensure_index_or_continue(cfg.db_path):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    if not _ensure_hf_token():
        console.print("[yellow]Cancelled — Hugging Face token required.[/yellow]")
        return
    if not session.train_first and not adapter_exists(cfg.literature, year_max):
        console.print(
            f"[red]No local adapter at year_max_{year_max} — cannot run inference only.[/red]"
        )
        if Confirm.ask("Train LoRA now?", default=True):
            session = replace(session, train_first=True)
        else:
            console.print("[yellow]Cancelled.[/yellow]")
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


def _run_compare(session: CompareSession) -> None:
    cfg = session.cfg
    if not _ensure_index_or_continue(cfg.db_path):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    if not _ensure_hf_token():
        console.print("[yellow]Cancelled — Hugging Face token required.[/yellow]")
        return

    if not _confirm_compare(session):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    run_eval_after = Confirm.ask(
        "Run evaluation (P@k, BrainBench, contamination) after benchmark?",
        default=True,
    )

    try:
        with console.status("[bold green]Running LLM benchmark…", spinner="dots"):
            pred_run_id = make_run_id("menu_benchmark")
            run_predict(cfg, run_id=pred_run_id)
        if run_eval_after:
            with console.status("[bold green]Evaluating…", spinner="dots"):
                eval_cfg = load_config(EVAL_COMPARE_CONFIG, EvalConfig)
                eval_cfg = replace(
                    eval_cfg,
                    db_path=cfg.db_path,
                    test_years=session.benchmark_years,
                    top_k=cfg.top_k,
                    models=cfg.models,
                    predict_config=PREDICT_COMPARE_CONFIG,
                    run_id=pred_run_id,
                )
                run_eval(eval_cfg, run_id=make_run_id("eval"))
        console.print(Panel("[bold green]✓ Benchmark done[/bold green]", border_style="green"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red] {e}", border_style="red"))
        logger.exception("Benchmark failed")


def _run_complete_workflow() -> None:
    console.print("\n[bold magenta]Complete workflow (GPU required)[/bold magenta]")
    console.print(
        "[dim]Index → LoRA@2022 → benchmark+eval → LoRA@2025 → benchmark+eval → forecast 2027. "
        "Uses locked protocol (temperature=0, 3 LLMs).[/dim]"
    )
    if not _ensure_hf_token():
        console.print("[yellow]Cancelled.[/yellow]")
        return
    skip_index = Confirm.ask("Skip index (database already built)?", default=False)
    db_path = load_config(INDEX_PIPELINE_CONFIG, IndexPipelineConfig).db_path
    if skip_index:
        if not _ensure_index_or_continue(db_path):
            console.print("[yellow]Cancelled.[/yellow]")
            return
    elif not Confirm.ask(
        "Run full index first (collect + impact + segment + embed)?", default=True
    ):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    device = "cuda" if cuda_available() else "cpu"
    if cuda_available():
        gpu = cuda_device_name()
        label = f"cuda ({gpu})" if gpu else "cuda"
        if Confirm.ask(f"Use GPU for segment? ({label})", default=True):
            device = "cuda"
        else:
            device = "cpu"
    else:
        console.print("[yellow]No CUDA — segment will run on CPU (slow).[/yellow]")

    _show_params_table(
        "complete workflow",
        [
            ("Skip index", str(skip_index)),
            ("Segment device", device),
            ("LoRA anchor 1", "2022"),
            ("LoRA anchor 2", "2025"),
            ("Forecast year", "2027"),
            ("Benchmark models", "literature_lora, mistral_base, braingpt"),
            ("Temperature", "0.0 (locked)"),
        ],
    )
    if not Confirm.ask("\nRun complete workflow?", default=True):
        console.print("[yellow]Cancelled.[/yellow]")
        return

    cfg = CompleteWorkflowConfig(skip_index=skip_index, segment_device=device)
    try:
        with console.status("[bold green]Running complete workflow…", spinner="dots"):
            run_complete_workflow(cfg)
        console.print(Panel("[bold green]✓ Complete workflow done[/bold green]", border_style="green"))
    except Exception as e:
        console.print(Panel(f"[bold red]Error:[/bold red] {e}", border_style="red"))
        logger.exception("Complete workflow failed")


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
    base_cmp = load_config(PREDICT_COMPARE_CONFIG, PredictConfig)

    while True:
        console.print()
        console.print(
            "  [cyan]1[/cyan] Index   [cyan]2[/cyan] LoRA   [cyan]3[/cyan] Benchmark LLMs   "
            "[cyan]5[/cyan] Complete workflow (GPU)   "
            "[cyan]4[/cyan] Database status   [cyan]0[/cyan] Quit"
        )
        choice = Prompt.ask(
            "[bold]Choose an action[/bold]",
            choices=["1", "2", "3", "4", "5", "0"],
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
            session = _configure_compare(base_cmp)
            if session is not None:
                _run_compare(session)
        elif choice == "4":
            _show_status()
        elif choice == "5":
            _run_complete_workflow()


def main() -> None:
    try:
        run_menu()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(0)
