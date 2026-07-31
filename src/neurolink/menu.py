"""Interactive terminal menu — cluster job launcher aligned with the README."""

from __future__ import annotations

import logging
import shutil

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .cluster_jobs import JOB_ORDER, JOBS, list_jobs, submit_job
from .db import Database
from .index.pipeline import check_index_ready, get_index_counts
from .utils.config import resolve_path
from .utils.hf_auth import hf_token_available, set_hf_token

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_DB_PATH = "data/neurolink.db"

BANNER = r"""
 _   _ _____ _   _ ____   ___  _     ___ _   _ _  __
| \ | | ____| | | |  _ \ / _ \| |   |_ _| \ | | |/ /
|  \| |  _| | | | | |_) | | | | |    | ||  \| | ' /
| |\  | |___| |_| |  _ <| |_| | |___ | || |\  | . \
|_| \_|_____|\___/|_| \_\\___/|_____|___|_| \_|_|\_\
"""


def _banner() -> None:
    width = shutil.get_terminal_size().columns
    for line in BANNER.splitlines():
        console.print(Text.from_markup(line.center(width), style="bold white"))
    console.print(
        Panel(
            "Forecast neuroscience research directions from PubMed.\n"
            "Menu launches cluster jobs (setup → index → LoRA → benchmark).",
            border_style="green",
        )
    )


def _ensure_hf_token() -> bool:
    if hf_token_available():
        return True
    console.print(
        "[yellow]No Hugging Face token (HF_TOKEN / huggingface-cli login).[/yellow]\n"
        "Required for gated models (Mistral, BrainGPT)."
    )
    if not Confirm.ask("Enter a token for this session?", default=True):
        return False
    token = Prompt.ask("HF token", password=True)
    if not token.strip():
        return False
    set_hf_token(token.strip())
    console.print("[green]Token set for this session.[/green]")
    return True


def _show_status() -> None:
    db_path = DEFAULT_DB_PATH
    ready, msg = check_index_ready(db_path)
    style = "green" if ready else "yellow"
    console.print(f"Database: [bold]{db_path}[/bold]")
    console.print(f"Index: [{style}]{msg}[/{style}]")
    counts = get_index_counts(db_path)
    table = Table(box=box.SIMPLE, border_style="cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("articles", str(counts.articles))
    table.add_row("directions", str(counts.directions))
    table.add_row("directions_missing", str(counts.directions_missing))
    table.add_row("questions", str(counts.questions))
    table.add_row("questions_unembedded", str(counts.questions_unembedded))
    table.add_row("index_ready", str(counts.ready))
    console.print(table)
    try:
        db = Database(resolve_path(db_path))
        with db.connect(readonly=True) as conn:
            for t in ("predictions", "evaluations"):
                n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                console.print(f"{t}: {n}")
    except Exception:
        pass


def _optional_override(label: str, current: str | None = None) -> str | None:
    hint = f" [{current}]" if current else " [keep script default]"
    raw = Prompt.ask(f"{label}{hint}", default="")
    return raw.strip() or None


def _launch_job(key: str) -> None:
    spec = JOBS[key]
    console.print(Panel(f"[bold]{spec.title}[/bold]\n{spec.description}", border_style="cyan"))
    console.print(f"[dim]{spec.script}[/dim]")

    if spec.kind == "bash" and key == "setup" and not _ensure_hf_token():
        console.print("[red]Aborted — HF token required for setup.[/red]")
        return

    account = time = qos = constraint = job_name = None
    if spec.kind == "sbatch" and Confirm.ask("Override SLURM parameters?", default=False):
        account = _optional_override("--account")
        time = _optional_override("--time (HH:MM:SS)")
        qos = _optional_override("--qos")
        constraint = _optional_override("--constraint / -C")
        job_name = _optional_override("--job-name")

    dry = Confirm.ask("Dry-run only (print command)?", default=False)
    if not dry and not Confirm.ask(f"Launch {key}?", default=True):
        return

    try:
        code = submit_job(
            key,
            dry_run=dry,
            account=account,
            time=time,
            qos=qos,
            constraint=constraint,
            job_name=job_name,
        )
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        return
    if code == 0:
        console.print(Panel("[bold green]✓ Done[/bold green]", border_style="green"))
    else:
        console.print(f"[red]Exit code {code}[/red]")


def _jobs_table() -> None:
    table = Table(title="Cluster jobs", box=box.SIMPLE_HEAVY, border_style="green")
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Key")
    table.add_column("Title")
    table.add_column("Type")
    for i, spec in enumerate(list_jobs(), start=1):
        table.add_row(str(i), spec.key, spec.title, spec.kind)
    console.print(table)


def run_menu() -> None:
    _banner()
    while True:
        console.print()
        _jobs_table()
        console.print(
            "  [cyan]s[/cyan] status   [cyan]q[/cyan] quit   "
            "or pick a job number / key"
        )
        choice = Prompt.ask("Choice", default="s").strip().lower()
        if choice in {"q", "quit", "exit"}:
            break
        if choice in {"s", "status"}:
            _show_status()
            continue
        key = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(JOB_ORDER):
                key = JOB_ORDER[idx]
        elif choice in JOBS:
            key = choice
        if key is None:
            console.print("[yellow]Unknown choice.[/yellow]")
            continue
        _launch_job(key)


def main() -> None:
    run_menu()


if __name__ == "__main__":
    main()
