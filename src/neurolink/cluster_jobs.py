"""Cluster job registry and sbatch / login-script launchers."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CLUSTER_DIR = REPO_ROOT / "scripts" / "cluster"


@dataclass(frozen=True)
class JobSpec:
    """One production job from the README / cluster scripts."""

    key: str
    title: str
    kind: str  # sbatch | bash
    script: str  # path relative to repo root
    description: str
    # Optional sbatch overrides the launcher may prompt for / accept via CLI.
    overridable: tuple[str, ...] = ()
    default_env: dict[str, str] = field(default_factory=dict)


JOBS: dict[str, JobSpec] = {
    "setup": JobSpec(
        key="setup",
        title="Setup login (deps + HF cache)",
        kind="bash",
        script="scripts/cluster/setup_login.sh",
        description="Install pip addons and cache Hugging Face models (login node, needs HF_TOKEN).",
    ),
    "login-index": JobSpec(
        key="login-index",
        title="Login index (init-db + collect + citations)",
        kind="bash",
        script="scripts/cluster/login_index.sh",
        description="HTTP stages on the login node before the GPU direction job.",
    ),
    "direction-embed": JobSpec(
        key="direction-embed",
        title="Job 1 — LLM directions → impact → embed",
        kind="sbatch",
        script="scripts/cluster/job1_direction_embed.slurm",
        description="GPU: Mistral-Instruct research directions, rebuild questions, MiniLM embed.",
        overridable=("account", "time", "qos", "constraint", "job_name"),
    ),
    "train-base": JobSpec(
        key="train-base",
        title="Train LoRA-base (year_max=2022)",
        kind="sbatch",
        script="scripts/cluster/job_train_lora_base.slurm",
        description="Fine-tune Mistral-7B continuation LoRA → literature_base/year_max_2022.",
        overridable=("account", "time", "qos", "constraint", "job_name"),
    ),
    "train-instruct": JobSpec(
        key="train-instruct",
        title="Train LoRA-instruct (year_max=2022)",
        kind="sbatch",
        script="scripts/cluster/job_train_lora_instruct.slurm",
        description="Optional parallel instruct LoRA → literature_instruct/year_max_2022.",
        overridable=("account", "time", "qos", "constraint", "job_name"),
    ),
    "benchmark": JobSpec(
        key="benchmark",
        title="Benchmark 2022 (compare + eval)",
        kind="sbatch",
        script="scripts/cluster/job_benchmark_2022.slurm",
        description="LoRA-base vs mistral_base vs braingpt (2023–2025); optional instruct round.",
        overridable=("account", "time", "qos", "constraint", "job_name"),
    ),
    "eval-thr90": JobSpec(
        key="eval-thr90",
        title="Eval thr90 (τ=0.90, no re-predict)",
        kind="sbatch",
        script="scripts/cluster/job_eval_thr90.slurm",
        description="Re-score existing 2022 bench preds with MiniLM match 0.90; corpus_minilm P@50 floor.",
        overridable=("account", "time", "qos", "constraint", "job_name"),
    ),
    "eval-thr70": JobSpec(
        key="eval-thr70",
        title="Eval thr70 (τ=0.70, no re-predict)",
        kind="sbatch",
        script="scripts/cluster/job_eval_thr70.slurm",
        description="Re-score existing 2022 bench preds with MiniLM match 0.70; corpus_minilm P@50 floor.",
        overridable=("account", "time", "qos", "constraint", "job_name"),
    ),
    "reindex-specific": JobSpec(
        key="reindex-specific",
        title="Specific dirs — copy DB + force re-extract → impact → embed",
        kind="sbatch",
        script="scripts/cluster/job_reindex_specific.slurm",
        description="Parallel DB neurolink_specific.db with specific-direction LLM prompt.",
        overridable=("account", "time", "qos", "constraint", "job_name"),
    ),
    "train-base-specific": JobSpec(
        key="train-base-specific",
        title="Train LoRA-base specific (year_max=2022)",
        kind="sbatch",
        script="scripts/cluster/job_train_lora_base_specific_2022.slurm",
        description="Fine-tune on neurolink_specific.db → literature_base_specific/year_max_2022.",
        overridable=("account", "time", "qos", "constraint", "job_name"),
    ),
    "benchmark-specific": JobSpec(
        key="benchmark-specific",
        title="Benchmark 2022 specific (compare + eval)",
        kind="sbatch",
        script="scripts/cluster/job_benchmark_2022_specific.slurm",
        description="Specific-direction LoRA vs mistral_base vs braingpt (2023–2025).",
        overridable=("account", "time", "qos", "constraint", "job_name"),
    ),
    "train-base-2026": JobSpec(
        key="train-base-2026",
        title="Train LoRA-base (year_max=2026)",
        kind="sbatch",
        script="scripts/cluster/job_train_lora_base_2026.slurm",
        description="Separate experiment: fine-tune on neurolink.db → literature_base/year_max_2026.",
        overridable=("account", "time", "qos", "constraint", "job_name"),
    ),
    "forecast-2025": JobSpec(
        key="forecast-2025",
        title="Job 3 — LoRA 2025 + bench 2026 + predict 2027",
        kind="sbatch",
        script="scripts/cluster/job3_lora2025_predict.slurm",
        description="Train year_max=2025, benchmark 2026, greedy 2027 (optional).",
        overridable=("account", "time", "qos", "constraint", "job_name"),
    ),
}

JOB_ORDER = (
    "setup",
    "login-index",
    "direction-embed",
    "train-base",
    "train-instruct",
    "benchmark",
    "reindex-specific",
    "train-base-specific",
    "benchmark-specific",
    "train-base-2026",
    "forecast-2025",
)


def list_jobs() -> list[JobSpec]:
    return [JOBS[k] for k in JOB_ORDER]


def resolve_script(spec: JobSpec) -> Path:
    path = REPO_ROOT / spec.script
    if not path.is_file():
        raise FileNotFoundError(f"Missing job script: {path}")
    return path


def build_sbatch_argv(
    spec: JobSpec,
    *,
    account: str | None = None,
    time: str | None = None,
    qos: str | None = None,
    constraint: str | None = None,
    job_name: str | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Build ``sbatch [overrides…] script`` argv."""
    script = resolve_script(spec)
    argv = ["sbatch"]
    if account:
        argv.append(f"--account={account}")
    if time:
        argv.append(f"--time={time}")
    if qos:
        argv.append(f"--qos={qos}")
    if constraint:
        argv.append(f"--constraint={constraint}")
    if job_name:
        argv.append(f"--job-name={job_name}")
    if extra:
        argv.extend(extra)
    argv.append(str(script))
    return argv


def submit_job(
    key: str,
    *,
    dry_run: bool = False,
    account: str | None = None,
    time: str | None = None,
    qos: str | None = None,
    constraint: str | None = None,
    job_name: str | None = None,
    extra: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Launch a registered job (sbatch or bash). Return process exit code."""
    if key not in JOBS:
        raise KeyError(f"Unknown job {key!r}. Choose from: {', '.join(JOB_ORDER)}")
    spec = JOBS[key]
    script = resolve_script(spec)
    run_env = {**os.environ, **spec.default_env, **(env or {})}

    if spec.kind == "bash":
        argv = ["bash", str(script)]
    else:
        if shutil.which("sbatch") is None and not dry_run:
            raise RuntimeError(
                "sbatch not found — run on the cluster login node, or pass --dry-run"
            )
        argv = build_sbatch_argv(
            spec,
            account=account,
            time=time,
            qos=qos,
            constraint=constraint,
            job_name=job_name,
            extra=extra,
        )

    logger.info("%s%s", "[dry-run] " if dry_run else "", " ".join(argv))
    if dry_run:
        print(" ".join(argv))
        return 0

    # Ensure logs/ exists for sbatch --output paths.
    (REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(argv, cwd=str(REPO_ROOT), env=run_env, check=False)
    return int(completed.returncode)
