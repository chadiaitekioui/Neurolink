"""Literature approach — LoRA-tuned LLM generates novel research questions."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ...eval.matching import TfidfMatcher
from .llm_core import CausalLMConfig, generate_questions

logger = logging.getLogger(__name__)


@dataclass
class LiteratureLoraConfig:
    base_model: str = "mistralai/Mistral-7B-v0.1"
    adapter_dir: str = "data/models/literature"
    max_context_questions: int = 40
    lora_r: int = 16
    lora_alpha: int = 32
    use_4bit: bool = True
    backend: str = "lora"
    train_epochs: int = 1
    train_lr: float = 1e-4
    train_fraction: float = 0.7
    max_examples_per_year: int = 0  # 0 = no cap; train_fraction applies
    error_train_epochs: int = 2
    error_max_examples: int = 50
    semantic_threshold: float = 0.55
    error_critical_only: bool = True
    braingpt_adapter: str = "BrainGPT/BrainGPT-7B-v0.2"
    benchmark_lora_year_max: int | None = None  # compare: fixed LoRA adapter for all benchmark years
    llm: CausalLMConfig = field(default_factory=CausalLMConfig)


def _questions_by_year(conn: sqlite3.Connection, year_max: int) -> dict[int, list[str]]:
    rows = conn.execute(
        """
        SELECT question_text, year, impact_score FROM questions
        WHERE year IS NOT NULL AND year <= ?
        """,
        (year_max,),
    ).fetchall()
    buckets: dict[int, list[tuple[float, str]]] = {}
    for r in rows:
        text = (r["question_text"] or "").strip()
        if not text:
            continue
        yr = int(r["year"])
        buckets.setdefault(yr, []).append((float(r["impact_score"] or 0), text))
    by_year: dict[int, list[str]] = {}
    for yr, items in buckets.items():
        items.sort(key=lambda x: (-x[0], x[1]))
        by_year[yr] = [text for _, text in items]
    return by_year


def _training_examples_for_year(questions: list[str], cfg: LiteratureLoraConfig) -> list[str]:
    if not questions:
        return []
    n_use = max(1, int(len(questions) * cfg.train_fraction))
    if cfg.max_examples_per_year > 0:
        n_use = min(n_use, cfg.max_examples_per_year)
    return questions[:n_use]


def build_context_summary(conn: sqlite3.Connection, N: int, max_q: int) -> str:
    rows = conn.execute(
        """
        SELECT question_text, impact_score, year FROM questions
        WHERE year IS NOT NULL AND year <= ? AND year >= ?
        ORDER BY COALESCE(impact_score, 0) DESC, year DESC
        LIMIT ?
        """,
        (N - 1, max(N - 5, 0), max_q),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT question_text, impact_score, year FROM questions WHERE year <= ? LIMIT ?",
            (N - 1, max_q),
        ).fetchall()
    lines = []
    for r in rows:
        yr = r["year"] or "?"
        lines.append(f"- [{yr}] {r['question_text'][:280]}")
    return "\n".join(lines)


def build_generation_prompt(conn: sqlite3.Connection, N: int, cfg: LiteratureLoraConfig) -> str:
    context = build_context_summary(conn, N, cfg.max_context_questions)
    return (
        f"You are a neuroscience research forecaster.\n"
        f"Literature and research questions published until year {N - 1}:\n"
        f"{context}\n\n"
        f"Predict NOVEL research questions likely to be studied in {N}.\n"
        f"Rules:\n"
        f"- Do NOT copy verbatim questions from the list above.\n"
        f"- Some may extend existing themes with new angles; others should be genuinely emergent.\n"
        f"- Output one question per line, each ending with '?'.\n\n"
        f"New research questions for {N}:\n"
    )


def build_temporal_training_prompt(
    conn: sqlite3.Connection,
    year: int,
    cfg: LiteratureLoraConfig,
) -> str:
    """Training prompt aligned with inference format for year+1."""
    return build_generation_prompt(conn, year + 1, cfg)


def build_temporal_examples(
    conn: sqlite3.Connection,
    year_max: int,
    cfg: LiteratureLoraConfig,
) -> list[tuple[str, str]]:
    """(prompt, completion) pairs: context ≤ T → questions at T+1, T+1 ≤ year_max."""
    by_year = _questions_by_year(conn, year_max)
    examples: list[tuple[str, str]] = []
    for year in sorted(by_year):
        if year + 1 > year_max:
            break
        prompt = build_temporal_training_prompt(conn, year, cfg)
        for q in _training_examples_for_year(by_year.get(year + 1, []), cfg):
            examples.append((prompt, q[:400]))
    return examples


def _ground_truth_for_year(
    conn: sqlite3.Connection,
    target_year: int,
    *,
    critical_only: bool,
) -> list[str]:
    rows = conn.execute(
        "SELECT question_text, is_critical FROM questions WHERE year = ?",
        (target_year,),
    ).fetchall()
    refs: list[str] = []
    for r in rows:
        if critical_only and not r["is_critical"]:
            continue
        text = (r["question_text"] or "").strip()
        if text:
            refs.append(text)
    if not refs:
        refs = [(r["question_text"] or "").strip() for r in rows if (r["question_text"] or "").strip()]
    return refs


def _predictions_for_year(
    conn: sqlite3.Connection,
    target_year: int,
    model: str,
    pred_run_id: str | None,
    eval_k: int,
) -> list[str]:
    if pred_run_id:
        rows = conn.execute(
            """
            SELECT question_predicted FROM predictions
            WHERE target_year=? AND model=? AND run_id=?
            ORDER BY rank LIMIT ?
            """,
            (target_year, model, pred_run_id, eval_k),
        ).fetchall()
        if rows:
            return [r["question_predicted"] for r in rows]

    row = conn.execute(
        """
        SELECT run_id FROM runs WHERE stage='predict' ORDER BY created_at DESC LIMIT 1
        """
    ).fetchone()
    if not row:
        return []
    rows = conn.execute(
        """
        SELECT question_predicted FROM predictions
        WHERE target_year=? AND model=? AND run_id=?
        ORDER BY rank LIMIT ?
        """,
        (target_year, model, row["run_id"], eval_k),
    ).fetchall()
    return [r["question_predicted"] for r in rows]


def build_error_examples(
    conn: sqlite3.Connection,
    target_year: int,
    cfg: LiteratureLoraConfig,
    *,
    pred_run_id: str | None = None,
    model: str = "literature_lora",
    eval_k: int = 50,
) -> list[tuple[str, str]]:
    """
    Build supervised examples from ground-truth questions missed by predictions.

  Temporal guard: target_year must not exceed the adapter's training horizon when
  used for honest evaluation — callers save the adapter as year_max_{target_year}
  only after measuring performance on target_year.
    """
    refs = _ground_truth_for_year(conn, target_year, critical_only=cfg.error_critical_only)
    if not refs:
        logger.warning("No ground truth for year %d — cannot build error examples", target_year)
        return []

    preds = _predictions_for_year(conn, target_year, model, pred_run_id, eval_k)
    if not preds:
        logger.warning(
            "No predictions for year %d (model=%s) — using all ground-truth as examples",
            target_year,
            model,
        )
        missed = refs
    else:
        matcher = TfidfMatcher(refs, cfg.semantic_threshold)
        missed = matcher.uncovered_references(preds, k=eval_k)

    if not missed:
        logger.info("All ground-truth questions covered for year %d — no error examples", target_year)
        return []

    prompt = build_generation_prompt(conn, target_year, cfg)
    examples = [(prompt, q[:400]) for q in missed[: cfg.error_max_examples]]
    logger.info(
        "Error examples for %d: %d missed / %d ground-truth",
        target_year,
        len(examples),
        len(refs),
    )
    return examples


def _adapter_path(cfg: LiteratureLoraConfig, year_max: int) -> Path:
    return Path(cfg.adapter_dir) / f"year_max_{year_max}"


def adapter_exists(cfg: LiteratureLoraConfig, year_max: int) -> bool:
    return (_adapter_path(cfg, year_max) / "lora").exists()


def list_saved_lora_year_max(cfg: LiteratureLoraConfig) -> list[int]:
    """year_max values with a saved LoRA adapter under adapter_dir."""
    root = Path(cfg.adapter_dir)
    if not root.is_dir():
        return []
    years: list[int] = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("year_max_"):
            continue
        try:
            year_max = int(path.name.removeprefix("year_max_"))
        except ValueError:
            continue
        if adapter_exists(cfg, year_max):
            years.append(year_max)
    return sorted(years)


def infer_benchmark_years(conn: sqlite3.Connection, lora_year_max: int) -> list[int]:
    """Forecast years strictly after lora_year_max that have indexed questions."""
    rows = conn.execute(
        """
        SELECT DISTINCT year FROM questions
        WHERE year IS NOT NULL AND year > ?
        ORDER BY year
        """,
        (lora_year_max,),
    ).fetchall()
    return [int(r["year"]) for r in rows]


def _load_train_model(cfg: LiteratureLoraConfig, continue_from: Path | None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    load_kwargs: dict = {}
    if cfg.use_4bit:
        try:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs["device_map"] = "auto"
        except Exception:
            pass

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)

    if continue_from and (continue_from / "lora").exists():
        model = PeftModel.from_pretrained(model, str(continue_from / "lora"), is_trainable=True)
        logger.info("Continuing LoRA from %s", continue_from / "lora")
    else:
        lora_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)

    model.train()
    return model, tokenizer, torch


def _train_on_examples(
    model,
    tokenizer,
    torch,
    examples: list[tuple[str, str]],
    *,
    epochs: int,
    lr: float,
    use_4bit: bool,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for prompt, completion in examples:
            text = prompt + " " + completion
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=768)
            if not use_4bit:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            labels = inputs["input_ids"].clone()
            prompt_len = len(tokenizer(prompt, truncation=True, max_length=768)["input_ids"])
            labels[0, :prompt_len] = -100
            out = model(**inputs, labels=labels)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(out.loss.item())
        logger.info(
            "LoRA epoch %d/%d — %d examples, avg loss=%.4f",
            epoch + 1,
            epochs,
            len(examples),
            total_loss / max(len(examples), 1),
        )


def _save_adapter(model, tokenizer, cfg: LiteratureLoraConfig, year_max: int) -> Path:
    out_dir = _adapter_path(cfg, year_max)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "lora")
    tokenizer.save_pretrained(out_dir / "lora")
    return out_dir


def train_literature_lora(
    conn: sqlite3.Connection,
    year_max: int,
    cfg: LiteratureLoraConfig,
    *,
    continue_from_year_max: int | None = None,
) -> int:
    """Fine-tune LoRA on (context ≤ T, questions at T+1) pairs — no temporal leakage."""
    if cfg.backend != "lora":
        return 0

    try:
        examples = build_temporal_examples(conn, year_max, cfg)
        if len(examples) < 3:
            logger.warning("Not enough examples for LoRA (%d)", len(examples))
            return 0

        continue_from = (
            _adapter_path(cfg, continue_from_year_max)
            if continue_from_year_max is not None
            else None
        )
        model, tokenizer, torch = _load_train_model(cfg, continue_from)
        _train_on_examples(
            model,
            tokenizer,
            torch,
            examples,
            epochs=cfg.train_epochs,
            lr=cfg.train_lr,
            use_4bit=cfg.use_4bit,
        )
        out_dir = _save_adapter(model, tokenizer, cfg, year_max)
        logger.info("Literature LoRA saved: %s (%d examples)", out_dir, len(examples))
        return len(examples)
    except ImportError:
        logger.warning("peft/torch unavailable — skipping literature LoRA training")
        return 0


def train_literature_lora_on_errors(
    conn: sqlite3.Connection,
    target_year: int,
    cfg: LiteratureLoraConfig,
    *,
    pred_run_id: str | None = None,
    model: str = "literature_lora",
    eval_k: int = 50,
) -> int:
    """
    Fine-tune on ground-truth questions missed at target_year.

    Loads adapter year_max_{target_year - 1} and saves year_max_{target_year}.
    """
    if cfg.backend != "lora":
        return 0

    try:
        examples = build_error_examples(
            conn,
            target_year,
            cfg,
            pred_run_id=pred_run_id,
            model=model,
            eval_k=eval_k,
        )
        if not examples:
            return 0

        continue_from = _adapter_path(cfg, target_year - 1)
        if not (continue_from / "lora").exists():
            logger.warning(
                "No adapter at %s — training error examples from scratch",
                continue_from / "lora",
            )
            continue_from = None

        peft_model, tokenizer, torch = _load_train_model(cfg, continue_from)
        _train_on_examples(
            peft_model,
            tokenizer,
            torch,
            examples,
            epochs=cfg.error_train_epochs,
            lr=cfg.train_lr,
            use_4bit=cfg.use_4bit,
        )
        out_dir = _save_adapter(peft_model, tokenizer, cfg, target_year)
        logger.info(
            "Error-correction LoRA saved: %s (%d examples for year %d)",
            out_dir,
            len(examples),
            target_year,
        )
        return len(examples)
    except ImportError:
        logger.warning("peft/torch unavailable — skipping error LoRA training")
        return 0


def resolve_literature_llm_cfg(
    cfg: LiteratureLoraConfig,
    year_max: int,
    model: str,
) -> CausalLMConfig:
    """Build CausalLMConfig for literature_lora, mistral_base, or braingpt."""
    llm_cfg = CausalLMConfig(
        base_model=cfg.base_model,
        use_4bit=cfg.use_4bit,
        max_new_tokens=cfg.llm.max_new_tokens,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
    )
    if model == "literature_lora":
        adapter_year = (
            cfg.benchmark_lora_year_max
            if cfg.benchmark_lora_year_max is not None
            else year_max
        )
        lora_dir = _adapter_path(cfg, adapter_year) / "lora"
        if lora_dir.exists():
            llm_cfg.adapter_path = str(lora_dir)
            if cfg.benchmark_lora_year_max is not None:
                logger.info(
                    "Benchmark LoRA adapter year_max_%d (predict year uses literature ≤ N−1)",
                    adapter_year,
                )
        else:
            logger.warning(
                "No local LoRA adapter for year_max=%d — falling back to base Mistral",
                adapter_year,
            )
    elif model == "braingpt":
        llm_cfg.adapter_path = cfg.braingpt_adapter
        logger.info("Using BrainGPT adapter: %s", cfg.braingpt_adapter)
    elif model == "mistral_base":
        logger.info("Using base Mistral (no adapter)")
    else:
        logger.warning("Unknown literature LM %r — base Mistral only", model)
    return llm_cfg


def predict_literature_lm(
    conn: sqlite3.Connection,
    N: int,
    k: int,
    cfg: LiteratureLoraConfig,
    *,
    model: str = "literature_lora",
) -> list[tuple[str, float]]:
    """Generate novel questions for year N with a literature LM variant."""
    year_max = N - 1
    prompt = build_generation_prompt(conn, N, cfg)
    llm_cfg = resolve_literature_llm_cfg(cfg, year_max, model)

    try:
        return generate_questions(prompt, llm_cfg, k, oversample=3)
    except ImportError as e:
        logger.error("%s: %s — pip install -e '.[train]'", model, e)
        return []


def make_literature_predictor(cfg: LiteratureLoraConfig, model: str = "literature_lora"):
    def _fn(conn: sqlite3.Connection, N: int, k: int, rng) -> list[tuple[str, float]]:
        return predict_literature_lm(conn, N, k, cfg, model=model)

    return _fn
