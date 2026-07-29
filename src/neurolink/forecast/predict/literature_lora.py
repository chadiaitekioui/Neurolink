"""Literature approach — LoRA-tuned LLM generates novel research directions."""

from __future__ import annotations

import logging
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ...eval.matching import make_matcher
from .generation_audit import GenerationAudit
from .direction_filter import build_context_blocklist
from .llm_core import (
    CausalLMConfig,
    generate_directions_batch,
    generate_directions_iterative,
    move_inputs_to_model,
    release_gpu_memory,
)

logger = logging.getLogger(__name__)


@dataclass
class LiteratureLoraConfig:
    base_model: str = "mistralai/Mistral-7B-v0.1"
    adapter_dir: str = "data/models/literature"
    max_context_questions: int = 30
    lora_r: int = 16
    lora_alpha: int = 32
    use_4bit: bool = True
    backend: str = "lora"
    train_epochs: int = 1
    train_lr: float = 1e-4
    train_fraction: float = 0.7
    max_examples_per_year: int = 0  # 0 = no cap; train_fraction applies
    train_sampling: str = "impact_topk"  # impact_topk | impact_weighted
    grad_accumulation_steps: int = 1
    shuffle_training_examples: bool = False
    use_hf_trainer: bool = True
    train_save_steps: int = 0  # 0 = checkpoints disabled (final save only)
    train_save_total_limit: int = 2
    resume_from_checkpoint: str | None = None
    train_log_every: int = 50
    # Train monitoring: held-out slice of temporal pairs (all ≤ year_max; never test years).
    train_val_fraction: float = 0.1
    train_eval_steps: int = 0  # 0 → same as train_save_steps (or 50)
    train_early_stopping_patience: int = 3  # 0 = disabled
    train_max_length: int = 1024
    error_train_epochs: int = 2
    error_max_examples: int = 50
    semantic_threshold: float = 0.50  # MiniLM cosine (as-is); TF-IDF ablation uses offset in matcher
    matcher_backend: str = "minilm"  # minilm | tfidf
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    error_critical_only: bool = True
    braingpt_model: str = "BrainGPT/BrainGPT-7B-v0.2"
    benchmark_lora_year_max: int | None = None  # compare: fixed LoRA adapter for all benchmark years
    context_year_max: int | None = None  # override context horizon (default: target_year - 1)
    training_prompt_k: int = 1  # k in training prompts (one completion per example)
    # Inference: iterative (k=1 × N, aligned with train) or batch (one shot for N lines).
    generation_mode: str = "iterative"  # iterative | batch
    filter_outputs: bool = True
    reject_context_copies: bool = True  # drop preds that recycle Prior themes / few-shots
    max_generation_attempts_factor: float = 2.0
    llm: CausalLMConfig = field(default_factory=CausalLMConfig)


def _questions_by_year(conn: sqlite3.Connection, year_max: int) -> dict[int, list[tuple[float, str]]]:
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
    by_year: dict[int, list[tuple[float, str]]] = {}
    for yr, items in buckets.items():
        items.sort(key=lambda x: (-x[0], x[1]))
        by_year[yr] = items
    return by_year


def _training_examples_for_year(
    scored_questions: list[tuple[float, str]],
    cfg: LiteratureLoraConfig,
    rng: random.Random,
) -> list[str]:
    if not scored_questions:
        return []
    n_use = max(1, int(len(scored_questions) * cfg.train_fraction))
    if cfg.max_examples_per_year > 0:
        n_use = min(n_use, cfg.max_examples_per_year)
    n_use = min(n_use, len(scored_questions))

    if cfg.train_sampling == "impact_weighted":
        weights = [max(score, 0.0) + 1e-6 for score, _ in scored_questions]
        available = list(range(len(scored_questions)))
        indices: list[int] = []
        while len(indices) < n_use and available:
            pick_weights = [weights[i] for i in available]
            pick = rng.choices(available, weights=pick_weights, k=1)[0]
            indices.append(pick)
            available.remove(pick)
        return [scored_questions[i][1] for i in indices]

    return [text for _, text in scored_questions[:n_use]]


def _fetch_context_question_rows(
    conn: sqlite3.Connection,
    context_year: int,
    max_q: int,
) -> list[sqlite3.Row]:
    """Same selection as build_context_summary — top-impact questions in the 5-year window."""
    year_floor = max(context_year - 5, 0)
    rows = conn.execute(
        """
        SELECT question_text, impact_score, year FROM questions
        WHERE year IS NOT NULL AND year <= ? AND year >= ?
        ORDER BY COALESCE(impact_score, 0) DESC, year DESC
        LIMIT ?
        """,
        (context_year, year_floor, max_q),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT question_text, impact_score, year FROM questions WHERE year <= ? LIMIT ?",
            (context_year, max_q),
        ).fetchall()
    return rows


def list_context_questions(
    conn: sqlite3.Connection,
    target_year: int,
    cfg: LiteratureLoraConfig,
    *,
    context_year: int | None = None,
) -> list[str]:
    """Question texts shown in the generation prompt CONTEXT block for target_year."""
    ctx_year = context_year if context_year is not None else resolve_context_year(target_year, cfg)
    rows = _fetch_context_question_rows(conn, ctx_year, cfg.max_context_questions)
    return [
        (r["question_text"] or "").strip()
        for r in rows
        if (r["question_text"] or "").strip()
    ]


def build_context_summary(conn: sqlite3.Connection, context_year: int, max_q: int) -> str:
    """Top-impact question texts on or before context_year (no per-line years/numbers)."""
    rows = _fetch_context_question_rows(conn, context_year, max_q)
    lines = []
    for r in rows:
        text = (r["question_text"] or "").strip()
        if text:
            lines.append(text[:280])
    return "\n".join(lines)


def resolve_context_year(target_year: int, cfg: LiteratureLoraConfig) -> int:
    if cfg.context_year_max is not None:
        return cfg.context_year_max
    return target_year - 1


def build_generation_prompt(
    conn: sqlite3.Connection,
    target_year: int,
    cfg: LiteratureLoraConfig,
    *,
    k: int,
    context_year: int | None = None,
    already: list[str] | None = None,
) -> str:
    """Shared train/predict prompt: year N themes → continue year N+1.

    Identical for literature_lora, mistral_base, and braingpt.
    Header shows context year N only; article lines have no year tags.
    ``k`` kept for API parity (iterative generation always asks one next line).
    """
    del k
    ctx_year = context_year if context_year is not None else resolve_context_year(target_year, cfg)
    if ctx_year >= target_year:
        raise ValueError(
            f"context_year ({ctx_year}) must be < target_year ({target_year})"
        )
    context = build_context_summary(conn, ctx_year, cfg.max_context_questions)
    parts = [f"Year {ctx_year}:", context] if context else [f"Year {ctx_year}:"]
    if already:
        listed = "\n".join(a.strip() for a in already[-15:] if (a or "").strip())
        if listed:
            parts.append("Already listed:")
            parts.append(listed)
    parts.append(f"Year {target_year}:")
    return "\n".join(parts) + "\n"


def build_temporal_training_prompt(
    conn: sqlite3.Connection,
    year: int,
    cfg: LiteratureLoraConfig,
) -> str:
    """Training prompt aligned with inference format for year+1."""
    target = year + 1
    return build_generation_prompt(
        conn,
        target,
        cfg,
        k=cfg.training_prompt_k,
        context_year=year,
    )


def build_temporal_examples(
    conn: sqlite3.Connection,
    year_max: int,
    cfg: LiteratureLoraConfig,
) -> list[tuple[str, str]]:
    """(prompt, completion) pairs: context ≤ T → questions at T+1, T+1 ≤ year_max."""
    by_year = _questions_by_year(conn, year_max)
    rng = random.Random(cfg.llm.seed)
    examples: list[tuple[str, str]] = []
    per_target_year: dict[int, int] = {}
    for year in sorted(by_year):
        if year + 1 > year_max:
            break
        prompt = build_temporal_training_prompt(conn, year, cfg)
        target_year = year + 1
        selected = _training_examples_for_year(by_year.get(target_year, []), cfg, rng)
        per_target_year[target_year] = len(selected)
        for q in selected:
            examples.append((prompt, q[:400]))
    cap_note = (
        f"max_examples_per_year={cfg.max_examples_per_year}"
        if cfg.max_examples_per_year > 0
        else "max_examples_per_year=0 (no cap)"
    )
    logger.info(
        "Training dataset year_max=%d: %d examples, sampling=%s, %s, fraction=%.2f — by target year: %s",
        year_max,
        len(examples),
        cfg.train_sampling,
        cap_note,
        cfg.train_fraction,
        per_target_year,
    )
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
        matcher = make_matcher(
            refs,
            cfg.semantic_threshold,
            backend=cfg.matcher_backend,
            model_name=cfg.embed_model,
        )
        missed = matcher.uncovered_references(preds, k=eval_k)

    if not missed:
        logger.info("All ground-truth questions covered for year %d — no error examples", target_year)
        return []

    prompt = build_generation_prompt(
        conn, target_year, cfg, k=cfg.training_prompt_k
    )
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


def _bnb_config(cfg: LiteratureLoraConfig):
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.float16
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
    except Exception:
        pass
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _load_train_model(cfg: LiteratureLoraConfig, continue_from: Path | None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict = {}
    if cfg.use_4bit:
        try:
            load_kwargs["quantization_config"] = _bnb_config(cfg)
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
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_cfg)

    model.train()
    return model, tokenizer, torch


def _prepare_training_examples(
    examples: list[tuple[str, str]],
    cfg: LiteratureLoraConfig,
) -> list[tuple[str, str]]:
    prepared = list(examples)
    if cfg.shuffle_training_examples and len(prepared) > 1:
        rng = random.Random(cfg.llm.seed)
        rng.shuffle(prepared)
    return prepared


def _find_latest_checkpoint(checkpoint_dir: Path) -> str | None:
    if not checkpoint_dir.is_dir():
        return None
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return str(checkpoints[-1]) if checkpoints else None


def _resolve_resume_checkpoint(cfg: LiteratureLoraConfig, year_max: int) -> str | None:
    if cfg.resume_from_checkpoint:
        path = Path(cfg.resume_from_checkpoint)
        if path.is_dir():
            return str(path)
        logger.warning("resume_from_checkpoint not found: %s", path)
        return None
    return _find_latest_checkpoint(_adapter_path(cfg, year_max) / "checkpoints")


def _split_train_val(
    examples: list[tuple[str, str]],
    val_fraction: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Random split; examples must already exclude future test years (≤ year_max)."""
    if val_fraction <= 0.0 or len(examples) < 10:
        return examples, []
    frac = min(0.5, max(0.0, float(val_fraction)))
    n_val = max(1, int(round(len(examples) * frac)))
    n_val = min(n_val, len(examples) // 2)
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    val_idx = set(order[:n_val])
    train = [examples[i] for i in order if i not in val_idx]
    val = [examples[i] for i in order if i in val_idx]
    return train, val


class _PromptCompletionDataset:
    def __init__(
        self,
        examples: list[tuple[str, str]],
        tokenizer,
        *,
        max_length: int = 1024,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        """Pack prompt+completion so the completion is never truncated away (fixes loss=0)."""
        prompt, completion = self.examples[idx]
        completion = (completion or "").strip()
        # Reserve room for completion tokens; left-truncate the prompt if needed.
        comp_ids = self.tokenizer(
            " " + completion,
            truncation=True,
            max_length=max(32, self.max_length // 4),
            add_special_tokens=False,
        )["input_ids"]
        budget = max(16, self.max_length - len(comp_ids))
        trunc_side = getattr(self.tokenizer, "truncation_side", "right")
        self.tokenizer.truncation_side = "left"
        try:
            prompt_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=budget,
                add_special_tokens=True,
            )["input_ids"]
        finally:
            self.tokenizer.truncation_side = trunc_side

        input_ids = prompt_ids + comp_ids
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            prompt_len = max(0, len(prompt_ids) - overflow)
        else:
            prompt_len = len(prompt_ids)

        labels = list(input_ids)
        labels[:prompt_len] = [-100] * prompt_len
        if all(x == -100 for x in labels):
            # Safety: if masking wiped everything, supervise last tokens.
            keep = min(16, len(labels))
            labels[-keep:] = input_ids[-keep:]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class _PromptMaskCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        import torch

        labels = [f["labels"] for f in features]
        batch = self.tokenizer.pad(
            {k: [f[k] for f in features] for k in ("input_ids", "attention_mask")},
            padding=True,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for row in labels:
            padded = row + [-100] * (max_len - len(row))
            padded_labels.append(padded[:max_len])
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def _train_with_hf_trainer(
    model,
    tokenizer,
    examples: list[tuple[str, str]],
    *,
    cfg: LiteratureLoraConfig,
    year_max: int,
    epochs: int,
    lr: float,
) -> None:
    import torch
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    checkpoint_dir = _adapter_path(cfg, year_max) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume = _resolve_resume_checkpoint(cfg, year_max)
    n_accum = max(1, cfg.grad_accumulation_steps)
    max_len = max(256, int(cfg.train_max_length))

    train_ex, val_ex = _split_train_val(
        examples, cfg.train_val_fraction, cfg.llm.seed + year_max
    )
    use_val = len(val_ex) >= 2
    if use_val:
        # Need periodic saves to pick best checkpoint (HF: save_steps % eval_steps == 0).
        eval_steps = (
            cfg.train_eval_steps
            if cfg.train_eval_steps > 0
            else (cfg.train_save_steps if cfg.train_save_steps > 0 else 50)
        )
        eval_steps = max(1, eval_steps)
        save_steps = cfg.train_save_steps if cfg.train_save_steps > 0 else eval_steps
        save_steps = max(eval_steps, save_steps)
        if save_steps % eval_steps != 0:
            save_steps = ((save_steps + eval_steps - 1) // eval_steps) * eval_steps
    else:
        save_steps = cfg.train_save_steps if cfg.train_save_steps > 0 else max(10_000_000, len(examples))
        eval_steps = save_steps

    args_kwargs: dict = dict(
        output_dir=str(checkpoint_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=n_accum,
        learning_rate=lr,
        logging_steps=max(1, cfg.train_log_every // n_accum),
        save_steps=save_steps,
        save_total_limit=max(1, cfg.train_save_total_limit),
        optim="paged_adamw_8bit",
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    # transformers ≥4.41 uses eval_strategy; older uses evaluation_strategy.
    if use_val:
        args_kwargs.update(
            {
                "eval_strategy": "steps",
                "eval_steps": eval_steps,
                "save_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
                "per_device_eval_batch_size": 1,
            }
        )
    try:
        training_args = TrainingArguments(**args_kwargs)
    except TypeError:
        if use_val:
            args_kwargs.pop("eval_strategy", None)
            args_kwargs["evaluation_strategy"] = "steps"
        training_args = TrainingArguments(**args_kwargs)

    train_ds = _PromptCompletionDataset(train_ex, tokenizer, max_length=max_len)
    eval_ds = (
        _PromptCompletionDataset(val_ex, tokenizer, max_length=max_len) if use_val else None
    )
    callbacks = []
    if use_val and cfg.train_early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=max(1, cfg.train_early_stopping_patience)
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_PromptMaskCollator(tokenizer),
        callbacks=callbacks or None,
    )
    logger.info(
        "HF Trainer: train=%d val=%d epochs=%d lr=%g grad_accum=%d "
        "save_steps=%d eval_steps=%s early_stop=%s resume=%s",
        len(train_ex),
        len(val_ex),
        epochs,
        lr,
        n_accum,
        save_steps,
        eval_steps if use_val else "off",
        cfg.train_early_stopping_patience if use_val else "off",
        resume or "none",
    )
    trainer.train(resume_from_checkpoint=resume)
    if use_val and getattr(trainer.state, "best_metric", None) is not None:
        logger.info(
            "Best checkpoint: step=%s eval_loss=%s",
            getattr(trainer.state, "best_global_step", None),
            trainer.state.best_metric,
        )


def _train_on_examples(
    model,
    tokenizer,
    torch,
    examples: list[tuple[str, str]],
    *,
    epochs: int,
    lr: float,
    use_4bit: bool,
    grad_accumulation_steps: int = 1,
    log_every: int = 50,
) -> None:
    del use_4bit  # device_map / 4-bit: always move batches via move_inputs_to_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n_ex = len(examples)
    n_accum = max(1, grad_accumulation_steps)
    log_every = max(1, log_every)
    optimizer.zero_grad()
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0.0
        accum_loss = 0.0
        for i, (prompt, completion) in enumerate(examples, start=1):
            text = prompt + " " + completion
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=768)
            inputs = move_inputs_to_model(inputs, model)
            labels = inputs["input_ids"].clone()
            prompt_len = len(tokenizer(prompt, truncation=True, max_length=768)["input_ids"])
            labels[0, :prompt_len] = -100
            out = model(**inputs, labels=labels)
            (out.loss / n_accum).backward()
            accum_loss += float(out.loss.item())
            if i % n_accum == 0 or i == n_ex:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                total_loss += accum_loss
                if global_step % log_every == 0 or i == n_ex:
                    logger.info(
                        "LoRA epoch %d/%d — example %d/%d (step %d), running avg loss=%.4f",
                        epoch + 1,
                        epochs,
                        i,
                        n_ex,
                        global_step,
                        total_loss / max(global_step, 1),
                    )
                accum_loss = 0.0
        logger.info(
            "LoRA epoch %d/%d — %d examples, %d optimizer steps, avg loss=%.4f",
            epoch + 1,
            epochs,
            n_ex,
            global_step,
            total_loss / max(global_step, 1),
        )


def _run_training(
    model,
    tokenizer,
    torch,
    examples: list[tuple[str, str]],
    *,
    cfg: LiteratureLoraConfig,
    year_max: int,
    epochs: int,
    lr: float,
) -> None:
    prepared = _prepare_training_examples(examples, cfg)
    if cfg.use_hf_trainer:
        try:
            _train_with_hf_trainer(
                model,
                tokenizer,
                prepared,
                cfg=cfg,
                year_max=year_max,
                epochs=epochs,
                lr=lr,
            )
            return
        except Exception as exc:
            logger.warning("HF Trainer failed (%s) — falling back to manual loop", exc)
    _train_on_examples(
        model,
        tokenizer,
        torch,
        prepared,
        epochs=epochs,
        lr=lr,
        use_4bit=cfg.use_4bit,
        grad_accumulation_steps=cfg.grad_accumulation_steps,
        log_every=cfg.train_log_every,
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
        logger.info(
            "Literature LoRA training: %d examples, %d epoch(s), lr=%g, 4bit=%s, "
            "sampling=%s, grad_accum=%d, hf_trainer=%s",
            len(examples),
            cfg.train_epochs,
            cfg.train_lr,
            cfg.use_4bit,
            cfg.train_sampling,
            cfg.grad_accumulation_steps,
            cfg.use_hf_trainer,
        )
        _run_training(
            model,
            tokenizer,
            torch,
            examples,
            cfg=cfg,
            year_max=year_max,
            epochs=cfg.train_epochs,
            lr=cfg.train_lr,
        )
        out_dir = _save_adapter(model, tokenizer, cfg, year_max)
        logger.info("Literature LoRA saved: %s (%d examples)", out_dir, len(examples))
        del model, tokenizer
        release_gpu_memory()
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
        _run_training(
            peft_model,
            tokenizer,
            torch,
            examples,
            cfg=cfg,
            year_max=target_year,
            epochs=cfg.error_train_epochs,
            lr=cfg.train_lr,
        )
        out_dir = _save_adapter(peft_model, tokenizer, cfg, target_year)
        logger.info(
            "Error-correction LoRA saved: %s (%d examples for year %d)",
            out_dir,
            len(examples),
            target_year,
        )
        del peft_model, tokenizer
        release_gpu_memory()
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
        tokens_per_direction=cfg.llm.tokens_per_direction,
        prompt_max_length=cfg.llm.prompt_max_length,
        num_return_sequences=cfg.llm.num_return_sequences,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        seed=cfg.llm.seed,
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
        # BrainGPT on HF is a PEFT adapter on Mistral-7B-v0.1 (adapter_config.json + weights).
        llm_cfg.base_model = cfg.base_model
        llm_cfg.adapter_path = cfg.braingpt_model
        logger.info(
            "Using BrainGPT adapter %s on base %s",
            cfg.braingpt_model,
            cfg.base_model,
        )
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
    """Generate novel research directions for year N with a literature LM variant."""
    year_max = N - 1
    context_year = resolve_context_year(N, cfg)
    llm_cfg = resolve_literature_llm_cfg(cfg, year_max, model)
    mode = (cfg.generation_mode or "iterative").lower().strip()

    context_qs = list_context_questions(conn, N, cfg, context_year=context_year)
    blocklist: set[str] | None = None
    if cfg.reject_context_copies:
        blocklist = build_context_blocklist(context_qs)

    audit = GenerationAudit(
        model=model,
        target_year=N,
        requested_k=k,
        generation_mode=mode,
        filter_outputs=cfg.filter_outputs,
        adapter=llm_cfg.adapter_path,
        temperature=llm_cfg.temperature,
        context_year=context_year,
        context_lines=len(context_qs),
    )

    try:
        if mode == "batch":
            prompt = build_generation_prompt(
                conn, N, cfg, k=k, context_year=context_year
            )
            audit.prompt_chars = len(prompt)
            oversample = 1 if cfg.llm.temperature <= 0.0 else 3
            return generate_directions_batch(
                prompt,
                llm_cfg,
                k,
                oversample=oversample,
                apply_filter=cfg.filter_outputs,
                audit=audit,
                blocklist=blocklist,
            )

        def _builder(n_req: int, already: list[str]) -> str:
            return build_generation_prompt(
                conn,
                N,
                cfg,
                k=n_req,
                context_year=context_year,
                already=already,
            )

        # Prime prompt stats from first builder call.
        first_prompt = _builder(1, [])
        audit.prompt_chars = len(first_prompt)

        return generate_directions_iterative(
            _builder,
            llm_cfg,
            k,
            apply_filter=cfg.filter_outputs,
            attempts_factor=cfg.max_generation_attempts_factor,
            audit=audit,
            blocklist=blocklist,
        )
    except ImportError as e:
        audit.error = str(e)
        audit.log()
        logger.error("%s: %s — pip install -e '.[train]'", model, e)
        return []
    except Exception as e:
        audit.error = f"{type(e).__name__}: {e}"
        audit.log()
        raise


def make_literature_predictor(cfg: LiteratureLoraConfig, model: str = "literature_lora"):
    def _fn(conn: sqlite3.Connection, N: int, k: int, rng) -> list[tuple[str, float]]:
        return predict_literature_lm(conn, N, k, cfg, model=model)

    return _fn
