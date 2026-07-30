"""Causal LM generation for literature research directions."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from .direction_filter import (
    classify_direction_rejection,
    clamp_direction_words,
    filter_directions,
    filter_directions_audited,
    is_near_duplicate,
    is_valid_direction,
    strip_list_prefix,
)
from .generation_audit import GenerationAudit, truncate_preview

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, tuple[object, object]] = {}


def model_input_device(model) -> object:
    """Device for batch tensors (works with device_map='auto' / 4-bit)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


def move_inputs_to_model(inputs: dict, model) -> dict:
    """Move tokenizer tensors onto the same device as the model weights."""
    device = model_input_device(model)
    return {k: v.to(device) for k, v in inputs.items()}


def release_gpu_memory() -> None:
    """Drop cached LMs and return VRAM to the driver (safe between train/benchmark stages)."""
    global _MODEL_CACHE
    for model, _tokenizer in _MODEL_CACHE.values():
        del model
    _MODEL_CACHE.clear()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


@dataclass
class CausalLMConfig:
    base_model: str = "mistralai/Mistral-7B-v0.1"
    adapter_path: str | None = None  # local lora/ dir or Hub id (BrainGPT/...)
    use_4bit: bool = True
    max_new_tokens: int = 80
    # Tokens budget per single research direction (iterative mode / scaling base).
    tokens_per_direction: int = 48
    # Prompt truncate for generate (V100-32g + 4-bit: 4096 + 2048 new fits Mistral 8k).
    prompt_max_length: int = 4096
    num_return_sequences: int = 1
    temperature: float = 0.0
    top_p: float = 0.9
    seed: int = 42


def _load_model(cfg: CausalLMConfig) -> tuple[object, object]:
    cache_key = f"{cfg.base_model}|{cfg.adapter_path or ''}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError("transformers and torch required — pip install -e '.[train]'") from e

    load_kwargs: dict = {}
    if cfg.use_4bit:
        try:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            load_kwargs["device_map"] = "auto"
        except Exception:
            logger.warning("4-bit unavailable — standard load")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading LM base=%s adapter=%s", cfg.base_model, cfg.adapter_path or "none")
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)

    if cfg.adapter_path:
        from pathlib import Path

        from peft import PeftModel

        adapter_ref = cfg.adapter_path
        adapter_path = Path(adapter_ref)
        if adapter_path.is_dir():
            model = PeftModel.from_pretrained(model, str(adapter_path))
        elif "/" in adapter_ref and not adapter_path.exists():
            model = PeftModel.from_pretrained(model, adapter_ref)
        else:
            raise FileNotFoundError(
                f"LoRA adapter not found (local directory or Hub id): {cfg.adapter_path}"
            )

    model.eval()
    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def parse_generated_directions(text: str, min_len: int = 15) -> list[str]:
    """Parse generated research directions (numbered lines or bullets)."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = strip_list_prefix(raw)
        if len(line) < min_len:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    # Fallback: single-line blob without newlines.
    if not out:
        blob = strip_list_prefix(text.replace("\n", " "))
        if len(blob) >= min_len:
            out.append(blob)
    return out


# Backward-compatible alias
parse_generated_questions = parse_generated_directions


def score_completion(prompt: str, completion: str, cfg: CausalLMConfig) -> float:
    """Score = −loss (higher = more plausible)."""
    import torch

    model, tokenizer = _load_model(cfg)
    text = prompt.rstrip() + "\n" + completion.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=768)
    inputs = move_inputs_to_model(inputs, model)
    prompt_len = len(tokenizer(prompt, truncation=True, max_length=768)["input_ids"])
    labels = inputs["input_ids"].clone()
    labels[0, :prompt_len] = -100
    with torch.no_grad():
        out = model(**inputs, labels=labels)
        return -float(out.loss.item())


def sequence_perplexity(text: str, cfg: CausalLMConfig, *, max_length: int = 2048) -> float:
    """Token-sequence perplexity (BrainBench eq. 1): exp(mean negative log-likelihood)."""
    import math

    import torch

    model, tokenizer = _load_model(cfg)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = move_inputs_to_model(inputs, model)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return math.exp(float(out.loss.item()))


def _prompt_token_stats(prompt: str, tokenizer, max_length: int) -> tuple[int, int, bool]:
    """Return (full_tokens, used_tokens, was_truncated)."""
    full_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    trunc_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        used_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]
    finally:
        tokenizer.truncation_side = trunc_side
    return len(full_ids), len(used_ids), len(full_ids) > len(used_ids)


def _generate_raw(
    prompt: str,
    cfg: CausalLMConfig,
    *,
    max_new_tokens: int,
    n_seq: int,
    do_sample: bool,
    audit: GenerationAudit | None = None,
) -> list[str]:
    import torch

    model, tokenizer = _load_model(cfg)
    max_len = max(512, int(cfg.prompt_max_length))
    if audit is not None:
        full_tok, used_tok, truncated = _prompt_token_stats(prompt, tokenizer, max_len)
        audit.prompt_tokens_full = full_tok
        audit.prompt_tokens_used = used_tok
        audit.prompt_truncated = truncated
        audit.max_new_tokens = max_new_tokens
        audit.num_return_sequences = n_seq
        audit.do_sample = do_sample
    # Keep the end of the prompt (instructions + open "1.") if truncating.
    trunc_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
    finally:
        tokenizer.truncation_side = trunc_side
    inputs = move_inputs_to_model(inputs, model)

    gen_kwargs: dict = {"do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = max(cfg.temperature, 1e-5)
        gen_kwargs["top_p"] = cfg.top_p

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_seq,
            pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs,
        )

    decoded: list[str] = []
    prompt_len = inputs["input_ids"].shape[1]
    for seq in outputs:
        decoded.append(tokenizer.decode(seq[prompt_len:], skip_special_tokens=True))
    if audit is not None:
        audit.set_raw_outputs(decoded)
        combined = "\n---\n".join(decoded)
        audit._raw_preview = truncate_preview(combined)  # noqa: SLF001
    return decoded


def _score_unique(
    prompt: str,
    candidates: list[str],
    cfg: CausalLMConfig,
    *,
    apply_filter: bool,
    audit: GenerationAudit | None = None,
    blocklist: set[str] | None = None,
) -> list[tuple[str, float]]:
    if apply_filter:
        kept, counts, samples = filter_directions_audited(
            candidates, blocklist=blocklist
        )
        if audit is not None:
            for reason, count in counts.items():
                audit.rejection_counts[reason] = (
                    audit.rejection_counts.get(reason, 0) + count
                )
            for reason, text in samples:
                if len(audit.rejected_samples) < 5:
                    audit.rejected_samples.append(
                        {"reason": reason, "text": truncate_preview(text, 200)}
                    )
        candidates = kept
    else:
        cleaned: list[str] = []
        seen: set[str] = set()
        for c in candidates:
            s = strip_list_prefix(c)
            key = s.lower()
            if not s or key in seen:
                continue
            seen.add(key)
            cleaned.append(s)
        candidates = cleaned

    if audit is not None:
        audit.after_filter = len(candidates)

    scored: list[tuple[str, float]] = []
    for q in candidates:
        try:
            s = score_completion(prompt, q, cfg)
        except Exception:
            s = 0.0
        scored.append((q, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def generate_directions_batch(
    prompt: str,
    cfg: CausalLMConfig,
    n: int,
    *,
    oversample: int = 2,
    apply_filter: bool = True,
    audit: GenerationAudit | None = None,
    blocklist: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Batch generation: one (or few) sequences aiming for n numbered directions."""
    import torch

    greedy = cfg.temperature <= 0.0
    if greedy:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        n_seq = 1
        do_sample = False
    else:
        n_seq = min(max(n * oversample, n + 2), max(cfg.num_return_sequences, 8))
        do_sample = True

    # ~tokens_per_direction per requested line, with a floor for short lists.
    per = max(16, int(cfg.tokens_per_direction))
    max_new = min(2048, max(cfg.max_new_tokens, per * max(n, 1)))

    raw_outputs = _generate_raw(
        prompt,
        cfg,
        max_new_tokens=max_new,
        n_seq=n_seq,
        do_sample=do_sample,
        audit=audit,
    )
    candidates: list[str] = []
    for decoded in raw_outputs:
        candidates.extend(parse_generated_directions(decoded))

    if not candidates and raw_outputs:
        candidates = parse_generated_directions(raw_outputs[0])
        if audit is not None:
            audit.parse_fallback_blob = True

    if audit is not None:
        audit.parsed_candidates = len(candidates)

    scored = _score_unique(
        prompt,
        candidates,
        cfg,
        apply_filter=apply_filter,
        audit=audit,
        blocklist=blocklist,
    )
    result = scored[:n]
    if audit is not None:
        audit.returned = len(result)
        for text, _score in result:
            audit.record_kept(text)
        audit.log()
    return result


# Backward-compatible alias
generate_questions = generate_directions_batch


def generate_directions_iterative(
    prompt_builder: Callable[[int, list[str]], str],
    cfg: CausalLMConfig,
    n: int,
    *,
    apply_filter: bool = True,
    attempts_factor: float = 2.0,
    pool_factor: float = 3.0,
    soft_truncate_words: bool = True,
    max_direction_words: int = 25,
    reject_near_duplicates: bool = True,
    near_duplicate_threshold: float = 0.85,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    audit: GenerationAudit | None = None,
    blocklist: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Generate n directions with k=1 prompts (aligned with training_prompt_k=1).

    Collects a candidate pool (``n * pool_factor``), filters / near-dedups, then
    reranks by completion score and returns top-n.
    """
    import torch

    pool_target = max(n, int(n * max(pool_factor, 1.0)))
    max_attempts = max(pool_target, int(pool_target * max(attempts_factor, 1.0)))
    if audit is not None:
        audit.attempts_budget = max_attempts
    per_tokens = max(24, int(cfg.tokens_per_direction))
    # Prefer sampling for diversity when predicting a top-k set.
    greedy = cfg.temperature <= 0.0
    raw_chunks: list[str] = []
    collected: list[tuple[str, float]] = []
    already: list[str] = []

    for attempt in range(max_attempts):
        if len(collected) >= pool_target:
            break
        if audit is not None:
            audit.attempts_used = attempt + 1
        prompt = prompt_builder(1, already)
        if greedy:
            torch.manual_seed(cfg.seed + attempt)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed + attempt)
            n_seq = 1
            do_sample = False
        else:
            n_seq = min(3, max(1, cfg.num_return_sequences))
            do_sample = True

        raw_outputs = _generate_raw(
            prompt,
            cfg,
            max_new_tokens=per_tokens,
            n_seq=n_seq,
            do_sample=do_sample,
            audit=audit if attempt == 0 else None,
        )
        raw_chunks.extend(raw_outputs)
        candidates: list[str] = []
        for decoded in raw_outputs:
            candidates.extend(parse_generated_directions(decoded, min_len=12))

        if audit is not None:
            audit.parsed_candidates += len(candidates)

        for cand in candidates:
            s = strip_list_prefix(cand)
            if soft_truncate_words:
                s = clamp_direction_words(s, max_words=max_direction_words)
            if apply_filter:
                reason = classify_direction_rejection(
                    s,
                    max_words=max_direction_words,
                    blocklist=blocklist,
                )
                if reason is not None:
                    if audit is not None:
                        audit.record_rejection(reason, s)
                    continue
            elif not s:
                if audit is not None:
                    audit.record_rejection("empty", cand)
                continue
            if not s:
                continue
            key = s.lower()
            if key in {a.lower() for a in already}:
                if audit is not None:
                    audit.record_rejection("duplicate", s)
                continue
            if reject_near_duplicates and is_near_duplicate(
                s,
                already,
                threshold=near_duplicate_threshold,
                embed_model=embed_model,
            ):
                if audit is not None:
                    audit.record_rejection("near_duplicate", s)
                continue
            try:
                score = score_completion(prompt, s, cfg)
            except Exception:
                score = 0.0
            collected.append((s, score))
            already.append(s)
            if len(collected) >= pool_target:
                break

    collected.sort(key=lambda x: x[1], reverse=True)
    best: list[tuple[str, float]] = []
    seen: set[str] = set()
    for text, score in collected:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        best.append((text, score))
        if len(best) >= n:
            break
    if audit is not None:
        audit.set_raw_outputs(raw_chunks)
        audit._raw_preview = truncate_preview("\n---\n".join(raw_chunks))  # noqa: SLF001
        audit.after_filter = len(collected)
        audit.returned = len(best)
        for text, _score in best:
            audit.record_kept(text)
        audit.log()
    logger.info(
        "Iterative generation: %d/%d valid directions (pool=%d/%d attempts=%d used=%d, "
        "filter=%s near_dup=%s soft_trunc=%s)",
        len(best),
        n,
        len(collected),
        pool_target,
        max_attempts,
        audit.attempts_used if audit else max_attempts,
        apply_filter,
        reject_near_duplicates,
        soft_truncate_words,
    )
    return best
