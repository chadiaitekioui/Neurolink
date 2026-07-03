"""Causal LM question generation (shared by literature and centroid predictors)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, tuple[object, object]] = {}


@dataclass
class CausalLMConfig:
    base_model: str = "mistralai/Mistral-7B-v0.1"
    adapter_path: str | None = None
    use_4bit: bool = True
    max_new_tokens: int = 80
    num_return_sequences: int = 1
    temperature: float = 0.7
    top_p: float = 0.9


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

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, **load_kwargs)

    if cfg.adapter_path:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, cfg.adapter_path)
        except Exception as e:
            logger.warning("LoRA adapter not loaded (%s)", e)

    model.eval()
    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def parse_generated_questions(text: str, min_len: int = 25) -> list[str]:
    """Parse generated questions (numbered lines or bullets)."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\.\):\-]\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line)
        if len(line) < min_len:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def score_completion(prompt: str, completion: str, cfg: CausalLMConfig) -> float:
    """Score = −loss (higher = more plausible)."""
    import torch

    model, tokenizer = _load_model(cfg)
    text = prompt.rstrip() + "\n" + completion.strip()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=768)
    if not cfg.use_4bit:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
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
    if not cfg.use_4bit:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return math.exp(float(out.loss.item()))


def generate_questions(
    prompt: str,
    cfg: CausalLMConfig,
    n: int,
    *,
    oversample: int = 2,
) -> list[tuple[str, float]]:
    """Generate n novel questions (no corpus recycling)."""
    import torch

    model, tokenizer = _load_model(cfg)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    if not cfg.use_4bit:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    n_seq = min(max(n * oversample, n + 2), 8)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens * max(2, n // 2),
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            num_return_sequences=n_seq,
            pad_token_id=tokenizer.pad_token_id,
        )

    candidates: list[str] = []
    for seq in outputs:
        decoded = tokenizer.decode(seq[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        candidates.extend(parse_generated_questions(decoded))

    if not candidates:
        full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        candidates = parse_generated_questions(full[len(prompt) :])

    scored: list[tuple[str, float]] = []
    seen: set[str] = set()
    for q in candidates:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            s = score_completion(prompt, q, cfg)
        except Exception:
            s = 0.0
        scored.append((q, s))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]
