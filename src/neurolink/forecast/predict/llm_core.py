"""Causal LM question generation for literature predictors."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

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
    num_return_sequences: int = 1
    temperature: float = 0.7
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
            # Hugging Face Hub adapter id (e.g. BrainGPT/BrainGPT-7B-v0.2)
            model = PeftModel.from_pretrained(model, adapter_ref)
        else:
            raise FileNotFoundError(
                f"LoRA adapter not found (local directory or Hub id): {cfg.adapter_path}"
            )

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
    inputs = move_inputs_to_model(inputs, model)

    greedy = cfg.temperature <= 0.0
    if greedy:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        n_seq = 1
        gen_kwargs = {"do_sample": False}
    else:
        n_seq = min(max(n * oversample, n + 2), 8)
        gen_kwargs = {
            "do_sample": True,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens * max(2, n // 2),
            num_return_sequences=n_seq,
            pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs,
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
