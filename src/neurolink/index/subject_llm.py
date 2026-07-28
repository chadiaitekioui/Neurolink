"""Level-3 research-direction extraction via causal LM (indexing only).

Uses ``mistralai/Mistral-7B-Instruct-v0.2`` (no LoRA) — orthogonal to the literature
LoRA adapter used at predict time.
"""

from __future__ import annotations

import logging
import re

from ..forecast.predict.direction_filter import classify_direction_rejection, strip_list_prefix
from ..forecast.predict.llm_core import CausalLMConfig, _generate_raw, _load_model
from ..utils.pubmed_clean import polish_segment_field, structure_abstract_sections
from .subject import (
    SubjectClassifier,
    SubjectConfig,
    SubjectLlmConfig,
    SubjectResult,
    compress_to_subject_span,
    get_subject_classifier,
    heuristic_subjectness,
)

logger = logging.getLogger(__name__)

_USER_PROMPT = """Summarize this neuroscience paper as ONE research direction (8-25 words, noun phrase, no question mark).

Title: {title}

Abstract:
{abstract}"""


def build_extraction_user_content(
    title: str,
    abstract: str,
    *,
    abstract_max_chars: int = 3500,
) -> str:
    title = polish_segment_field(title or "Untitled")
    body = polish_segment_field(abstract or "")
    if len(body) > abstract_max_chars:
        body = body[: abstract_max_chars - 3].rstrip() + "..."
    return _USER_PROMPT.format(title=title, abstract=body)


def build_extraction_prompt(
    title: str,
    abstract: str,
    *,
    abstract_max_chars: int = 3500,
) -> str:
    """Plain user message (no chat template). Kept for tests and debugging."""
    return build_extraction_user_content(
        title, abstract, abstract_max_chars=abstract_max_chars
    )


def format_extraction_prompt(
    title: str,
    abstract: str,
    *,
    llm: SubjectLlmConfig,
) -> str:
    """Build the full model input (chat template when enabled)."""
    user = build_extraction_user_content(
        title,
        abstract,
        abstract_max_chars=llm.abstract_max_chars,
    )
    if not llm.use_chat_template:
        return user + "\n\nDirection:"

    causal = _llm_cfg_to_causal(llm)
    _, tokenizer = _load_model(causal)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        logger.warning("Tokenizer has no chat template — using plain prompt")
        return user + "\n\nDirection:"

    messages = [{"role": "user", "content": user}]
    try:
        return apply(messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        logger.warning("apply_chat_template failed (%s) — using plain prompt", exc)
        return user + "\n\nDirection:"


def parse_llm_direction(raw: str) -> str | None:
    """Take the first non-empty line from model output."""
    if not raw or not raw.strip():
        return None
    for line in raw.splitlines():
        cleaned = strip_list_prefix(line)
        cleaned = re.sub(r"^(?:direction|research direction)\s*:\s*", "", cleaned, flags=re.I)
        cleaned = cleaned.strip(" \"'")
        if not cleaned or cleaned.lower().startswith(("rules:", "note:")):
            continue
        if re.match(r"^what is the main research direction", cleaned, re.I):
            continue
        if re.fullmatch(r"[-\s]+", cleaned):
            continue
        return cleaned
    blob = strip_list_prefix(raw.replace("\n", " "))
    return blob.strip() or None


def validate_llm_direction(
    text: str,
    *,
    min_words: int = 8,
    max_words: int = 25,
) -> str | None:
    """Return cleaned span or None if rejected by shared direction filters."""
    span, reason = diagnose_llm_direction(text, min_words=min_words, max_words=max_words)
    if reason is not None:
        logger.debug("LLM direction rejected (%s): %s", reason, (span or text)[:80])
        return None
    return span


def diagnose_llm_direction(
    text: str,
    *,
    min_words: int = 8,
    max_words: int = 25,
) -> tuple[str | None, str | None]:
    """Return (best-effort text, reject_reason). Text is kept even when rejected."""
    if not text or not text.strip():
        return None, "empty"
    span = compress_to_subject_span(
        text,
        min_words=min_words,
        max_words=max_words,
        absolute_min_words=min_words,
    )
    candidate = span or strip_list_prefix(text)
    if not candidate:
        return None, "empty"
    reason = classify_direction_rejection(
        candidate,
        min_words=min_words,
        max_words=max_words,
    )
    if reason is not None:
        return candidate, reason
    return span or candidate, None


def _llm_cfg_to_causal(llm: SubjectLlmConfig) -> CausalLMConfig:
    return CausalLMConfig(
        base_model=llm.base_model,
        adapter_path=None,
        use_4bit=llm.use_4bit,
        max_new_tokens=llm.max_new_tokens,
        prompt_max_length=llm.prompt_max_length,
        temperature=llm.temperature,
        num_return_sequences=1,
    )


def extract_subject_llm(
    title: str | None,
    abstract: str,
    *,
    cfg: SubjectConfig,
    classifier: SubjectClassifier | None = None,
) -> SubjectResult | None:
    """Extract one research direction with an instruct LM (no adapter)."""
    llm = cfg.llm
    if not (abstract or "").strip() and not (title or "").strip():
        return None

    prompt = format_extraction_prompt(title or "", abstract or "", llm=llm)
    causal = _llm_cfg_to_causal(llm)
    try:
        decoded = _generate_raw(
            prompt,
            causal,
            max_new_tokens=llm.max_new_tokens,
            n_seq=1,
            do_sample=llm.temperature > 0,
        )
    except Exception as exc:
        logger.warning("LLM subject extraction failed: %s", exc)
        return None

    raw = decoded[0] if decoded else ""
    parsed = parse_llm_direction(raw)
    if not parsed:
        logger.info("LLM subject rejected (empty_parse): raw=%r", raw[:160])
        return None

    span, fmt_reason = diagnose_llm_direction(
        parsed,
        min_words=cfg.min_words,
        max_words=cfg.max_words,
    )
    if not span:
        logger.info("LLM subject rejected (empty_parse): parsed=%r", parsed[:160])
        return None

    if fmt_reason is not None:
        logger.info("LLM subject rejected (%s): %s", fmt_reason, span)
        return SubjectResult(
            text=span,
            subjectness=0.0,
            label="subject",
            source_section="LLM",
            reject_reason=fmt_reason,
        )

    score = max(heuristic_subjectness(span), llm.llm_subjectness_floor)
    label = "subject"
    clf = classifier
    if cfg.use_level2_classifier and clf is None:
        clf = get_subject_classifier(cfg.classifier_model)
    if clf is not None and cfg.use_level2_classifier:
        label, conf = clf.classify(span)
        if label == "noise":
            score *= 0.15
        elif label == "methods":
            score *= 0.35
        else:
            score = min(1.0, score + 0.15 * conf)
    if score < cfg.min_subjectness:
        reason = f"low_subjectness:{label}"
        logger.info("LLM subject rejected (%s, score=%.2f): %s", reason, score, span)
        return SubjectResult(
            text=span,
            subjectness=score,
            label=label,
            source_section="LLM",
            reject_reason=reason,
        )
    if label != "subject" and score < cfg.min_subjectness + 0.15:
        reason = f"low_subjectness:{label}"
        logger.info("LLM subject rejected (%s, score=%.2f): %s", reason, score, span)
        return SubjectResult(
            text=span,
            subjectness=score,
            label=label,
            source_section="LLM",
            reject_reason=reason,
        )

    return SubjectResult(
        text=span,
        subjectness=score,
        label=label,
        source_section="LLM",
    )


def results_from_sections(text: str) -> str:
    """Rule-based results bucket when PubMedBERT segmentation is skipped."""
    parts: list[str] = []
    for content, bucket, section in structure_abstract_sections(text):
        sec = (section or "").upper()
        if bucket == "results" or sec in ("RESULTS", "CONCLUSIONS"):
            parts.append(content)
    return polish_segment_field(" ".join(parts))
