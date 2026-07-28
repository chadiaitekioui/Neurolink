"""Level-3 research-direction extraction via causal LM (indexing only).

Uses ``mistralai/Mistral-7B-v0.1`` **base** (no LoRA) — same infra as forecast but
orthogonal to the literature LoRA adapter used at predict time.
"""

from __future__ import annotations

import logging
import re

from ..forecast.predict.direction_filter import classify_direction_rejection, strip_list_prefix
from ..forecast.predict.llm_core import CausalLMConfig, _generate_raw
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


_EXTRACTION_PROMPT = """Extract the main neuroscience research direction from this paper.

Title: {title}

Abstract:
{abstract}

Write exactly ONE research direction (8-25 words, noun phrase).
Rules: no question mark; no "We aimed" or methods-only text; grounded in title/abstract only.
Direction:"""


def build_extraction_prompt(
    title: str,
    abstract: str,
    *,
    abstract_max_chars: int = 3500,
) -> str:
    title = polish_segment_field(title or "Untitled")
    body = polish_segment_field(abstract or "")
    if len(body) > abstract_max_chars:
        body = body[: abstract_max_chars - 3].rstrip() + "..."
    return _EXTRACTION_PROMPT.format(title=title, abstract=body)


def parse_llm_direction(raw: str) -> str | None:
    """Take the first non-empty line from model output."""
    if not raw or not raw.strip():
        return None
    for line in raw.splitlines():
        cleaned = strip_list_prefix(line)
        cleaned = re.sub(r"^(?:direction|research direction)\s*:\s*", "", cleaned, flags=re.I)
        cleaned = cleaned.strip(" \"'")
        if cleaned and not cleaned.lower().startswith(("rules:", "note:")):
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
    span = compress_to_subject_span(
        text,
        min_words=min_words,
        max_words=max_words,
        absolute_min_words=min_words,
    )
    if span is None:
        span = strip_list_prefix(text)
    if not span:
        return None
    reason = classify_direction_rejection(
        span,
        min_words=min_words,
        max_words=max_words,
    )
    if reason is not None:
        logger.debug("LLM direction rejected (%s): %s", reason, span[:80])
        return None
    return span


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
    """Extract one research direction with Mistral base (no adapter)."""
    llm = cfg.llm
    if not (abstract or "").strip() and not (title or "").strip():
        return None

    prompt = build_extraction_prompt(
        title or "",
        abstract or "",
        abstract_max_chars=llm.abstract_max_chars,
    )
    causal = _llm_cfg_to_causal(llm)
    try:
        decoded = _generate_raw(
            prompt,
            causal,
            max_new_tokens=llm.max_new_tokens,
            do_sample=llm.temperature > 0,
        )
    except Exception as exc:
        logger.warning("LLM subject extraction failed: %s", exc)
        return None

    raw = decoded[0] if decoded else ""
    parsed = parse_llm_direction(raw)
    if not parsed:
        logger.debug("LLM subject extraction: empty parse from %r", raw[:120])
        return None

    span = validate_llm_direction(
        parsed,
        min_words=cfg.min_words,
        max_words=cfg.max_words,
    )
    if span is None:
        return None

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
        return None
    if label != "subject" and score < cfg.min_subjectness + 0.15:
        return None

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
