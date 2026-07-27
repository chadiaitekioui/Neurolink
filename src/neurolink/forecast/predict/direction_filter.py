"""Post-generation filters and format-compliance metrics for research directions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...utils.pubmed_clean import is_junk_sentence

_DOI = re.compile(r"(?i)(?:\bdoi\s*:|\b10\.\d{4,}/)")
_PMID = re.compile(r"(?i)\b(?:pmid|pmcid)\s*:")
_YEAR_PREFIX = re.compile(r"^\[\d{4}\]")
_NUMBER_ONLY = re.compile(r"^[\d.#\s]+$")
_DEGENERATE = re.compile(r"^(?:1\.){2,}|(?:#+\s*){2,}|(?:\d+\.){4,}")
_AUTHOR_HEAVY = re.compile(r"(?:\([#\d]+\)){2,}|\bCollaborators?\s*:", re.IGNORECASE)
_COMMENT = re.compile(r"\bComment in\b|\bContributed equally\b", re.IGNORECASE)
_HERE_WE = re.compile(
    r"\b(?:here,?\s+we|we (?:found|show(?:ed)?|demonstrate(?:d)?|used))\b",
    re.IGNORECASE,
)
# Prompt / rubric leakage seen in Job 2 v3–v4 generations.
_PROMPT_LEAK = re.compile(
    r"(?i)\b(?:"
    r"TASK\s*:|"
    r"OUTPUT\s+FORMAT|"
    r"CONSTRAINTS\s*:|"
    r"ALREADY\s+PROPOSED|"
    r"Propose\s+exactly|"
    r"Propose\s+\d+\s+research|"
    r"research\s+directions?\s+likely\s+to\s+be\s+studied|"
    r"\d+\s+points?\s+for\s+each|"
    r"points?\s+for\s+each\s+direction|"
    r"noun-phrase\s+style|"
    r"Do\s+NOT\s+(?:copy|include|repeat)|"
    r"Each\s+direction\s*:\s*one\s+line|"
    r"Mix\s+extensions\s+of\s+existing|"
    r"year\s+tags\s+like|"
    r"no\s+preamble|"
    r"<research\s+direction>|"
    r"Style\s+examples|"
    r"Prior\s+themes|"
    r"Already\s+listed|"
    r"Numbered\s+list\s+only|"
    r"published\s+in\s+20\d{2}\b|"
    r"not\s+published|"
    r"any\s+other\s+participant|"
    r"plausible,?\s+and\s+interesting|"
    r"direct\s+extension\s+of\s+a\s+direction\s+in\s+CONTEXT"
    r")\b"
)
_INSTRUCTION_LINE = re.compile(
    r"(?i)^(CONTEXT|Prior themes|Write (?:exactly|up to)|Numbered list only|Neuroscience forecast)\b"
)
# Exact copies of prompt few-shots (must stay in sync with literature_lora._STYLE_EXAMPLES).
_STYLE_FEWSHOT_BLOCKLIST = frozenset(
    {
        "role of cerebellar nuclei in top-down motor cortex control",
        "microglial modulation of synaptic pruning during development",
        "prefrontal dopamine signaling in flexible decision making",
    }
)


@dataclass
class FormatCompliance:
    n: int
    frac_numbered_raw: float  # reserved; set by caller if raw lines available
    frac_valid: float
    frac_noise: float
    frac_with_question_mark: float
    frac_with_doi: float
    frac_year_prefix: float
    frac_word_len_ok: float  # 8–25 words
    mean_words: float


def strip_list_prefix(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\d+[\.\):\-]\s*", "", line)
    line = re.sub(r"^[-*•]\s*", "", line)
    if line.endswith("?"):
        line = line[:-1].rstrip()
    return line.strip()


def is_prompt_leak(text: str) -> bool:
    """True if the line is instruction / scoring-rubric text rather than a direction."""
    s = strip_list_prefix(text)
    if not s:
        return True
    if s.lower() in _STYLE_FEWSHOT_BLOCKLIST:
        return True
    if _PROMPT_LEAK.search(s):
        return True
    if _INSTRUCTION_LINE.match(s):
        return True
    return False


def is_noise_direction(text: str) -> bool:
    """True if the line looks like metadata / corpus junk / prompt leakage."""
    return classify_direction_rejection(text) in _NOISE_REASONS


def classify_direction_rejection(
    text: str,
    *,
    min_words: int = 8,
    max_words: int = 25,
    min_chars: int = 15,
) -> str | None:
    """Return a rejection reason code, or None if the direction passes noise + length checks."""
    s = strip_list_prefix(text)
    if not s:
        return "empty"
    if len(s) < min_chars:
        return "too_short_chars"
    if s.lower() in _STYLE_FEWSHOT_BLOCKLIST:
        return "fewshot_copy"
    if _PROMPT_LEAK.search(s):
        return "prompt_leak"
    if _INSTRUCTION_LINE.match(s):
        return "instruction_line"
    if _YEAR_PREFIX.match(s):
        return "year_prefix"
    if _DOI.search(s) or _PMID.search(s):
        return "doi_pmid"
    if _COMMENT.search(s):
        return "comment_in"
    if _AUTHOR_HEAVY.search(s):
        return "author_heavy"
    if _DEGENERATE.search(s):
        return "degenerate"
    if _NUMBER_ONLY.match(s):
        return "number_only"
    if is_junk_sentence(s):
        return "junk_sentence"
    if _HERE_WE.search(s) and len(s.split()) > 30:
        return "narrative_here_we"
    n_words = len(s.split())
    if n_words < min_words:
        return "too_few_words"
    if n_words > max_words:
        return "too_many_words"
    if s.endswith("?"):
        return "question_mark"
    return None


_NOISE_REASONS = frozenset(
    {
        "empty",
        "too_short_chars",
        "fewshot_copy",
        "prompt_leak",
        "instruction_line",
        "year_prefix",
        "doi_pmid",
        "comment_in",
        "author_heavy",
        "degenerate",
        "number_only",
        "junk_sentence",
        "narrative_here_we",
    }
)


def is_valid_direction(
    text: str,
    *,
    min_words: int = 8,
    max_words: int = 25,
    min_chars: int = 15,
) -> bool:
    """Accept a short research-direction span for ranking / storage."""
    return classify_direction_rejection(
        text, min_words=min_words, max_words=max_words, min_chars=min_chars
    ) is None


def filter_directions(
    texts: list[str],
    *,
    min_words: int = 8,
    max_words: int = 25,
    min_chars: int = 15,
) -> list[str]:
    kept, _counts, _samples = filter_directions_audited(
        texts, min_words=min_words, max_words=max_words, min_chars=min_chars
    )
    return kept


def filter_directions_audited(
    texts: list[str],
    *,
    min_words: int = 8,
    max_words: int = 25,
    min_chars: int = 15,
) -> tuple[list[str], dict[str, int], list[tuple[str, str]]]:
    """Filter directions; return kept list, rejection counts, and sample (reason, text)."""
    out: list[str] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    samples: list[tuple[str, str]] = []
    for raw in texts:
        s = strip_list_prefix(raw)
        reason = classify_direction_rejection(
            s, min_words=min_words, max_words=max_words, min_chars=min_chars
        )
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
            if len(samples) < 5:
                samples.append((reason, s))
            continue
        key = s.lower()
        if key in seen:
            counts["duplicate"] = counts.get("duplicate", 0) + 1
            if len(samples) < 5:
                samples.append(("duplicate", s))
            continue
        seen.add(key)
        out.append(s)
    return out, counts, samples


def format_compliance(preds: list[str]) -> FormatCompliance:
    """Compute format stats on stored predictions (already stripped or raw)."""
    n = len(preds)
    if n == 0:
        return FormatCompliance(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    with_q = sum(1 for p in preds if "?" in p)
    with_doi = sum(1 for p in preds if _DOI.search(p) or _PMID.search(p))
    year_pref = sum(1 for p in preds if _YEAR_PREFIX.match(p.strip()))
    noise = sum(1 for p in preds if is_noise_direction(p))
    valid = sum(1 for p in preds if is_valid_direction(p))
    words = [len(strip_list_prefix(p).split()) for p in preds]
    word_ok = sum(1 for w in words if 8 <= w <= 25)
    numbered = sum(1 for p in preds if re.match(r"^\d+[\.\)]\s+\S", p.strip()))

    return FormatCompliance(
        n=n,
        frac_numbered_raw=numbered / n,
        frac_valid=valid / n,
        frac_noise=noise / n,
        frac_with_question_mark=with_q / n,
        frac_with_doi=with_doi / n,
        frac_year_prefix=year_pref / n,
        frac_word_len_ok=word_ok / n,
        mean_words=sum(words) / n,
    )


def is_clean_gt_text(text: str) -> bool:
    """GT research direction usable for eval (metadata-free)."""
    s = (text or "").strip()
    if not s:
        return False
    if is_noise_direction(s):
        return False
    # Allow slightly longer GT spans than generated directions.
    if len(s.split()) > 40:
        return False
    return True
