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


def is_noise_direction(text: str) -> bool:
    """True if the line looks like metadata / corpus junk."""
    s = (text or "").strip()
    if not s:
        return True
    if _YEAR_PREFIX.match(s):
        return True
    if _DOI.search(s) or _PMID.search(s):
        return True
    if _COMMENT.search(s) or _AUTHOR_HEAVY.search(s):
        return True
    if _DEGENERATE.search(s) or _NUMBER_ONLY.match(s):
        return True
    if is_junk_sentence(s):
        return True
    if _HERE_WE.search(s) and len(s.split()) > 30:
        return True
    return False


def is_valid_direction(
    text: str,
    *,
    min_words: int = 8,
    max_words: int = 25,
    min_chars: int = 15,
) -> bool:
    """Accept a short research-direction span for ranking / storage."""
    s = strip_list_prefix(text)
    if len(s) < min_chars:
        return False
    if is_noise_direction(s):
        return False
    n_words = len(s.split())
    if n_words < min_words or n_words > max_words:
        return False
    # Reject pure interrogatives kept as questions.
    if s.endswith("?"):
        return False
    return True


def filter_directions(
    texts: list[str],
    *,
    min_words: int = 8,
    max_words: int = 25,
    min_chars: int = 15,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in texts:
        s = strip_list_prefix(raw)
        if not is_valid_direction(s, min_words=min_words, max_words=max_words, min_chars=min_chars):
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


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
