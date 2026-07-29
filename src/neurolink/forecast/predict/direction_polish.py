"""Lightweight polish for forecast smoke outputs (parallel test only).

Does not alter production parse/filter paths.
"""

from __future__ import annotations

import re

from .direction_filter import strip_list_prefix

_RESULT_CLAUSE = re.compile(
    r"\s*[,:;]?\s*(?:"
    r"reveal(?:s|ing)?|showing that|finding that|finds that|demonstrat(?:es|ing) that|"
    r"suggesting that|indicating that|concluding that|we (?:found|show|demonstrate)|"
    r"results? (?:show|indicate)|the study (?:finds|shows|reveals)"
    r")\b.*$",
    re.IGNORECASE,
)
_META_PREFIX = re.compile(
    r"^(?:"
    r"(?:neuroscience )?(?:research )?direction\s*:\s*|"
    r"write (?:exactly |one |a )?|"
    r"numbered list only\s*|"
    r"prior themes\s*|"
    r"style examples\s*"
    r")",
    re.IGNORECASE,
)


_INSTRUCTION_ECHO = re.compile(
    r"(?i)^(write|direction:|prior themes|good direction|neuroscience forecast|"
    r"numbered list|do not copy|already written)\b"
)


def polish_direction(text: str) -> str:
    """Strip list prefixes, meta echoes, result clauses, and dangling conjunctions."""
    s = strip_list_prefix(text or "")
    # Drop markdown / section headers often emitted by BrainGPT.
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"^\d{4}(-\d+)?\s*$", "", s)  # bare year headings
    s = re.sub(r"(?i)^year\s+\d{4}\s*:?\s*$", "", s)
    s = re.sub(r"^\d+\.\d+\.?\s*", "", s)
    s = re.sub(r"^(?:direction|research direction)\s*:\s*", "", s, flags=re.I)
    s = s.strip(" \"'")
    if not s:
        return ""
    if re.fullmatch(r"[-\s.#\d]+", s):
        return ""
    if re.match(r"^\d+\.\s*$", s):
        return ""
    if _INSTRUCTION_ECHO.match(s):
        return ""
    s = _META_PREFIX.sub("", s).strip()
    s = _RESULT_CLAUSE.sub("", s).strip(" .;:,")
    s = re.sub(r"\b(?:and|or|of|in|the|a|an|with|for|to|by)$", "", s, flags=re.I)
    return s.strip(" .;:,")
