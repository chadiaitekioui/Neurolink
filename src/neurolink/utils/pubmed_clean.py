"""PubMed abstract cleaning and IMRaD structure (rules)."""

from __future__ import annotations

import re
from typing import Literal

Bucket = Literal["question", "results"]

SECTION_HEADER = re.compile(
    r"^(BACKGROUND(?:/OBJECTIVES?)?|OBJECTIVE(?:S)?|INTRODUCTION|AIMS?|PURPOSE|"
    r"METHODS|RESULTS|FINDINGS|CONCLUSIONS|EXPECTED\s+RESULTS|SIGNIFICANCE)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)
_SECTION_INLINE = re.compile(
    r"(?<=[.;])\s+("
    r"BACKGROUND(?:/OBJECTIVES?)?|OBJECTIVE(?:S)?|INTRODUCTION|AIMS?|PURPOSE|"
    r"METHODS|RESULTS|FINDINGS|CONCLUSIONS|EXPECTED\s+RESULTS|SIGNIFICANCE"
    r")\s*:\s*",
    re.IGNORECASE,
)

_QUESTION_HEADERS = frozenset(
    {"BACKGROUND", "OBJECTIVE", "OBJECTIVES", "INTRODUCTION", "AIM", "AIMS", "PURPOSE"}
)
_RESULTS_HEADERS = frozenset({"METHODS", "RESULTS", "FINDINGS", "CONCLUSION", "CONCLUSIONS"})
_SKIP_HEADERS = frozenset({"EXPECTED RESULTS", "SIGNIFICANCE"})

_JOURNAL_CITATION = re.compile(r"^\d+\.\s+\S+.*\b(19|20)\d{2}\b")
_CITATION_CONTINUATION = re.compile(r"^(10\.\d|Epub\b)", re.IGNORECASE)
_PUBMED_TAIL = re.compile(r"^(PMID:|DOI:|PMCID:)", re.IGNORECASE)
_AUTHOR_LINE = re.compile(
    r"^[\w'’.\-]+(?:\s+[\w'’.\-]+)*\(\d+\)"
    r"(?:,\s*[\w'’.\-]+(?:\s+[\w'’.\-]+)*\(\d+\))*\.?\s*$"
)
_AFFILIATION_LINE = re.compile(r"^\(\d+\)\s*.+")
_AUTHOR_INFO = re.compile(r"^Author information:\s*$", re.IGNORECASE)
_EMAIL_ANY = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_ADDRESS_PREFIX = re.compile(r"\b(?:Electronic )?address:\s*\S+", re.IGNORECASE)
_AUTHOR_WITH_INDEX = re.compile(r"\b[A-Z][\w'’.\-]+(?:\s+[A-Z][\w'’.\-]+)*\(\d+\)")
_STANDALONE_AUTHOR = re.compile(
    r"^[A-Z][\w'’.\-]+(?:\s+[A-Z][\w'’.\-]+)*\s+[A-Z]{1,4}\.?\s*$"
)
_INSTITUTION_LINE_START = re.compile(
    r"^(?:\(\d+\)|Department of|Depertment of|Psychiatry,|address:|[\d]{4,6}\b)",
    re.IGNORECASE,
)
_AFFILIATION_KEYWORDS = re.compile(
    r"\b(?:Department of|Depertment of|University|Université|Universität|"
    r"University Hospital|Hospital of|Hôpital|Institute of|College of Medicine|"
    r"Medical Faculty|School of Medicine|Faculty of|Division of|"
    r"Center for|Centre for|Medical Center|Biostatistics)\b",
    re.IGNORECASE,
)
_SCIENTIFIC_PROSE = re.compile(
    r"\b(?:we|our|study|studied|patients?|results?|show(?:ed)?|demonstrat|"
    r"evaluat|investigat|found|significant|method|conclusion|background|purpose|aim)\b",
    re.IGNORECASE,
)
_COUNTRY_TAIL = re.compile(
    r"^(?:USA|France|Germany|Israel|Korea|China|UK|Morocco|Brazil)\.?\s*$",
    re.IGNORECASE,
)
_LEGAL_START = re.compile(r"^(Copyright|Conflict of interest)\b", re.IGNORECASE)
_POSTAL_LINE = re.compile(r"\b\d{4,5}\b.*\b(?:Morocco|France|Brazil|USA|Korea|China)\b", re.IGNORECASE)
_CITY_COUNTRY_LINE = re.compile(
    r"^[A-Za-zÀ-ÿ\s.'-]+,\s*(?:Morocco|France|Brazil|USA|Korea|China|UK|Israel|Germany)\.?\s*$",
    re.IGNORECASE,
)
_AUTHOR_ONLY = re.compile(r"^[A-Z][\w'’.\-]+ [A-Z]{1,3}(?:\(\d+\))+\.?\s*$")


def _normalize_section_layout(text: str) -> str:
    """Put inline section headers (e.g. 'France. OBJECTIVE:') on their own line."""
    return _SECTION_INLINE.sub(r"\n\1: ", text)


def is_citation_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _JOURNAL_CITATION.match(stripped):
        return True
    if _CITATION_CONTINUATION.match(stripped):
        return True
    if re.search(r"\bdoi:\s*$", stripped, re.IGNORECASE):
        return True
    return False


def _normalize_header(raw: str) -> str:
    u = re.sub(r"\s+", " ", raw.upper().strip()).rstrip(":")
    if u.startswith("BACKGROUND"):
        return "BACKGROUND"
    if u.startswith("OBJECTIVE"):
        return "OBJECTIVES"
    if u.startswith("METHOD"):
        return "METHODS"
    if u.startswith("RESULT") and not u.startswith("EXPECTED"):
        return "RESULTS"
    if u.startswith("CONCLUSION"):
        return "CONCLUSIONS"
    if u.startswith("EXPECTED"):
        return "EXPECTED RESULTS"
    return u


def _header_bucket(header: str) -> Bucket | None:
    if header in _SKIP_HEADERS:
        return None
    if header in _QUESTION_HEADERS or header == "BACKGROUND":
        return "question"
    if header in _RESULTS_HEADERS:
        return "results"
    return None


def _is_institution_only_line(stripped: str) -> bool:
    if _INSTITUTION_LINE_START.match(stripped):
        return True
    if _POSTAL_LINE.search(stripped) and not _SCIENTIFIC_PROSE.search(stripped):
        return True
    if not _AFFILIATION_KEYWORDS.search(stripped):
        return False
    if SECTION_HEADER.match(stripped + " "):
        return False
    if _SCIENTIFIC_PROSE.search(stripped):
        return False
    return len(stripped) < 220


def _is_metadata_line(stripped: str) -> bool:
    if not stripped:
        return False
    if _CITY_COUNTRY_LINE.match(stripped):
        return True
    if _AUTHOR_ONLY.match(stripped):
        return True
    if _STANDALONE_AUTHOR.match(stripped):
        return True
    if _AUTHOR_LINE.match(stripped):
        return True
    if _AUTHOR_WITH_INDEX.search(stripped) and not _SCIENTIFIC_PROSE.search(stripped):
        return len(stripped) < 280
    if _AFFILIATION_LINE.match(stripped):
        return True
    if _EMAIL_ANY.search(stripped) or _ADDRESS_PREFIX.search(stripped):
        return True
    if _COUNTRY_TAIL.match(stripped):
        return True
    if _is_institution_only_line(stripped):
        return True
    if re.match(r"^\d{4,6}\b", stripped) and not _SCIENTIFIC_PROSE.search(stripped):
        return True
    return False


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _strip_duplicate_title(text: str, title: str | None) -> str:
    if not title or not text:
        return text
    norm_title = _normalize_ws(title)
    if not norm_title or len(norm_title) < 12:
        return text
    lines = text.splitlines()
    if not lines:
        return text
    first = _normalize_ws(lines[0])
    if first.lower().startswith(norm_title.lower()[: min(len(norm_title), len(first))]):
        if len(first) >= len(norm_title) * 0.75:
            return "\n".join(lines[1:]).strip()
    return text


def _strip_inline_noise(text: str) -> str:
    text = _EMAIL_ANY.sub(" ", text)
    text = _ADDRESS_PREFIX.sub(" ", text)
    text = _AUTHOR_WITH_INDEX.sub(" ", text)
    text = re.sub(r"^[A-Z][\w'’.\-]+ [A-Z]{1,3}(?:\(\d+\))+\.\s*", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def clean_abstract(text: str, title: str | None = None) -> str:
    """Remove PubMed metadata once at collect time."""
    if not text or not text.strip():
        return ""

    text = re.sub(
        r"Author information:\s*\n(?:\(\d+\)[^\n]*\n?)+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    kept: list[str] = []
    in_affiliations = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            in_affiliations = False
            continue
        if _LEGAL_START.match(stripped):
            break
        if _PUBMED_TAIL.match(stripped):
            continue
        if _AUTHOR_INFO.match(stripped):
            in_affiliations = True
            continue
        if in_affiliations and _AFFILIATION_LINE.match(stripped):
            continue
        if _is_metadata_line(stripped):
            continue
        in_affiliations = False
        kept.append(_strip_inline_noise(stripped))

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    cleaned = _strip_duplicate_title(cleaned, title)
    return _strip_inline_noise(cleaned)


_DOI_OR_PMID = re.compile(r"(?i)(?:\bdoi\s*:|\bpmid\s*:|\bpmcid\s*:)")
_COMMENT_IN = re.compile(r"\bComment in\b", re.IGNORECASE)
_CONTRIB_EQ = re.compile(r"\(#\)?\s*Contributed equally\b", re.IGNORECASE)
_COLLABORATORS = re.compile(r"\bCollaborators?\s*:", re.IGNORECASE)
_UPDATE_BIORXIV = re.compile(r"\bUpdate of\s+(?:bioRxiv|medRxiv)\b", re.IGNORECASE)
_HEAVY_AUTHOR_INDEX = re.compile(r"(?:\([#\d]+\)){2,}|\([#\d]+\)(?:\([#\d]+\))+")


def is_junk_sentence(sentence: str) -> bool:
    """Drop residual metadata before BERT / subject extraction."""
    s = _normalize_ws(sentence)
    if len(s) < 12:
        return True
    if _AUTHOR_ONLY.match(s):
        return True
    if _is_metadata_line(s):
        return True
    if _STANDALONE_AUTHOR.match(s):
        return True
    if _AFFILIATION_KEYWORDS.search(s) and not _SCIENTIFIC_PROSE.search(s):
        return True
    if _DOI_OR_PMID.search(s):
        return True
    if _COMMENT_IN.search(s):
        return True
    if _CONTRIB_EQ.search(s):
        return True
    if _COLLABORATORS.search(s):
        return True
    if _UPDATE_BIORXIV.search(s):
        return True
    if _HEAVY_AUTHOR_INDEX.search(s) and s.count("(") >= 3:
        return True
    return False


def polish_segment_field(text: str) -> str:
    """Final whitespace / author cleanup on segmented output."""
    return _strip_inline_noise(text)


def structure_abstract_sections(text: str) -> list[tuple[str, Bucket | None, str | None]]:
    """Rules: split on IMRaD headers; return (content, bucket hint, section name).

    bucket hint is set when a header is found; None means BERT must classify.
    section is e.g. OBJECTIVES / BACKGROUND / METHODS, or None for preamble.
    """
    if not text or not text.strip():
        return []

    text = _normalize_section_layout(text)
    matches = list(SECTION_HEADER.finditer(text))
    if not matches:
        return [(text.strip(), None, None)]

    chunks: list[tuple[str, Bucket | None, str | None]] = []
    preamble = _normalize_ws(text[: matches[0].start()])
    if preamble and not _is_metadata_line(preamble):
        chunks.append((preamble, None, None))

    for i, match in enumerate(matches):
        header = _normalize_header(match.group(1))
        bucket = _header_bucket(header)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = _strip_inline_noise(_normalize_ws(text[start:end]))
        if not content or bucket is None:
            continue
        chunks.append((content, bucket, header))
    return chunks


def structure_abstract(text: str) -> list[tuple[str, Bucket | None]]:
    """Rules: split on IMRaD headers; return (content, bucket hint)."""
    return [(content, bucket) for content, bucket, _section in structure_abstract_sections(text)]


def body_starts_at_section(lines: list[str], start: int) -> int:
    """First scientific line when parsing PubMed blocks."""
    in_affiliations = False
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            in_affiliations = False
            continue
        if _LEGAL_START.match(stripped) or stripped.startswith(("PMID:", "DOI:")):
            return i
        if _AUTHOR_INFO.match(stripped):
            in_affiliations = True
            continue
        if in_affiliations and _AFFILIATION_LINE.match(stripped):
            continue
        if _is_metadata_line(stripped):
            continue
        if SECTION_HEADER.match(stripped + " "):
            return i
        if not is_citation_line(stripped):
            return i
    return start
