"""Parse PubMed abstract export text into structured articles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

from .pubmed_clean import (
    SECTION_HEADER,
    body_starts_at_section,
    clean_abstract,
    is_citation_line,
)

_ARTICLE_START = re.compile(r"^\d+\.\s")
_PMID = re.compile(r"^PMID:\s*(\d+)", re.MULTILINE)
_DOI = re.compile(r"^DOI:\s*(\S+)", re.MULTILINE)
_YEAR_LINE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})\b"
)
_BAD_TITLE = re.compile(
    r"^(?:\d+\.\s+)?(?:eCollection\s+)?(?:\d{4}\s+)?"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    re.IGNORECASE,
)


@dataclass
class ParsedArticle:
    pmid: str
    year: int | None
    doi: str | None
    title: str
    abstract: str
    raw_block: str


def split_pubmed_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if _ARTICLE_START.match(line) and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return ["\n".join(b).strip() for b in blocks if b]


def _extract_title_and_body(block: str) -> tuple[str, str]:
    lines = block.splitlines()
    start = 1
    while start < len(lines):
        stripped = lines[start].strip()
        if not stripped or is_citation_line(stripped):
            start += 1
            continue
        break

    title_lines: list[str] = []
    i = start
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            if title_lines:
                break
            continue
        if line.startswith("Author information:") or SECTION_HEADER.match(line + " "):
            break
        if re.match(r"^\(\d+\)", line):
            break
        if is_citation_line(line):
            break
        title_lines.append(line)
        i += 1

    title = " ".join(title_lines).strip()
    if _BAD_TITLE.match(title):
        title = ""

    body_start = body_starts_at_section(lines, i if title else start)
    body_lines: list[str] = []
    for line in lines[body_start:]:
        stripped = line.strip()
        if stripped.startswith(("PMID:", "DOI:", "PMCID:")):
            continue
        if stripped.startswith(("Copyright", "Conflict of interest")):
            break
        body_lines.append(line)

    raw_body = "\n".join(body_lines).strip()
    if not title and raw_body:
        first_line = raw_body.splitlines()[0].strip()
        if first_line and not SECTION_HEADER.match(first_line + " ") and len(first_line) < 300:
            title = first_line
            raw_body = "\n".join(raw_body.splitlines()[1:]).strip()

    abstract = clean_abstract(raw_body, title=title or None)
    return title, abstract


def _infer_year(block: str) -> int | None:
    m = _YEAR_LINE.search(block)
    if m:
        return int(m.group(1))
    m2 = re.search(r"eCollection\s+(\d{4})", block)
    if m2:
        return int(m2.group(1))
    m3 = re.search(r"\b(20\d{2})\b", block[:400])
    if m3:
        y = int(m3.group(1))
        year_hi = datetime.now(timezone.utc).year + 1
        if 2000 <= y <= year_hi:
            return y
    return None


def parse_pubmed_block(block: str) -> ParsedArticle | None:
    pm = _PMID.search(block)
    if not pm:
        return None
    doi_m = _DOI.search(block)
    title, abstract = _extract_title_and_body(block)
    if not abstract:
        return None
    return ParsedArticle(
        pmid=pm.group(1),
        year=_infer_year(block),
        doi=doi_m.group(1) if doi_m else None,
        title=title,
        abstract=abstract,
        raw_block=block,
    )


def parse_pubmed_text(text: str) -> Iterator[ParsedArticle]:
    for block in split_pubmed_blocks(text):
        parsed = parse_pubmed_block(block)
        if parsed:
            yield parsed
