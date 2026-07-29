"""Parallel forecast prompt v2 (smoke test only — does not replace production prompt).

Same DB selection as production (``_fetch_context_question_rows``), but lines are
``[year] text`` without numbering and without style examples. Ends with
``Directions {target_year}:`` for continuation.
"""

from __future__ import annotations

import sqlite3

from .literature_lora import (
    LiteratureLoraConfig,
    _fetch_context_question_rows,
    resolve_context_year,
)


def _context_lines(conn: sqlite3.Connection, context_year: int, max_q: int) -> list[str]:
    """Same rows as ``build_context_summary``, without ``1. 2. 3.`` prefixes."""
    rows = _fetch_context_question_rows(conn, context_year, max_q)
    lines: list[str] = []
    for r in rows:
        text = (r["question_text"] or "").strip()
        if not text:
            continue
        yr = r["year"] or "?"
        lines.append(f"[{yr}] {text[:280]}")
    return lines


def build_generation_prompt_v2(
    conn: sqlite3.Connection,
    target_year: int,
    cfg: LiteratureLoraConfig,
    *,
    k: int = 1,
    context_year: int | None = None,
    already: list[str] | None = None,
) -> str:
    """DB themes (≤ ctx_year) as ``[year] text`` + open ``Directions {target_year}:``.

    Context selection matches production; no style examples, no numbered list.
    ``k`` unused in the text (API parity); iterative smoke asks one next line.
    """
    del k
    ctx_year = context_year if context_year is not None else resolve_context_year(target_year, cfg)
    if ctx_year >= target_year:
        raise ValueError(
            f"context_year ({ctx_year}) must be < target_year ({target_year}) "
            "so the forecast does not see same-year articles"
        )

    lines = _context_lines(conn, ctx_year, cfg.max_context_questions)
    # Hard guard: drop any row that somehow has year >= target.
    filtered: list[str] = []
    for line in lines:
        try:
            yr = int(line[1:5])  # "[YYYY] ..."
            if yr >= target_year:
                continue
        except (ValueError, IndexError):
            pass
        filtered.append(line)

    parts = [
        f"Prior themes (until {ctx_year}, by impact):",
        *filtered,
    ]
    if already:
        parts.append("Already listed (do not repeat):")
        for a in already[-15:]:
            t = (a or "").strip()
            if t:
                parts.append(t)
    parts.append(f"Directions {target_year}:")
    return "\n".join(parts) + "\n"
