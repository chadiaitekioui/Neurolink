"""Parallel forecast prompt v2 (smoke test only — does not replace production prompt).

Base LMs (Mistral / BrainGPT) continue text; they do not follow ``Write…`` instructions.
This prompt is a bare list of noun-phrase directions so the next tokens are another direction.
"""

from __future__ import annotations

import sqlite3

from .literature_lora import (
    LiteratureLoraConfig,
    _fetch_context_question_rows,
    resolve_context_year,
)

# Fixed style seeds (must stay out of CONTEXT copies via blocklist in the smoke runner).
STYLE_EXAMPLES_V2 = (
    "Role of cerebellar nuclei in top-down motor cortex control",
    "Microglial modulation of synaptic pruning during development",
    "Prefrontal dopamine signaling in flexible decision making",
)


def _context_direction_lines(
    conn: sqlite3.Connection,
    context_year: int,
    max_q: int,
) -> list[str]:
    """Prior themes as bare direction lines (no year tags / numbering)."""
    rows = _fetch_context_question_rows(conn, context_year, max_q)
    out: list[str] = []
    for r in rows:
        text = (r["question_text"] or "").strip()
        if text:
            out.append(text[:200])
    return out


def build_generation_prompt_v2(
    conn: sqlite3.Connection,
    target_year: int,
    cfg: LiteratureLoraConfig,
    *,
    k: int = 1,
    context_year: int | None = None,
    already: list[str] | None = None,
) -> str:
    """Few-shot continuation list; ends with a newline so the model starts a new line.

    ``target_year`` / ``k`` are unused in the text (kept for API parity with prod builder).
    """
    del k  # iterative smoke always asks for one next line via continuation
    ctx_year = context_year if context_year is not None else resolve_context_year(target_year, cfg)
    # Mix: style seeds first, then high-impact corpus lines (same format).
    lines: list[str] = list(STYLE_EXAMPLES_V2)
    for text in _context_direction_lines(conn, ctx_year, cfg.max_context_questions):
        if text.lower() not in {x.lower() for x in lines}:
            lines.append(text)
    if already:
        for text in already:
            t = (text or "").strip()
            if t and t.lower() not in {x.lower() for x in lines}:
                lines.append(t)
    # Trailing newline = open next direction line (no "Direction:" / "1." / "Write").
    return "\n".join(lines) + "\n"
