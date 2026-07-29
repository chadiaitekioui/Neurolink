"""Parallel forecast prompt v2 (smoke test only — does not replace production prompt)."""

from __future__ import annotations

import sqlite3

from .literature_lora import (
    LiteratureLoraConfig,
    _fetch_context_question_rows,
    resolve_context_year,
)

# Plain style examples (no list markers — numbered tails make base LMs emit empty 2. 3. 4.).
STYLE_EXAMPLES_V2 = (
    "Role of cerebellar nuclei in top-down motor cortex control",
    "Microglial modulation of synaptic pruning during development",
    "Prefrontal dopamine signaling in flexible decision making",
)


def build_context_summary_v2(conn: sqlite3.Connection, context_year: int, max_q: int) -> str:
    """CONTEXT without leading ``1. 2. 3.`` (avoids empty-number continuation)."""
    rows = _fetch_context_question_rows(conn, context_year, max_q)
    lines = []
    for r in rows:
        yr = r["year"] or "?"
        text = (r["question_text"] or "").strip()[:280]
        if text:
            lines.append(f"[{yr}] {text}")
    return "\n".join(lines)


def build_generation_prompt_v2(
    conn: sqlite3.Connection,
    target_year: int,
    cfg: LiteratureLoraConfig,
    *,
    k: int = 1,
    context_year: int | None = None,
    already: list[str] | None = None,
) -> str:
    """Completion prompt ending with ``Direction:`` (not bare ``1.``)."""
    ctx_year = context_year if context_year is not None else resolve_context_year(target_year, cfg)
    context = build_context_summary_v2(conn, ctx_year, cfg.max_context_questions)
    examples = "\n".join(STYLE_EXAMPLES_V2)
    avoid = ""
    if already:
        lines = "\n".join(already[-15:])
        avoid = f"\nAlready written (do not repeat):\n{lines}\n"
    return (
        f"Neuroscience forecast for {target_year}.\n\n"
        f"Prior themes (until {ctx_year}):\n"
        f"{context}\n"
        f"{avoid}\n"
        "Good direction style (different topics; do not copy):\n"
        f"{examples}\n\n"
        "Write ONE new research direction: 8-25 words, noun phrase, no question mark, "
        "no numbering, no headings.\n"
        "Direction:"
    )
