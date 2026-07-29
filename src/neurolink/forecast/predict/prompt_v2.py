"""Parallel forecast prompt v2 (smoke test only — does not replace production prompt)."""

from __future__ import annotations

import sqlite3

from .literature_lora import (
    LiteratureLoraConfig,
    build_context_summary,
    resolve_context_year,
)

# Numbered style examples (no "-" bullets — those encourage dash degeneration on base LMs).
STYLE_EXAMPLES_V2 = (
    "Role of cerebellar nuclei in top-down motor cortex control",
    "Microglial modulation of synaptic pruning during development",
    "Prefrontal dopamine signaling in flexible decision making",
)


def build_generation_prompt_v2(
    conn: sqlite3.Connection,
    target_year: int,
    cfg: LiteratureLoraConfig,
    *,
    k: int = 1,
    context_year: int | None = None,
    already: list[str] | None = None,
) -> str:
    """Completion-friendly prompt for Mistral-base / BrainGPT (ends with ``1.``)."""
    ctx_year = context_year if context_year is not None else resolve_context_year(target_year, cfg)
    context = build_context_summary(conn, ctx_year, cfg.max_context_questions)
    examples = "\n".join(f"{i}. {ex}" for i, ex in enumerate(STYLE_EXAMPLES_V2, start=1))
    avoid = ""
    if already:
        lines = "\n".join(f"- {a}" for a in already[-15:])
        avoid = f"\nAlready listed (do not repeat):\n{lines}\n"
    # Keep k in the instruction for API parity; smoke uses k=1 iterative.
    n_line = "one new research direction" if k == 1 else f"exactly {k} novel research directions"
    return (
        f"Neuroscience research directions for {target_year}.\n\n"
        f"Prior themes (until {ctx_year}, by impact):\n"
        f"{context}\n"
        f"{avoid}\n"
        "Examples of good directions:\n"
        f"{examples}\n\n"
        f"Write {n_line} (8-25 words, noun phrase, no question mark).\n"
        "Do not copy Prior themes or Examples.\n"
        "1."
    )
