"""Parallel smoke prompt — identical to production train/predict prompt.

Kept as a thin alias so smoke scripts stay explicit about using the shared
``build_generation_prompt`` (Year N → Year N+1, no per-article years).
"""

from __future__ import annotations

import sqlite3

from .literature_lora import LiteratureLoraConfig, build_generation_prompt

# Back-compat name for smoke runners that still import it (empty: no style seeds).
STYLE_EXAMPLE_TEXTS_V2: tuple[str, ...] = ()


def build_generation_prompt_v2(
    conn: sqlite3.Connection,
    target_year: int,
    cfg: LiteratureLoraConfig,
    *,
    k: int = 1,
    context_year: int | None = None,
    already: list[str] | None = None,
) -> str:
    """Same string as production ``build_generation_prompt``."""
    return build_generation_prompt(
        conn,
        target_year,
        cfg,
        k=k,
        context_year=context_year,
        already=already,
    )
