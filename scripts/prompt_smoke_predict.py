#!/usr/bin/env python3
"""Parallel prompt-v2 smoke test (mistral_base + braingpt).

Does NOT call neurolink compare / production build_generation_prompt.
Writes JSON under eval/ — does not write predictions by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Allow `python scripts/prompt_smoke_predict.py` from repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neurolink.db import Database
from neurolink.forecast.predict.direction_filter import (
    build_context_blocklist,
    classify_direction_rejection,
)
from neurolink.forecast.predict.direction_polish import polish_direction
from neurolink.forecast.predict.literature_lora import (
    LiteratureLoraConfig,
    list_context_questions,
    resolve_context_year,
    resolve_literature_llm_cfg,
)
from neurolink.forecast.predict.llm_core import (
    CausalLMConfig,
    _generate_raw,
    parse_generated_directions,
    release_gpu_memory,
    score_completion,
)
from neurolink.forecast.predict.prompt_v2 import STYLE_EXAMPLES_V2, build_generation_prompt_v2
from neurolink.utils.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("prompt_smoke")


@dataclass
class PromptSmokeConfig:
    db_path: str = "data/neurolink.db"
    test_years: list[int] = field(default_factory=lambda: [2023])
    top_k: int = 10
    models: list[str] = field(default_factory=lambda: ["mistral_base", "braingpt"])
    filter_outputs: bool = True
    reject_context_copies: bool = True
    max_generation_attempts_factor: float = 2.0
    output_dir: str = "eval"
    literature: LiteratureLoraConfig = field(default_factory=LiteratureLoraConfig)


def _iterative_v2(
    conn,
    target_year: int,
    k: int,
    lit: LiteratureLoraConfig,
    llm_cfg: CausalLMConfig,
    *,
    filter_outputs: bool,
    reject_context_copies: bool,
    attempts_factor: float,
) -> tuple[list[dict], dict]:
    """Generate k directions with prompt v2 + polish + direction_filter."""
    import torch

    ctx_year = resolve_context_year(target_year, lit)
    context_qs = list_context_questions(conn, target_year, lit, context_year=ctx_year)
    blocklist = None
    if reject_context_copies:
        blocklist = build_context_blocklist([*context_qs, *STYLE_EXAMPLES_V2])

    collected: list[tuple[str, float]] = []
    already: list[str] = []
    rejection_counts: dict[str, int] = {}
    raw_samples: list[str] = []
    max_attempts = max(k, int(k * max(attempts_factor, 1.0)))
    per_tokens = max(24, int(llm_cfg.tokens_per_direction))
    greedy = llm_cfg.temperature <= 0.0

    for attempt in range(max_attempts):
        if len(collected) >= k:
            break
        prompt = build_generation_prompt_v2(
            conn,
            target_year,
            lit,
            k=1,
            context_year=ctx_year,
            already=already,
        )
        if greedy:
            torch.manual_seed(llm_cfg.seed + attempt)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(llm_cfg.seed + attempt)
            n_seq, do_sample = 1, False
        else:
            n_seq, do_sample = 1, True

        raw_outputs = _generate_raw(
            prompt,
            llm_cfg,
            max_new_tokens=per_tokens,
            n_seq=n_seq,
            do_sample=do_sample,
        )
        if attempt < 3:
            raw_samples.extend(raw_outputs[:1])

        for decoded in raw_outputs:
            # Prefer first non-empty polished line; fall back to full parse.
            line_cands: list[str] = []
            for line in decoded.splitlines():
                polished = polish_direction(line)
                if polished:
                    line_cands.append(polished)
            if not line_cands:
                line_cands = [
                    polish_direction(c)
                    for c in parse_generated_directions(decoded, min_len=8)
                ]
                line_cands = [c for c in line_cands if c]

            for s in line_cands:
                if filter_outputs:
                    reason = classify_direction_rejection(s, blocklist=blocklist)
                    if reason is not None:
                        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                        continue
                if s.lower() in {a.lower() for a in already}:
                    rejection_counts["duplicate"] = rejection_counts.get("duplicate", 0) + 1
                    continue
                try:
                    score = score_completion(prompt, s, llm_cfg)
                except Exception:
                    score = 0.0
                collected.append((s, score))
                already.append(s)
                if len(collected) >= k:
                    break
            if len(collected) >= k:
                break

    collected.sort(key=lambda x: x[1], reverse=True)
    kept = [{"rank": i + 1, "text": t, "score": float(s)} for i, (t, s) in enumerate(collected[:k])]
    audit = {
        "target_year": target_year,
        "context_year": ctx_year,
        "requested_k": k,
        "returned": len(kept),
        "attempts_budget": max_attempts,
        "rejection_counts": rejection_counts,
        "raw_samples": [r[:300] for r in raw_samples[:5]],
        "prompt_preview": build_generation_prompt_v2(
            conn, target_year, lit, k=1, context_year=ctx_year
        )[:800],
    }
    return kept, audit


def run_smoke(cfg: PromptSmokeConfig) -> Path:
    db = Database(resolve_path(cfg.db_path))
    out_dir = resolve_path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"prompt_smoke_{stamp}.json"

    results: dict = {
        "created_at": stamp,
        "config": {
            "db_path": cfg.db_path,
            "test_years": cfg.test_years,
            "top_k": cfg.top_k,
            "models": cfg.models,
            "prompt": "v2",
        },
        "runs": [],
    }

    with db.connect(readonly=True) as conn:
        for model in cfg.models:
            for year in cfg.test_years:
                logger.info("=== prompt-v2 smoke model=%s year=%d k=%d ===", model, year, cfg.top_k)
                year_max = year - 1
                llm_cfg = resolve_literature_llm_cfg(cfg.literature, year_max, model)
                kept, audit = _iterative_v2(
                    conn,
                    year,
                    cfg.top_k,
                    cfg.literature,
                    llm_cfg,
                    filter_outputs=cfg.filter_outputs,
                    reject_context_copies=cfg.reject_context_copies,
                    attempts_factor=cfg.max_generation_attempts_factor,
                )
                results["runs"].append(
                    {
                        "model": model,
                        "year": year,
                        "predictions": kept,
                        "audit": audit,
                    }
                )
                logger.info(
                    "model=%s year=%d kept=%d/%d rejections=%s",
                    model,
                    year,
                    len(kept),
                    cfg.top_k,
                    audit["rejection_counts"],
                )
                release_gpu_memory()

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Parallel prompt-v2 smoke (base + BrainGPT)")
    p.add_argument("--config", default="config/forecast/prompt_smoke.yaml")
    args = p.parse_args(argv)
    cfg = load_config(args.config, PromptSmokeConfig)
    path = run_smoke(cfg)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
