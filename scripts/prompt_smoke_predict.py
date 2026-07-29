#!/usr/bin/env python3
"""Parallel prompt-v2 smoke test (mistral_base + braingpt).

Uses the same DB context as production (build_context_summary), predicts
target_year (default 2023) from articles ≤ target-1, then computes the core
benchmark metrics (format + MiniLM P@k/R@k, optional contamination).

Does NOT call neurolink compare / production build_generation_prompt.
Writes JSON under eval/.
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
from neurolink.eval.contamination import run_contamination_audit
from neurolink.eval.matching import make_matcher
from neurolink.forecast.predict.direction_filter import (
    build_context_blocklist,
    classify_direction_rejection,
    format_compliance,
    is_clean_gt_text,
)
from neurolink.forecast.predict.direction_polish import polish_direction
from neurolink.forecast.predict.literature_lora import (
    LiteratureLoraConfig,
    _fetch_context_question_rows,
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
from neurolink.forecast.predict.prompt_v2 import build_generation_prompt_v2
from neurolink.utils.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("prompt_smoke")


@dataclass
class PromptSmokeConfig:
    db_path: str = "data/neurolink.db"
    test_years: list[int] = field(default_factory=lambda: [2023])
    top_k: int = 10
    # Eval ks (same as benchmark); only computed when ≤ n predictions.
    eval_top_k: list[int] = field(default_factory=lambda: [10])
    models: list[str] = field(default_factory=lambda: ["mistral_base", "braingpt"])
    filter_outputs: bool = True
    reject_context_copies: bool = True
    max_generation_attempts_factor: float = 2.0
    output_dir: str = "eval"
    # Benchmark-aligned eval knobs.
    semantic_threshold: float = 0.50
    matcher_backend: str = "minilm"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    critical_only: bool = True
    filter_gt_noise: bool = True
    min_pred_coverage: float = 0.5
    contamination_enabled: bool = True
    contamination_corpus_sample: int = 200
    literature: LiteratureLoraConfig = field(default_factory=LiteratureLoraConfig)


def _load_ground_truth(conn, target_year: int, cfg: PromptSmokeConfig) -> list[str]:
    ref_rows = conn.execute(
        "SELECT question_text, is_critical FROM questions WHERE year = ?",
        (target_year,),
    ).fetchall()
    refs: list[str] = []
    for r in ref_rows:
        if cfg.critical_only and not r["is_critical"]:
            continue
        text = r["question_text"] or ""
        if cfg.filter_gt_noise and not is_clean_gt_text(text):
            continue
        refs.append(text)
    if not refs:
        refs = [r["question_text"] for r in ref_rows if r["question_text"]]
        if cfg.filter_gt_noise:
            cleaned = [t for t in refs if is_clean_gt_text(t)]
            if cleaned:
                refs = cleaned
    return refs


def compute_benchmark_metrics(
    conn,
    *,
    target_year: int,
    model: str,
    predictions: list[str],
    cfg: PromptSmokeConfig,
    year_max: int,
    llm_cfg: CausalLMConfig | None = None,
) -> dict:
    """Core Job-2 metrics: format compliance + MiniLM P@k/R@k (+ optional contamination)."""
    metrics: dict = {
        "target_year": target_year,
        "model": model,
        "n_predictions": len(predictions),
    }
    if not predictions:
        return metrics

    fc = format_compliance(predictions)
    metrics.update(
        {
            "format_frac_valid": fc.frac_valid,
            "format_frac_noise": fc.frac_noise,
            "format_frac_word_len_ok": fc.frac_word_len_ok,
            "format_frac_with_doi": fc.frac_with_doi,
            "format_mean_words": fc.mean_words,
        }
    )

    refs = _load_ground_truth(conn, target_year, cfg)
    metrics["n_ground_truth"] = len(refs)
    if not refs:
        logger.warning("No ground truth for year %d — skip P@k/R@k", target_year)
        return metrics

    matcher = make_matcher(
        refs,
        cfg.semantic_threshold,
        backend=cfg.matcher_backend,
        model_name=cfg.embed_model,
    )
    by_k: dict[str, dict] = {}
    for k in cfg.eval_top_k:
        coverage = min(1.0, len(predictions) / float(k)) if k > 0 else 0.0
        entry: dict = {"pred_coverage@k": coverage}
        if coverage < cfg.min_pred_coverage:
            entry["precision@k_skipped"] = 1.0
            by_k[str(k)] = entry
            continue
        p, r, r_norm = matcher.precision_recall_at_k(predictions, k)
        entry.update(
            {
                "precision@k": p,
                "recall@k": r,
                "recall@k_normalized": r_norm,
            }
        )
        by_k[str(k)] = entry
        logger.info(
            "metrics model=%s year=%d k=%d P=%.3f R=%.3f Rnorm=%.3f coverage=%.2f",
            model,
            target_year,
            k,
            p,
            r,
            r_norm,
            coverage,
        )
    metrics["at_k"] = by_k

    if cfg.contamination_enabled and llm_cfg is not None:
        lit = cfg.literature
        report = run_contamination_audit(
            conn,
            target_year=target_year,
            year_max=year_max,
            predictions=predictions,
            lit_cfg=lit,
            llm_cfg=llm_cfg,
            semantic_threshold=cfg.semantic_threshold,
            corpus_sample_size=cfg.contamination_corpus_sample,
            include_train_overlap=False,
            matcher_backend=cfg.matcher_backend,
            embed_model=cfg.embed_model,
        )
        if report:
            metrics["contamination"] = {
                "corpus_recycling": report.corpus_recycling_rate,
                "context_recycling": report.context_recycling_rate,
                "context_verbatim_recycling": report.context_verbatim_recycling_rate,
                "verbatim_recycling": report.verbatim_recycling_rate,
            }
            primary_k = max(cfg.eval_top_k) if cfg.eval_top_k else 10
            p_primary = by_k.get(str(primary_k), {}).get("precision@k")
            if p_primary is not None:
                metrics["contamination"]["extension_vs_context"] = (
                    p_primary - report.context_recycling_rate
                )
                metrics["contamination"]["extension_vs_corpus"] = (
                    p_primary - report.corpus_recycling_rate
                )
            logger.info(
                "contamination model=%s year=%d corpus=%.3f context=%.3f",
                model,
                target_year,
                report.corpus_recycling_rate,
                report.context_recycling_rate,
            )
    return metrics


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
    if ctx_year >= target_year:
        raise ValueError(
            f"Refuse leak: context_year={ctx_year} must be < target_year={target_year}"
        )
    context_rows = _fetch_context_question_rows(conn, ctx_year, lit.max_context_questions)
    context_qs = [
        (r["question_text"] or "").strip()
        for r in context_rows
        if (r["question_text"] or "").strip()
    ]
    context_years = sorted(
        {int(r["year"]) for r in context_rows if r["year"] is not None}
    )
    if context_years and max(context_years) >= target_year:
        raise ValueError(
            f"Refuse leak: context contains year>={target_year}: {context_years}"
        )
    blocklist = None
    if reject_context_copies:
        blocklist = build_context_blocklist(context_qs)

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
            n_seq = max(1, min(3, int(llm_cfg.num_return_sequences)))
            do_sample = True

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

            if not line_cands:
                rejection_counts["empty_raw"] = rejection_counts.get("empty_raw", 0) + 1
                continue

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
    full_prompt = build_generation_prompt_v2(
        conn, target_year, lit, k=1, context_year=ctx_year
    )
    audit = {
        "target_year": target_year,
        "context_year": ctx_year,
        "context_years_in_prompt": context_years,
        "context_n": len(context_qs),
        "context_source": "build_context_summary (same as production)",
        "requested_k": k,
        "returned": len(kept),
        "attempts_budget": max_attempts,
        "rejection_counts": rejection_counts,
        "raw_samples": [r[:300] for r in raw_samples[:5]],
        "prompt_preview": full_prompt[:1200],
        "prompt_ends_with": full_prompt.rstrip().splitlines()[-1] if full_prompt.strip() else "",
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
            "eval_top_k": cfg.eval_top_k,
            "models": cfg.models,
            "prompt": "v2_db_context",
            "setup": "build_context_summary year<=(target-1); Directions target_year; benchmark metrics",
            "semantic_threshold": cfg.semantic_threshold,
            "critical_only": cfg.critical_only,
            "contamination_enabled": cfg.contamination_enabled,
        },
        "runs": [],
        "metrics_summary": [],
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
                pred_texts = [p["text"] for p in kept]
                metrics = compute_benchmark_metrics(
                    conn,
                    target_year=year,
                    model=model,
                    predictions=pred_texts,
                    cfg=cfg,
                    year_max=year_max,
                    llm_cfg=llm_cfg,
                )
                results["runs"].append(
                    {
                        "model": model,
                        "year": year,
                        "predictions": kept,
                        "audit": audit,
                        "metrics": metrics,
                    }
                )
                results["metrics_summary"].append(
                    {
                        "model": model,
                        "year": year,
                        "n_predictions": metrics.get("n_predictions", 0),
                        "n_ground_truth": metrics.get("n_ground_truth"),
                        "at_k": metrics.get("at_k"),
                        "format_frac_valid": metrics.get("format_frac_valid"),
                        "contamination": metrics.get("contamination"),
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
    logger.info("metrics_summary=%s", json.dumps(results["metrics_summary"], indent=2))
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
