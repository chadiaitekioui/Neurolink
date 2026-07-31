#!/usr/bin/env python3
"""Offline stress metrics for Job-2 benches (no LM regeneration).

Computes indicators missing from the main eval summary:
  - retrieval baselines (context copy, corpus BM25, corpus MiniLM)
  - length-controlled MiniLM P@k / R@k
  - near-threshold match diagnostics
  - prediction diversity (trigrams + embedding pairwise distance)
  - composite novelty KPIs (P − recycling, etc.)
  - bounded recall@k_normalized

Writes a single JSON under eval/ (default: eval/stress_metrics_<ts>.json).

Example (bench22_inst_base Round A + B)::

  .venv/bin/python scripts/eval_stress_metrics.py \\
    --db-path bench22_inst_base/data/neurolink.db \\
    --out-dir bench22_inst_base/eval \\
    --predict-run-id benchmark_20260730T143510Z:round_a \\
    --predict-run-id benchmark_20260730T195112Z:round_b
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neurolink.forecast.predict.direction_filter import is_clean_gt_text
from neurolink.forecast.predict.literature_lora import LiteratureLoraConfig, list_context_questions
from neurolink.utils.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("eval_stress")

_WORD_RE = re.compile(r"[A-Za-z0-9α-ωΑ-Ω]+(?:[-'][A-Za-z0-9α-ωΑ-Ω]+)?")
_EMBED_CACHE: dict[str, np.ndarray] = {}
_ST_MODEL_CACHE: dict[str, object] = {}


def _st_model(model_name: str):
    """Load SentenceTransformer once per process (MiniLM required)."""
    if model_name in _ST_MODEL_CACHE:
        return _ST_MODEL_CACHE[model_name]
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    _ST_MODEL_CACHE[model_name] = model
    return model


@dataclass
class StressConfig:
    db_path: str = "data/neurolink.db"
    out_dir: str = "eval"
    predict_config: str = "config/forecast/predict_compare.yaml"
    test_years: list[int] = field(default_factory=lambda: [2023, 2024, 2025])
    top_k: list[int] = field(default_factory=lambda: [10, 50])
    models: list[str] = field(
        default_factory=lambda: ["literature_lora", "mistral_base", "braingpt"]
    )
    semantic_threshold: float = 0.50
    near_threshold_band: float = 0.05
    length_control_words: int = 12
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    critical_only: bool = True
    filter_gt_noise: bool = True
    max_context_questions: int = 30
    corpus_sample: int = 4000
    corpus_seed: int = 42
    # Optional: pull existing Job-2 recycling from evaluations table
    eval_run_id: str | None = None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _truncate_words(text: str, n: int) -> str:
    words = (text or "").split()
    return " ".join(words[:n]) if words else ""


def _word_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def _trigrams(text: str) -> list[str]:
    toks = _word_tokens(text)
    if len(toks) < 3:
        return [" ".join(toks)] if toks else []
    return [" ".join(toks[i : i + 3]) for i in range(len(toks) - 2)]


def _encode(
    texts: list[str],
    model_name: str,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts with MiniLM; caches embeddings by (model, text)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _st_model(model_name)
    missing_idx: list[int] = []
    missing_txt: list[str] = []
    cached: list[np.ndarray | None] = [None] * len(texts)
    for i, t in enumerate(texts):
        key = f"{model_name}::{t}"
        if key in _EMBED_CACHE:
            cached[i] = _EMBED_CACHE[key]
        else:
            missing_idx.append(i)
            missing_txt.append(t[:4000])
    if missing_txt:
        emb = model.encode(
            missing_txt,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb = np.asarray(emb, dtype=np.float32)
        for j, i in enumerate(missing_idx):
            _EMBED_CACHE[f"{model_name}::{texts[i]}"] = emb[j]
            cached[i] = emb[j]
    return np.stack([c for c in cached if c is not None], axis=0)


def _semantic_recycling_rate(
    preds: list[str],
    corpus: list[str],
    threshold: float,
    model_name: str,
) -> float:
    """Fraction of preds whose best MiniLM cosine to corpus ≥ threshold."""
    if not preds or not corpus:
        return 0.0
    pred_emb = _encode(preds, model_name)
    corp_emb = _encode(corpus, model_name)
    sims = pred_emb @ corp_emb.T
    best = sims.max(axis=1)
    return float((best >= threshold).mean())


def _precision_recall_from_emb(
    pred_emb: np.ndarray,
    ref_emb: np.ndarray,
    *,
    k: int,
    threshold: float,
    n_gt: int,
) -> dict[str, float]:
    top = pred_emb[:k]
    n_preds = int(top.shape[0])
    if n_preds == 0 or ref_emb.shape[0] == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "recall_normalized": 0.0,
            "recall_normalized_bounded": 0.0,
            "n_preds": float(n_preds),
            "n_gt": float(n_gt),
            "coverage": min(1.0, n_preds / float(k)) if k else 0.0,
        }
    sims = top @ ref_emb.T
    matched = float((sims.max(axis=0) >= threshold).sum())
    precision = float((sims.max(axis=1) >= threshold).sum()) / n_preds
    recall = matched / n_gt
    r_norm = matched / min(k, n_gt)
    return {
        "precision": precision,
        "recall": recall,
        "recall_normalized": r_norm,
        "recall_normalized_bounded": min(1.0, r_norm),
        "n_preds": float(n_preds),
        "n_gt": float(n_gt),
        "coverage": min(1.0, n_preds / float(k)),
    }


def _precision_recall(
    preds: list[str],
    refs: list[str],
    *,
    k: int,
    threshold: float,
    model_name: str,
    length_words: int | None = None,
) -> dict[str, float]:
    top = preds[:k]
    if length_words is not None:
        top = [_truncate_words(p, length_words) for p in top]
        refs_use = [_truncate_words(r, length_words) for r in refs]
    else:
        refs_use = refs
    if not top or not refs_use:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "recall_normalized": 0.0,
            "recall_normalized_bounded": 0.0,
            "n_preds": float(len(top)),
            "n_gt": float(len(refs_use)),
            "coverage": min(1.0, len(top) / float(k)) if k else 0.0,
        }
    pred_emb = _encode(top, model_name)
    ref_emb = _encode(refs_use, model_name)
    return _precision_recall_from_emb(
        pred_emb, ref_emb, k=k, threshold=threshold, n_gt=len(refs_use)
    )


def _near_threshold_stats(
    preds: list[str],
    refs: list[str],
    *,
    k: int,
    threshold: float,
    band: float,
    model_name: str,
) -> dict[str, float]:
    top = preds[:k]
    if not top or not refs:
        return {
            "frac_matched": 0.0,
            "frac_near_threshold": 0.0,
            "mean_best_sim": 0.0,
            "p10_best_sim": 0.0,
            "p50_best_sim": 0.0,
            "threshold": threshold,
            "band": band,
        }
    sims = _encode(top, model_name) @ _encode(refs, model_name).T
    best = sims.max(axis=1)
    matched = best >= threshold
    near = (best >= threshold - band) & (best < threshold)
    return {
        "frac_matched": float(matched.mean()),
        "frac_near_threshold": float(near.mean()),
        "mean_best_sim": float(best.mean()),
        "p10_best_sim": float(np.percentile(best, 10)),
        "p50_best_sim": float(np.percentile(best, 50)),
        "threshold": threshold,
        "band": band,
    }


def _diversity_metrics(preds: list[str], model_name: str, k: int) -> dict[str, float]:
    top = [p for p in preds[:k] if (p or "").strip()]
    if not top:
        return {
            "n": 0.0,
            "unique_trigram_ratio": 0.0,
            "mean_pairwise_cosine_distance": 0.0,
            "embedding_entropy": 0.0,
        }
    all_tri: list[str] = []
    for p in top:
        all_tri.extend(_trigrams(p))
    uniq_ratio = (len(set(all_tri)) / len(all_tri)) if all_tri else 0.0

    emb = _encode(top, model_name)
    if len(emb) < 2:
        mean_dist = 0.0
    else:
        sims = emb @ emb.T
        iu = np.triu_indices(len(emb), k=1)
        mean_dist = float((1.0 - sims[iu]).mean())

    centroid = emb.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    sims_c = emb @ centroid
    logits = sims_c / 0.1
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(probs)))

    return {
        "n": float(len(top)),
        "unique_trigram_ratio": float(uniq_ratio),
        "mean_pairwise_cosine_distance": mean_dist,
        "embedding_entropy": entropy,
    }


def _bm25_topk(
    query: str,
    corpus: list[str],
    k: int,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[str]:
    """Okapi BM25 over a bag-of-words corpus (sklearn-free)."""
    if not corpus or k <= 0:
        return []
    docs = [_word_tokens(c) for c in corpus]
    N = len(docs)
    avgdl = sum(len(d) for d in docs) / max(N, 1)
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    q_toks = _word_tokens(query)
    if not q_toks:
        return corpus[:k]
    scores = np.zeros(N, dtype=np.float64)
    for qi in set(q_toks):
        n_qi = df.get(qi, 0)
        if n_qi == 0:
            continue
        idf = math.log(1.0 + (N - n_qi + 0.5) / (n_qi + 0.5))
        for i, d in enumerate(docs):
            if not d:
                continue
            tf = d.count(qi)
            if tf == 0:
                continue
            denom = tf + k1 * (1.0 - b + b * len(d) / (avgdl + 1e-9))
            scores[i] += idf * (tf * (k1 + 1.0) / denom)
    order = np.argsort(-scores)
    out: list[str] = []
    seen: set[str] = set()
    for i in order:
        t = corpus[int(i)].strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= k:
            break
    return out


def _minilm_retrieve_topk(
    query_texts: list[str],
    corpus: list[str],
    k: int,
    model_name: str,
) -> list[str]:
    if not corpus or k <= 0 or not query_texts:
        return []
    q_emb = _encode([t[:4000] for t in query_texts], model_name)
    centroid = q_emb.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
    c_emb = _encode([t[:4000] for t in corpus], model_name)
    scores = c_emb @ centroid
    order = np.argsort(-scores)
    out: list[str] = []
    seen: set[str] = set()
    for i in order:
        t = corpus[int(i)].strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= k:
            break
    return out


def _load_gt(conn: sqlite3.Connection, year: int, cfg: StressConfig) -> list[str]:
    rows = conn.execute(
        "SELECT question_text, is_critical FROM questions WHERE year = ?",
        (year,),
    ).fetchall()
    refs: list[str] = []
    for r in rows:
        if cfg.critical_only and not r["is_critical"]:
            continue
        text = (r["question_text"] or "").strip()
        if not text:
            continue
        if cfg.filter_gt_noise and not is_clean_gt_text(text):
            continue
        refs.append(text)
    if not refs:
        refs = [(r["question_text"] or "").strip() for r in rows if (r["question_text"] or "").strip()]
        if cfg.filter_gt_noise:
            cleaned = [t for t in refs if is_clean_gt_text(t)]
            if cleaned:
                refs = cleaned
    return refs


def _load_preds(
    conn: sqlite3.Connection,
    year: int,
    model: str,
    run_id: str,
) -> list[str]:
    rows = conn.execute(
        """
        SELECT question_predicted FROM predictions
        WHERE target_year=? AND model=? AND run_id=?
        ORDER BY rank
        """,
        (year, model, run_id),
    ).fetchall()
    return [(r["question_predicted"] or "").strip() for r in rows if (r["question_predicted"] or "").strip()]


def _sample_corpus(
    conn: sqlite3.Connection,
    year: int,
    sample: int,
    seed: int,
) -> list[str]:
    rows = conn.execute(
        """
        SELECT question_text FROM questions
        WHERE year IS NOT NULL AND year < ?
        """,
        (year,),
    ).fetchall()
    texts = [(r["question_text"] or "").strip() for r in rows if (r["question_text"] or "").strip()]
    if len(texts) <= sample:
        return texts
    rng = np.random.default_rng(seed + year)
    idx = rng.choice(len(texts), size=sample, replace=False)
    return [texts[int(i)] for i in idx]


def _existing_eval_metrics(
    conn: sqlite3.Connection,
    eval_run_id: str | None,
    year: int,
    model: str,
) -> dict[str, float]:
    if not eval_run_id:
        # latest eval row for this model/year
        row = conn.execute(
            """
            SELECT run_id FROM evaluations
            WHERE target_year=? AND model=?
            ORDER BY id DESC LIMIT 1
            """,
            (year, model),
        ).fetchone()
        if not row:
            return {}
        eval_run_id = row["run_id"]
    rows = conn.execute(
        """
        SELECT metric, k, value FROM evaluations
        WHERE run_id=? AND target_year=? AND model=?
        """,
        (eval_run_id, year, model),
    ).fetchall()
    out: dict[str, float] = {}
    for r in rows:
        key = r["metric"] if not r["k"] else f"{r['metric']}@{r['k']}"
        out[key] = float(r["value"])
    return out


def _composites(
    precision_by_k: dict[int, float],
    context_rec: float,
    corpus_rec: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, p in precision_by_k.items():
        out[f"extension_vs_context@{k}"] = p - context_rec
        out[f"extension_vs_corpus@{k}"] = p - corpus_rec
        out[f"novelty_score@{k}"] = p - 0.5 * (context_rec + corpus_rec)
    return out


def _lit_cfg(predict_config: str, max_context: int) -> LiteratureLoraConfig:
    from neurolink.forecast.predict.models import PredictConfig

    pred = load_config(predict_config, PredictConfig)
    lit = pred.literature
    lit.max_context_questions = max_context
    return lit


def evaluate_cell(
    conn: sqlite3.Connection,
    *,
    year: int,
    model: str,
    predict_run_id: str,
    cfg: StressConfig,
    lit_cfg: LiteratureLoraConfig,
    baselines_cache: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any] | None:
    preds = _load_preds(conn, year, model, predict_run_id)
    refs = _load_gt(conn, year, cfg)
    if not preds or not refs:
        logger.warning("Skip %s %d: preds=%d refs=%d", model, year, len(preds), len(refs))
        return None

    context = list_context_questions(conn, year, lit_cfg)
    corpus = _sample_corpus(conn, year, cfg.corpus_sample, cfg.corpus_seed)

    # --- model-side metrics ---
    match: dict[str, Any] = {}
    length_match: dict[str, Any] = {}
    near: dict[str, Any] = {}
    precision_by_k: dict[int, float] = {}
    for k in cfg.top_k:
        m = _precision_recall(
            preds, refs, k=k, threshold=cfg.semantic_threshold, model_name=cfg.embed_model
        )
        match[str(k)] = m
        precision_by_k[k] = m["precision"]
        length_match[str(k)] = _precision_recall(
            preds,
            refs,
            k=k,
            threshold=cfg.semantic_threshold,
            model_name=cfg.embed_model,
            length_words=cfg.length_control_words,
        )
        near[str(k)] = _near_threshold_stats(
            preds,
            refs,
            k=k,
            threshold=cfg.semantic_threshold,
            band=cfg.near_threshold_band,
            model_name=cfg.embed_model,
        )

    diversity = {str(k): _diversity_metrics(preds, cfg.embed_model, k) for k in cfg.top_k}

    context_rec = _semantic_recycling_rate(
        preds[: max(cfg.top_k)], context, cfg.semantic_threshold, cfg.embed_model
    )
    corpus_rec = _semantic_recycling_rate(
        preds[: max(cfg.top_k)], corpus, cfg.semantic_threshold, cfg.embed_model
    )
    composites = _composites(precision_by_k, context_rec, corpus_rec)

    existing = _existing_eval_metrics(conn, cfg.eval_run_id, year, model)
    # Prefer freshly computed recycling; still expose Job-2 numbers for comparison.
    job2_ctx = existing.get("contamination_context_recycling")
    job2_corp = existing.get("contamination_corpus_recycling")
    job2_composites = {}
    if job2_ctx is not None and job2_corp is not None:
        job2_p = {
            k: existing.get(f"precision@k@{k}", precision_by_k.get(k, 0.0))
            for k in cfg.top_k
        }
        job2_composites = _composites(job2_p, float(job2_ctx), float(job2_corp))

    # --- retrieval baselines (cached per year, max k) ---
    max_k = max(cfg.top_k)
    bkey = (year, max_k)
    if bkey not in baselines_cache:
        query = " ".join(context[: cfg.max_context_questions])
        context_copy = context[:max_k]
        bm25 = _bm25_topk(query, corpus, max_k)
        mini = _minilm_retrieve_topk(context, corpus, max_k, cfg.embed_model)
        base_out: dict[str, Any] = {"context_copy": {}, "corpus_bm25": {}, "corpus_minilm": {}}
        for name, cand in (
            ("context_copy", context_copy),
            ("corpus_bm25", bm25),
            ("corpus_minilm", mini),
        ):
            for k in cfg.top_k:
                base_out[name][str(k)] = _precision_recall(
                    cand,
                    refs,
                    k=k,
                    threshold=cfg.semantic_threshold,
                    model_name=cfg.embed_model,
                )
            base_out[name]["n_candidates"] = float(len(cand))
            base_out[name]["length_controlled"] = {
                str(k): _precision_recall(
                    cand,
                    refs,
                    k=k,
                    threshold=cfg.semantic_threshold,
                    model_name=cfg.embed_model,
                    length_words=cfg.length_control_words,
                )
                for k in cfg.top_k
            }
        baselines_cache[bkey] = base_out
        logger.info(
            "Baselines year=%d: context=%d corpus_sample=%d bm25=%d minilm=%d",
            year,
            len(context),
            len(corpus),
            len(bm25),
            len(mini),
        )

    return {
        "year": year,
        "model": model,
        "predict_run_id": predict_run_id,
        "n_predictions": len(preds),
        "n_gt": len(refs),
        "n_context": len(context),
        "n_corpus_sample": len(corpus),
        "match": match,
        "match_length_controlled": length_match,
        "length_control_words": cfg.length_control_words,
        "near_threshold": near,
        "diversity": diversity,
        "recycling": {
            "context": context_rec,
            "corpus_sample": corpus_rec,
            "corpus_sample_size": len(corpus),
        },
        "composites": composites,
        "job2_eval_snapshot": existing,
        "job2_composites_from_snapshot": job2_composites,
        "vs_baselines": {
            "delta_p50_minus_context_copy": precision_by_k.get(50, 0.0)
            - baselines_cache[bkey]["context_copy"].get("50", {}).get("precision", 0.0),
            "delta_p50_minus_corpus_bm25": precision_by_k.get(50, 0.0)
            - baselines_cache[bkey]["corpus_bm25"].get("50", {}).get("precision", 0.0),
            "delta_p50_minus_corpus_minilm": precision_by_k.get(50, 0.0)
            - baselines_cache[bkey]["corpus_minilm"].get("50", {}).get("precision", 0.0),
            "delta_p10_minus_context_copy": precision_by_k.get(10, 0.0)
            - baselines_cache[bkey]["context_copy"].get("10", {}).get("precision", 0.0),
            "delta_p10_minus_corpus_bm25": precision_by_k.get(10, 0.0)
            - baselines_cache[bkey]["corpus_bm25"].get("10", {}).get("precision", 0.0),
            "delta_p10_minus_corpus_minilm": precision_by_k.get(10, 0.0)
            - baselines_cache[bkey]["corpus_minilm"].get("10", {}).get("precision", 0.0),
        },
    }


def parse_run_specs(specs: list[str]) -> list[tuple[str, str]]:
    """'run_id' or 'run_id:label' → (run_id, label)."""
    out: list[tuple[str, str]] = []
    for s in specs:
        if ":" in s:
            rid, label = s.split(":", 1)
            out.append((rid.strip(), label.strip() or rid.strip()))
        else:
            out.append((s.strip(), s.strip()))
    return out


def discover_latest_full_runs(conn: sqlite3.Connection, models: list[str], years: list[int]) -> list[str]:
    """Pick predict run_ids that have ≥25 preds for each model×year (heuristic)."""
    rows = conn.execute(
        """
        SELECT run_id, model, target_year, COUNT(*) AS n
        FROM predictions
        GROUP BY run_id, model, target_year
        """
    ).fetchall()
    by_run: dict[str, dict[tuple[str, int], int]] = {}
    for r in rows:
        by_run.setdefault(r["run_id"], {})[(r["model"], r["target_year"])] = int(r["n"])
    good: list[str] = []
    for rid, counts in by_run.items():
        ok = all(counts.get((m, y), 0) >= 25 for m in models for y in years)
        # allow one incomplete cell (BrainGPT 2024 = 25)
        if not ok:
            missing = sum(1 for m in models for y in years if counts.get((m, y), 0) < 25)
            if missing <= 1 and all(counts.get((m, y), 0) >= 10 for m in models for y in years):
                ok = True
        if ok:
            good.append(rid)
    # chronological by run_id timestamp suffix
    return sorted(good)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default="data/neurolink.db")
    p.add_argument("--out-dir", default="eval", help="Directory for JSON output")
    p.add_argument("--out", default=None, help="Explicit output JSON path (overrides --out-dir)")
    p.add_argument("--predict-config", default="config/forecast/predict_compare.yaml")
    p.add_argument("--predict-run-id", action="append", default=None,
                   help="run_id or run_id:label (repeatable). Default: auto-discover full runs.")
    p.add_argument("--eval-run-id", default=None, help="Optional Job-2 evaluations.run_id snapshot")
    p.add_argument("--years", default="2023,2024,2025")
    p.add_argument("--models", default="literature_lora,mistral_base,braingpt")
    p.add_argument("--top-k", default="10,50")
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--near-band", type=float, default=0.05)
    p.add_argument("--length-words", type=int, default=12)
    p.add_argument("--corpus-sample", type=int, default=4000)
    p.add_argument("--max-context", type=int, default=30)
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--no-critical-only", action="store_true")
    args = p.parse_args(argv)

    cfg = StressConfig(
        db_path=args.db_path,
        out_dir=args.out_dir,
        predict_config=args.predict_config,
        test_years=[int(x) for x in args.years.split(",") if x.strip()],
        top_k=[int(x) for x in args.top_k.split(",") if x.strip()],
        models=[x.strip() for x in args.models.split(",") if x.strip()],
        semantic_threshold=args.threshold,
        near_threshold_band=args.near_band,
        length_control_words=args.length_words,
        embed_model=args.embed_model,
        critical_only=not args.no_critical_only,
        corpus_sample=args.corpus_sample,
        max_context_questions=args.max_context,
        eval_run_id=args.eval_run_id,
    )

    # Fail fast if MiniLM stack missing (user requires MiniLM; no TF-IDF fallback).
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as e:
        logger.error(
            "sentence-transformers is required (MiniLM). Install: pip install sentence-transformers"
        )
        raise SystemExit(2) from e

    logger.info("Loading MiniLM once: %s", cfg.embed_model)
    _st_model(cfg.embed_model)

    db_path = resolve_path(cfg.db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    if args.predict_run_id:
        run_specs = parse_run_specs(args.predict_run_id)
    else:
        found = discover_latest_full_runs(conn, cfg.models, cfg.test_years)
        if not found:
            logger.error("No complete predict run_ids found; pass --predict-run-id")
            return 1
        run_specs = [(rid, rid) for rid in found]
        logger.info("Auto-discovered predict runs: %s", run_specs)

    lit_cfg = _lit_cfg(cfg.predict_config, cfg.max_context_questions)

    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "config": asdict(cfg),
        "matcher_backend": "minilm",
        "embed_model": cfg.embed_model,
        "rounds": {},
        "notes": [
            "Offline stress metrics; no LM regeneration.",
            "Retrieval baselines use the same MiniLM threshold as model P@k.",
            "corpus_sample is a seeded subset of questions with year < target_year.",
            "novelty_score@k = P@k − 0.5*(context_recycling + corpus_recycling).",
            "recall_normalized_bounded = min(1, recall_normalized).",
        ],
    }

    for predict_run_id, label in run_specs:
        logger.info("=== Round %s (%s) ===", label, predict_run_id)
        baselines_cache: dict[tuple[int, int], dict[str, Any]] = {}
        cells: list[dict[str, Any]] = []
        for model in cfg.models:
            for year in cfg.test_years:
                logger.info("Evaluating %s year=%d", model, year)
                cell = evaluate_cell(
                    conn,
                    year=year,
                    model=model,
                    predict_run_id=predict_run_id,
                    cfg=cfg,
                    lit_cfg=lit_cfg,
                    baselines_cache=baselines_cache,
                )
                if cell:
                    cells.append(cell)
        # attach year-level baselines once
        baselines_by_year = {
            str(year): baselines_cache[(year, max(cfg.top_k))]
            for year in cfg.test_years
            if (year, max(cfg.top_k)) in baselines_cache
        }
        payload["rounds"][label] = {
            "predict_run_id": predict_run_id,
            "cells": cells,
            "retrieval_baselines_by_year": baselines_by_year,
        }

    out_path = Path(args.out) if args.out else resolve_path(cfg.out_dir) / f"stress_metrics_{_utc_stamp()}.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    logger.info("Wrote %s", out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
