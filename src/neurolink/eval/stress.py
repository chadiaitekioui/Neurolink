"""Stress / beyond-retrieval metrics (MiniLM offline, no LM regeneration).

Computed inside ``run_eval`` and written to ``evaluations``:
  - retrieval baselines (context copy, corpus BM25, corpus MiniLM)
  - length-controlled MiniLM P@k / R@k
  - near-threshold match diagnostics
  - prediction diversity
  - novelty / extension composites
  - beyond-retrieval KPIs (conditional_beyond, incremental_recall)
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..forecast.predict.direction_filter import is_clean_gt_text
from ..forecast.predict.literature_lora import LiteratureLoraConfig, list_context_questions

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9α-ωΑ-Ω]+(?:[-'][A-Za-z0-9α-ωΑ-Ω]+)?")
_EMBED_CACHE: dict[str, np.ndarray] = {}
_ST_MODEL_CACHE: dict[str, object] = {}


@dataclass
class StressConfig:
    """Options for the stress / beyond-retrieval block inside ``run_eval``."""

    semantic_threshold: float = 0.50
    near_threshold_band: float = 0.05
    length_control_words: int = 12
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    critical_only: bool = True
    filter_gt_noise: bool = True
    max_context_questions: int = 30
    corpus_sample: int = 4000
    corpus_seed: int = 42
    top_k: list[int] = field(default_factory=lambda: [10, 50])


def _st_model(model_name: str):
    """Load SentenceTransformer once per process (MiniLM required)."""
    if model_name in _ST_MODEL_CACHE:
        return _ST_MODEL_CACHE[model_name]
    from sentence_transformers import SentenceTransformer

    logger.info("Loading SentenceTransformer: %s", model_name)
    model = SentenceTransformer(model_name)
    _ST_MODEL_CACHE[model_name] = model
    return model


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


def load_gt(conn: sqlite3.Connection, year: int, cfg: StressConfig) -> list[str]:
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


def load_preds(
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


def sample_corpus(
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


def _gt_covered_mask(
    preds: list[str],
    refs: list[str],
    *,
    k: int,
    threshold: float,
    model_name: str,
) -> np.ndarray:
    """Boolean mask over GT: True if some pred in top-k matches that GT."""
    n_gt = len(refs)
    top = preds[:k]
    if not top or n_gt == 0:
        return np.zeros(n_gt, dtype=bool)
    sims = _encode(top, model_name) @ _encode(refs, model_name).T
    return sims.max(axis=0) >= threshold


def _beyond_retrieval(
    model_mask: np.ndarray,
    baseline_mask: np.ndarray,
) -> dict[str, float]:
    """Success = GT hit by the model that the retrieval baseline missed."""
    n_gt = int(model_mask.shape[0])
    if n_gt == 0:
        return {
            "n_gt": 0.0,
            "n_gt_model": 0.0,
            "n_gt_baseline": 0.0,
            "n_gt_beyond_baseline": 0.0,
            "n_gt_union": 0.0,
            "n_gt_room_after_baseline": 0.0,
            "frac_gt_model": 0.0,
            "frac_gt_baseline": 0.0,
            "frac_gt_beyond_baseline": 0.0,
            "frac_gt_union": 0.0,
            "incremental_recall": 0.0,
            "conditional_beyond": 0.0,
            "beyond_of_model_hits": 0.0,
        }
    n_model = int(model_mask.sum())
    n_base = int(baseline_mask.sum())
    beyond = model_mask & ~baseline_mask
    union = model_mask | baseline_mask
    n_beyond = int(beyond.sum())
    n_union = int(union.sum())
    room = max(n_gt - n_base, 0)
    return {
        "n_gt": float(n_gt),
        "n_gt_model": float(n_model),
        "n_gt_baseline": float(n_base),
        "n_gt_beyond_baseline": float(n_beyond),
        "n_gt_union": float(n_union),
        "n_gt_room_after_baseline": float(room),
        "frac_gt_model": n_model / n_gt,
        "frac_gt_baseline": n_base / n_gt,
        "frac_gt_beyond_baseline": n_beyond / n_gt,
        "frac_gt_union": n_union / n_gt,
        "incremental_recall": (n_union - n_base) / n_gt,
        "conditional_beyond": (n_beyond / room) if room > 0 else 0.0,
        "beyond_of_model_hits": (n_beyond / n_model) if n_model > 0 else 0.0,
    }


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


def evaluate_cell(
    conn: sqlite3.Connection,
    *,
    year: int,
    model: str,
    predict_run_id: str,
    cfg: StressConfig,
    lit_cfg: LiteratureLoraConfig,
    baselines_cache: dict[tuple[int, int], dict[str, Any]],
    preds: list[str] | None = None,
    refs: list[str] | None = None,
) -> dict[str, Any] | None:
    if preds is None:
        preds = load_preds(conn, year, model, predict_run_id)
    if refs is None:
        refs = load_gt(conn, year, cfg)
    preds = [(p or "").strip() for p in preds if (p or "").strip()]
    refs = [(r or "").strip() for r in refs if (r or "").strip()]
    if not preds or not refs:
        logger.warning("Skip %s %d: preds=%d refs=%d", model, year, len(preds), len(refs))
        return None

    context = list_context_questions(conn, year, lit_cfg)
    corpus = sample_corpus(conn, year, cfg.corpus_sample, cfg.corpus_seed)

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

    max_k = max(cfg.top_k)
    bkey = (year, max_k)
    if bkey not in baselines_cache:
        query = " ".join(context[: cfg.max_context_questions])
        context_copy = context[:max_k]
        bm25 = _bm25_topk(query, corpus, max_k)
        mini = _minilm_retrieve_topk(context, corpus, max_k, cfg.embed_model)
        base_out: dict[str, Any] = {
            "context_copy": {},
            "corpus_bm25": {},
            "corpus_minilm": {},
            "_masks": {},
            "_cands": {
                "context_copy": context_copy,
                "corpus_bm25": bm25,
                "corpus_minilm": mini,
            },
        }
        for name, cand in (
            ("context_copy", context_copy),
            ("corpus_bm25", bm25),
            ("corpus_minilm", mini),
        ):
            base_out["_masks"][name] = {}
            for k in cfg.top_k:
                base_out[name][str(k)] = _precision_recall(
                    cand,
                    refs,
                    k=k,
                    threshold=cfg.semantic_threshold,
                    model_name=cfg.embed_model,
                )
                base_out["_masks"][name][str(k)] = _gt_covered_mask(
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

    beyond: dict[str, Any] = {}
    for k in cfg.top_k:
        model_mask = _gt_covered_mask(
            preds,
            refs,
            k=k,
            threshold=cfg.semantic_threshold,
            model_name=cfg.embed_model,
        )
        beyond[str(k)] = {}
        for bname in ("corpus_minilm", "corpus_bm25", "context_copy"):
            bmask = baselines_cache[bkey]["_masks"][bname][str(k)]
            beyond[str(k)][bname] = _beyond_retrieval(model_mask, bmask)

    baselines_public = {
        name: {kk: vv for kk, vv in baselines_cache[bkey][name].items()}
        for name in ("context_copy", "corpus_bm25", "corpus_minilm")
    }

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
        "beyond_retrieval": beyond,
        "vs_baselines": {
            "delta_p50_minus_context_copy": precision_by_k.get(50, 0.0)
            - baselines_public["context_copy"].get("50", {}).get("precision", 0.0),
            "delta_p50_minus_corpus_bm25": precision_by_k.get(50, 0.0)
            - baselines_public["corpus_bm25"].get("50", {}).get("precision", 0.0),
            "delta_p50_minus_corpus_minilm": precision_by_k.get(50, 0.0)
            - baselines_public["corpus_minilm"].get("50", {}).get("precision", 0.0),
            "delta_p10_minus_context_copy": precision_by_k.get(10, 0.0)
            - baselines_public["context_copy"].get("10", {}).get("precision", 0.0),
            "delta_p10_minus_corpus_bm25": precision_by_k.get(10, 0.0)
            - baselines_public["corpus_bm25"].get("10", {}).get("precision", 0.0),
            "delta_p10_minus_corpus_minilm": precision_by_k.get(10, 0.0)
            - baselines_public["corpus_minilm"].get("10", {}).get("precision", 0.0),
        },
        "_baselines_public": baselines_public,
        "_precision_by_k": precision_by_k,
    }


def flatten_stress_metrics(cell: dict[str, Any]) -> list[tuple[str, int, float]]:
    """Flatten a stress cell into ``(metric, k, value)`` rows for ``evaluations``.

    Skips duplicate classic ``precision@k`` / ``recall@k`` (already written by ``run_eval``).
    Primary success KPI: ``beyond_conditional_corpus_minilm`` (= conditional_beyond).
    """
    rows: list[tuple[str, int, float]] = []

    def add(metric: str, value: float, k: int = 0) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        rows.append((metric, k, float(value)))

    rec = cell.get("recycling") or {}
    add("stress_recycling_context", rec.get("context", 0.0))
    add("stress_recycling_corpus", rec.get("corpus_sample", 0.0))
    add("stress_corpus_sample_size", float(rec.get("corpus_sample_size", 0)))

    for k_str, m in (cell.get("match_length_controlled") or {}).items():
        k = int(k_str)
        add("stress_precision_len_ctrl", m["precision"], k)
        add("stress_recall_len_ctrl", m["recall"], k)
        add("stress_recall_norm_len_ctrl", m["recall_normalized"], k)

    for k_str, n in (cell.get("near_threshold") or {}).items():
        k = int(k_str)
        add("stress_near_frac_matched", n["frac_matched"], k)
        add("stress_near_frac_band", n["frac_near_threshold"], k)
        add("stress_near_mean_best_sim", n["mean_best_sim"], k)

    for k_str, d in (cell.get("diversity") or {}).items():
        k = int(k_str)
        add("stress_div_trigram", d["unique_trigram_ratio"], k)
        add("stress_div_pairwise", d["mean_pairwise_cosine_distance"], k)
        add("stress_div_entropy", d["embedding_entropy"], k)

    for key, val in (cell.get("composites") or {}).items():
        # novelty_score@50 → metric=stress_novelty, k=50
        if key.startswith("novelty_score@"):
            add("stress_novelty", float(val), int(key.rsplit("@", 1)[1]))
        elif key.startswith("extension_vs_context@"):
            add("stress_extension_vs_context", float(val), int(key.rsplit("@", 1)[1]))
        elif key.startswith("extension_vs_corpus@"):
            add("stress_extension_vs_corpus", float(val), int(key.rsplit("@", 1)[1]))

    baselines = cell.get("_baselines_public") or {}
    for bname in ("context_copy", "corpus_bm25", "corpus_minilm"):
        b = baselines.get(bname) or {}
        for k_str, m in b.items():
            if k_str in ("n_candidates", "length_controlled") or not isinstance(m, dict):
                continue
            if "precision" not in m:
                continue
            k = int(k_str)
            add(f"baseline_{bname}_precision", m["precision"], k)
            add(f"baseline_{bname}_recall", m["recall"], k)

    for k_str, by_base in (cell.get("beyond_retrieval") or {}).items():
        k = int(k_str)
        for bname, stats in by_base.items():
            add(f"beyond_conditional_{bname}", stats["conditional_beyond"], k)
            add(f"beyond_incremental_{bname}", stats["incremental_recall"], k)
            add(f"beyond_n_gt_{bname}", stats["n_gt_beyond_baseline"], k)
            add(f"beyond_of_model_{bname}", stats["beyond_of_model_hits"], k)
            add(f"beyond_frac_gt_{bname}", stats["frac_gt_beyond_baseline"], k)

    precision_by_k: dict[int, float] = cell.get("_precision_by_k") or {}
    for k, p in precision_by_k.items():
        for bname in ("context_copy", "corpus_bm25", "corpus_minilm"):
            bp = (baselines.get(bname) or {}).get(str(k), {}).get("precision")
            if bp is None:
                continue
            add(f"delta_p_vs_{bname}", p - float(bp), k)

    return rows


def stress_config_from_eval(
    *,
    top_k: list[int],
    semantic_threshold: float,
    embed_model: str,
    critical_only: bool,
    filter_gt_noise: bool,
    corpus_sample: int = 4000,
    near_threshold_band: float = 0.05,
    length_control_words: int = 12,
    max_context_questions: int = 30,
) -> StressConfig:
    return StressConfig(
        top_k=list(top_k),
        semantic_threshold=semantic_threshold,
        embed_model=embed_model,
        critical_only=critical_only,
        filter_gt_noise=filter_gt_noise,
        corpus_sample=corpus_sample,
        near_threshold_band=near_threshold_band,
        length_control_words=length_control_words,
        max_context_questions=max_context_questions,
    )
