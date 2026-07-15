"""Collect articles from PubMed API or import a local export file."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from ..db import ArticleRow, Database
from ..utils.config import load_config, make_run_id, resolve_path
from ..utils.pubmed_parse import ParsedArticle, parse_pubmed_text

logger = logging.getLogger(__name__)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_FETCH_RETRIES = 6


@dataclass
class CollectConfig:
    db_path: str = "data/neurolink.db"
    mesh: str = "Neurosciences"
    term: str | None = "cortex neuroscience"
    year_from: int = 2000
    year_to: int = 2026
    exclude_reviews: bool = True
    retmax: int = 100000
    batch_size: int = 200
    email: str | None = None
    delay_seconds: float = 0.34


def build_search_term(cfg: CollectConfig) -> str:
    base = cfg.term if cfg.term else f"{cfg.mesh}[MeSH Terms]"
    if cfg.exclude_reviews and "review" not in base.lower():
        base = f"({base}) NOT review[pt]"
    return (
        f'{base} AND ("{cfg.year_from:04d}/01/01"[PDAT] : "{cfg.year_to:04d}/12/31"[PDAT])'
    )


def _ncbi_params(cfg: CollectConfig) -> dict[str, str]:
    params: dict[str, str] = {"tool": "neurolink"}
    email = (cfg.email or os.environ.get("NCBI_EMAIL", "")).strip()
    if email:
        params["email"] = email
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    return params


def _fetch_url(url: str, *, expect_json: bool = False, label: str = "NCBI") -> str:
    last_exc: Exception | None = None
    for attempt in range(_FETCH_RETRIES):
        proc = subprocess.run(
            ["curl", "-sS", "-f", "--max-time", "120", url],
            capture_output=True,
            text=True,
            check=False,
        )
        body = proc.stdout
        if proc.returncode != 0:
            err = proc.stderr.strip() or (body.strip()[:300] if body else "empty body")
            last_exc = RuntimeError(f"{label}: curl exit {proc.returncode}: {err}")
        elif not body.strip():
            last_exc = RuntimeError(f"{label}: empty response")
        elif expect_json:
            try:
                json.loads(body)
            except json.JSONDecodeError as exc:
                last_exc = RuntimeError(
                    f"{label}: invalid JSON ({exc}); preview={body[:200]!r}"
                )
            else:
                return body
        else:
            return body
        logger.warning("%s (retry %d/%d)", last_exc, attempt + 1, _FETCH_RETRIES)
        time.sleep(min(2.0**attempt, 60.0))
    raise last_exc or RuntimeError(f"{label}: fetch failed")


def esearch_pmids(
    cfg: CollectConfig,
    term: str,
    retmax: int,
    retstart: int = 0,
) -> tuple[list[str], int]:
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": str(retmax),
        "retstart": str(retstart),
        "retmode": "json",
        **_ncbi_params(cfg),
    }
    url = f"{ESEARCH}?{urllib.parse.urlencode(params)}"
    data = json.loads(_fetch_url(url, expect_json=True, label="esearch"))
    er = data["esearchresult"]
    return er.get("idlist", []), int(er.get("count", 0))


def efetch_abstracts(cfg: CollectConfig, pmids: list[str]) -> str:
    if not pmids:
        return ""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "text",
        **_ncbi_params(cfg),
    }
    url = f"{EFETCH}?{urllib.parse.urlencode(params)}"
    return _fetch_url(url, label="efetch")


def _article_row(art: ParsedArticle) -> ArticleRow:
    return ArticleRow(
        pmid=art.pmid,
        year=art.year,
        doi=art.doi,
        title=art.title,
        abstract=art.abstract,
        text_work=art.abstract,
    )


def _parse_efetch_batch(text: str, requested_pmids: list[str]) -> tuple[list[ParsedArticle], list[str]]:
    articles = list(parse_pubmed_text(text))
    parsed_ids = {art.pmid for art in articles}
    skipped = [pmid for pmid in requested_pmids if pmid not in parsed_ids]
    return articles, skipped


def _existing_pmids(db: Database) -> set[str]:
    with db.connect() as conn:
        return {row[0] for row in conn.execute("SELECT pmid FROM articles")}


def collect_pubmed(cfg: CollectConfig, run_id: str) -> int:
    db = Database(resolve_path(cfg.db_path))
    db.init_schema()
    term = build_search_term(cfg)
    logger.info("PubMed query: %s", term)

    stored_pmids = _existing_pmids(db)
    if stored_pmids:
        logger.info("Resuming collect: %d articles already in database", len(stored_pmids))

    tried_pmids: set[str] = set()
    retstart = 0
    total_count = 0
    stored_total = len(stored_pmids)

    while stored_total < cfg.retmax:
        remaining = cfg.retmax - stored_total
        batch_max = min(cfg.batch_size, remaining)
        pmids, total_count = esearch_pmids(cfg, term, batch_max, retstart)
        if not pmids:
            break

        retstart += len(pmids)
        candidates = [pmid for pmid in pmids if pmid not in tried_pmids]
        tried_pmids.update(candidates)
        if not candidates:
            if retstart >= total_count:
                break
            continue

        text = efetch_abstracts(cfg, candidates)
        articles, skipped = _parse_efetch_batch(text, candidates)
        if skipped:
            logger.info(
                "Skipped %d PMID(s) without parseable abstract: %s",
                len(skipped),
                ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else ""),
            )

        batch_rows: list[ArticleRow] = []
        for art in articles:
            if art.pmid in stored_pmids:
                continue
            batch_rows.append(_article_row(art))
            stored_pmids.add(art.pmid)
            stored_total += 1
            if stored_total >= cfg.retmax:
                break

        if batch_rows:
            db.upsert_articles(batch_rows)

        if stored_total >= cfg.retmax:
            break
        if retstart >= total_count:
            logger.info(
                "Only %d/%d articles with parseable abstracts available in PubMed",
                stored_total,
                cfg.retmax,
            )
            break
        time.sleep(cfg.delay_seconds)

    logger.info(
        "Collect: %d parseable articles stored (%d PMID candidates tried / %d available, retmax=%d)",
        stored_total,
        len(tried_pmids),
        total_count,
        cfg.retmax,
    )
    if total_count < cfg.retmax:
        logger.info(
            "PubMed returned fewer matches than retmax (%d < %d) for the query/date range",
            total_count,
            cfg.retmax,
        )

    db.record_run(run_id, "collect", notes=f"{stored_total} articles")
    logger.info("Collect finished: %d articles in database", stored_total)
    return stored_total


def import_pubmed_text_file(cfg: CollectConfig, text_path: Path, run_id: str) -> int:
    db = Database(resolve_path(cfg.db_path))
    db.init_schema()
    text = text_path.read_text(encoding="utf-8")
    rows = [
        ArticleRow(
            pmid=a.pmid,
            year=a.year,
            doi=a.doi,
            title=a.title,
            abstract=a.abstract,
            text_work=a.abstract,
        )
        for a in parse_pubmed_text(text)
    ]
    n = db.upsert_articles(rows)
    db.record_run(run_id, "import", notes=f"from {text_path}")
    logger.info("Import finished: %d articles from %s", n, text_path)
    return n


def run_collect(config_path: str | Path | CollectConfig) -> int:
    cfg = load_config(config_path, CollectConfig)
    return collect_pubmed(cfg, make_run_id("collect"))
