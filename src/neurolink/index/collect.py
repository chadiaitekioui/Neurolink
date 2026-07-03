"""Collect articles from PubMed API or import a local export file."""

from __future__ import annotations

import json
import logging
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


@dataclass
class CollectConfig:
    db_path: str = "data/neurolink.db"
    mesh: str = "Neurosciences"
    term: str | None = "cortex neuroscience"
    year_from: int = 2000
    year_to: int = 2025
    exclude_reviews: bool = True
    retmax: int = 60000
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


def _curl(url: str) -> str:
    proc = subprocess.run(["curl", "-s", url], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"curl failed: {proc.stderr}")
    return proc.stdout


def esearch_pmids(term: str, retmax: int, retstart: int = 0) -> tuple[list[str], int]:
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": str(retmax),
        "retstart": str(retstart),
        "retmode": "json",
    }
    url = f"{ESEARCH}?{urllib.parse.urlencode(params)}"
    data = json.loads(_curl(url))
    er = data["esearchresult"]
    return er.get("idlist", []), int(er.get("count", 0))


def efetch_abstracts(pmids: list[str]) -> str:
    if not pmids:
        return ""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "text",
    }
    url = f"{EFETCH}?{urllib.parse.urlencode(params)}"
    return _curl(url)


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


def collect_pubmed(cfg: CollectConfig, run_id: str) -> int:
    db = Database(resolve_path(cfg.db_path))
    db.init_schema()
    term = build_search_term(cfg)
    logger.info("PubMed query: %s", term)

    rows: list[ArticleRow] = []
    stored_pmids: set[str] = set()
    tried_pmids: set[str] = set()
    retstart = 0
    total_count = 0

    while len(rows) < cfg.retmax:
        remaining = cfg.retmax - len(rows)
        batch_max = min(cfg.batch_size, remaining)
        pmids, total_count = esearch_pmids(term, batch_max, retstart)
        if not pmids:
            break

        retstart += len(pmids)
        candidates = [pmid for pmid in pmids if pmid not in tried_pmids]
        tried_pmids.update(candidates)
        if not candidates:
            if retstart >= total_count:
                break
            continue

        text = efetch_abstracts(candidates)
        articles, skipped = _parse_efetch_batch(text, candidates)
        if skipped:
            logger.info(
                "Skipped %d PMID(s) without parseable abstract: %s",
                len(skipped),
                ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else ""),
            )

        for art in articles:
            if art.pmid in stored_pmids:
                continue
            rows.append(_article_row(art))
            stored_pmids.add(art.pmid)
            if len(rows) >= cfg.retmax:
                break

        if len(rows) >= cfg.retmax:
            break
        if retstart >= total_count:
            logger.info(
                "Only %d/%d articles with parseable abstracts available in PubMed",
                len(rows),
                cfg.retmax,
            )
            break
        time.sleep(cfg.delay_seconds)

    logger.info(
        "Collect: %d parseable articles stored (%d PMID candidates tried / %d available, retmax=%d)",
        len(rows),
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

    n = db.upsert_articles(rows)
    db.record_run(run_id, "collect", notes=f"{n} articles")
    logger.info("Collect finished: %d articles in database", n)
    return n


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
