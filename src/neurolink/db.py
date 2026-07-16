"""SQLite schema and CRUD operations."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterable

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    config_hash TEXT,
    created_at TEXT NOT NULL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS articles (
    pmid TEXT PRIMARY KEY,
    year INTEGER,
    doi TEXT,
    title TEXT,
    abstract TEXT,
    intro TEXT,
    text_work TEXT,
    mesh_terms TEXT,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS article_segments (
    pmid TEXT PRIMARY KEY REFERENCES articles(pmid),
    question TEXT,
    results TEXT,
    segmentation_method TEXT,
    qc_score REAL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS citations (
    pmid TEXT PRIMARY KEY REFERENCES articles(pmid),
    citation_count INTEGER,
    horizon_years INTEGER,
    source TEXT,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS article_impact (
    pmid TEXT PRIMARY KEY REFERENCES articles(pmid),
    impact_score REAL,
    is_critical INTEGER
);

CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid TEXT REFERENCES articles(pmid),
    question_text TEXT NOT NULL,
    year INTEGER,
    embedding BLOB,
    impact_score REAL,
    is_critical INTEGER
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_year INTEGER,
    model TEXT,
    rank INTEGER,
    question_predicted TEXT,
    score REAL,
    run_id TEXT REFERENCES runs(run_id),
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_year INTEGER,
    model TEXT,
    metric TEXT,
    k INTEGER,
    value REAL,
    ci_low REAL,
    ci_high REAL,
    run_id TEXT REFERENCES runs(run_id)
);
"""


@dataclass
class ArticleRow:
    pmid: str
    year: int | None = None
    doi: str | None = None
    title: str | None = None
    abstract: str | None = None
    intro: str | None = None
    text_work: str | None = None
    mesh_terms: str | None = None
    fetched_at: str | None = None


class Database:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.path, timeout=30)
        # WAL + mmap on Lustre ($WORK) can trigger Bus error on login nodes — use DELETE there.
        journal = os.environ.get("NEUROLINK_SQLITE_JOURNAL", "WAL").strip() or "WAL"
        conn.execute(f"PRAGMA journal_mode={journal}")
        # Avoid mmap issues on Lustre by disabling memory mapping.
        conn.execute("PRAGMA mmap_size=0")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def upsert_articles(self, rows: Iterable[ArticleRow]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        n = 0
        with self.connect() as conn:
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO articles (pmid, year, doi, title, abstract, intro, text_work, mesh_terms, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pmid) DO UPDATE SET
                        year=excluded.year, doi=excluded.doi, title=excluded.title,
                        abstract=excluded.abstract, intro=excluded.intro,
                        text_work=excluded.text_work, mesh_terms=excluded.mesh_terms,
                        fetched_at=excluded.fetched_at
                    """,
                    (
                        row.pmid,
                        row.year,
                        row.doi,
                        row.title,
                        row.abstract,
                        row.intro,
                        row.text_work or row.abstract,
                        row.mesh_terms,
                        row.fetched_at or now,
                    ),
                )
                n += 1
        return n

    def record_run(
        self,
        run_id: str,
        stage: str,
        config_hash: str | None = None,
        notes: str = "",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, stage, config_hash, created_at, notes) VALUES (?,?,?,?,?)",
                (run_id, stage, config_hash, datetime.now(timezone.utc).isoformat(), notes),
            )

