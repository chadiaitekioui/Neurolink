"""SQLite schema and CRUD operations."""

from __future__ import annotations

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

CREATE TABLE IF NOT EXISTS topic_tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_label TEXT NOT NULL,
    first_seen_year INTEGER,
    birth_target_year INTEGER,
    last_seen_year INTEGER,
    centroid BLOB,
    run_id TEXT REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_label TEXT,
    track_id INTEGER REFERENCES topic_tracks(id),
    year INTEGER,
    count INTEGER,
    representative_terms TEXT,
    run_id TEXT REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS topic_emergence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id INTEGER REFERENCES topics(id),
    track_id INTEGER REFERENCES topic_tracks(id),
    target_year INTEGER,
    growth_rate REAL,
    novelty_score REAL,
    atypicality_score REAL,
    semantic_shift REAL,
    emergence_score REAL,
    run_id TEXT REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS topic_track_yearly (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER REFERENCES topic_tracks(id),
    calendar_year INTEGER NOT NULL,
    target_year INTEGER NOT NULL,
    count INTEGER NOT NULL,
    run_id TEXT REFERENCES runs(run_id),
    UNIQUE(track_id, calendar_year, target_year, run_id)
);

CREATE TABLE IF NOT EXISTS topic_centroid_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER REFERENCES topic_tracks(id),
    target_year INTEGER NOT NULL,
    centroid BLOB NOT NULL,
    velocity BLOB,
    micro_count INTEGER,
    run_id TEXT REFERENCES runs(run_id),
    UNIQUE(track_id, target_year, run_id)
);

CREATE TABLE IF NOT EXISTS topic_dynamics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id_a INTEGER REFERENCES topic_tracks(id),
    track_id_b INTEGER REFERENCES topic_tracks(id),
    target_year INTEGER NOT NULL,
    proximity REAL,
    convergence REAL,
    fusion_score REAL,
    run_id TEXT REFERENCES runs(run_id),
    UNIQUE(track_id_a, track_id_b, target_year, run_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_year INTEGER,
    model TEXT,
    rank INTEGER,
    question_predicted TEXT,
    topic_id INTEGER,
    score REAL,
    run_id TEXT REFERENCES runs(run_id),
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS question_topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER REFERENCES questions(id),
    topic_id INTEGER REFERENCES topics(id),
    target_year INTEGER,
    run_id TEXT REFERENCES runs(run_id)
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
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._migrate_schema(conn)

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        """Add dynamic-topic columns/tables when upgrading older databases."""
        topic_cols = {row[1] for row in conn.execute("PRAGMA table_info(topics)")}
        if "track_id" not in topic_cols:
            conn.execute("ALTER TABLE topics ADD COLUMN track_id INTEGER REFERENCES topic_tracks(id)")

        emergence_cols = {row[1] for row in conn.execute("PRAGMA table_info(topic_emergence)")}
        if "track_id" not in emergence_cols:
            conn.execute("ALTER TABLE topic_emergence ADD COLUMN track_id INTEGER REFERENCES topic_tracks(id)")
        if "semantic_shift" not in emergence_cols:
            conn.execute("ALTER TABLE topic_emergence ADD COLUMN semantic_shift REAL")

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS topic_tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_label TEXT NOT NULL,
                first_seen_year INTEGER,
                birth_target_year INTEGER,
                last_seen_year INTEGER,
                centroid BLOB,
                run_id TEXT REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS topic_track_yearly (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER REFERENCES topic_tracks(id),
                calendar_year INTEGER NOT NULL,
                target_year INTEGER NOT NULL,
                count INTEGER NOT NULL,
                run_id TEXT REFERENCES runs(run_id),
                UNIQUE(track_id, calendar_year, target_year, run_id)
            );
            CREATE TABLE IF NOT EXISTS topic_centroid_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER REFERENCES topic_tracks(id),
                target_year INTEGER NOT NULL,
                centroid BLOB NOT NULL,
                velocity BLOB,
                micro_count INTEGER,
                run_id TEXT REFERENCES runs(run_id),
                UNIQUE(track_id, target_year, run_id)
            );
            CREATE TABLE IF NOT EXISTS topic_dynamics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id_a INTEGER REFERENCES topic_tracks(id),
                track_id_b INTEGER REFERENCES topic_tracks(id),
                target_year INTEGER NOT NULL,
                proximity REAL,
                convergence REAL,
                fusion_score REAL,
                run_id TEXT REFERENCES runs(run_id),
                UNIQUE(track_id_a, track_id_b, target_year, run_id)
            );
            """
        )

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

