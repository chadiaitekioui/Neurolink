"""Dynamic topic tracking, emergence scores, and topics pipeline step."""

from __future__ import annotations

import logging
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..db import Database
from ..utils.config import infer_test_years, load_config, make_run_id, resolve_path

logger = logging.getLogger(__name__)


# ── Track dynamics (core algorithms) ──────────────────────────────────────────

@dataclass
class Track:
    track_id: int
    centroid: np.ndarray
    first_seen_year: int
    representative_terms: str = ""
    prev_centroid: np.ndarray | None = None


@dataclass
class TrackMetrics:
    track_id: int
    growth_rate: float
    novelty_score: float
    atypicality_score: float
    semantic_shift: float
    emergence_score: float
    count_n1: int
    representative_terms: str
    is_birth: bool


@dataclass
class TrackPairDynamics:
    track_id_a: int
    track_id_b: int
    proximity: float
    convergence: float
    fusion_score: float


@dataclass
class DynamicTopicsConfig:
    growth_weight: float = 0.35
    novelty_weight: float = 0.25
    atypicality_weight: float = 0.25
    semantic_shift_weight: float = 0.15
    history_window: int = 5
    match_threshold: float = 0.55
    n_topics: int = 15
    fusion_proximity_threshold: float = 0.75
    convergence_threshold: float = 0.05


def compute_centroids(vectors: np.ndarray, labels: list[int]) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    labels_arr = np.asarray(labels)
    for lbl in sorted(set(labels)):
        if lbl < 0:
            continue
        mask = labels_arr == lbl
        centroids[int(lbl)] = vectors[mask].mean(axis=0)
    return centroids


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def match_clusters_to_tracks(
    new_centroids: dict[int, np.ndarray],
    tracks: dict[int, Track],
    min_similarity: float,
) -> tuple[dict[int, int], list[int]]:
    if not new_centroids:
        return {}, []
    if not tracks:
        return {}, list(new_centroids.keys())

    local_ids = list(new_centroids.keys())
    track_ids = list(tracks.keys())
    local_mat = np.vstack([new_centroids[i] for i in local_ids])
    track_mat = np.vstack([tracks[i].centroid for i in track_ids])
    sim = _cosine_matrix(local_mat, track_mat)
    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[int, int] = {}
    matched_locals: set[int] = set()
    for row, col in zip(row_ind, col_ind):
        local_id = local_ids[row]
        track_id = track_ids[col]
        if sim[row, col] >= min_similarity:
            mapping[local_id] = track_id
            matched_locals.add(local_id)

    new_locals = [local_id for local_id in local_ids if local_id not in matched_locals]
    return mapping, new_locals


def register_new_tracks(
    new_locals: list[int],
    centroids: dict[int, np.ndarray],
    terms_by_local: dict[int, str],
    years: list[int],
    labels: list[int],
    target_year: int,
    tracks: dict[int, Track],
    next_track_id: int,
) -> tuple[dict[int, int], int]:
    mapping = dict()
    for local_id in new_locals:
        track_years = [y for lbl, y in zip(labels, years) if lbl == local_id and y is not None]
        first_year = min(track_years) if track_years else target_year - 1
        tracks[next_track_id] = Track(
            track_id=next_track_id,
            centroid=centroids[local_id],
            first_seen_year=first_year,
            representative_terms=terms_by_local.get(local_id, ""),
        )
        mapping[local_id] = next_track_id
        next_track_id += 1
    return mapping, next_track_id


def _yearly_counts(
    track_id: int,
    labels: list[int],
    years: list[int],
    local_to_track: dict[int, int],
) -> Counter:
    counts: Counter = Counter()
    for lbl, year in zip(labels, years):
        if lbl < 0 or year is None:
            continue
        if local_to_track.get(lbl) == track_id:
            counts[int(year)] += 1
    return counts


def compute_track_metrics(
    track: Track,
    labels: list[int],
    years: list[int],
    texts: list[str],
    local_to_track: dict[int, int],
    target_year: int,
    cfg: DynamicTopicsConfig,
    n_tracks: int,
) -> TrackMetrics:
    counts = _yearly_counts(track.track_id, labels, years, local_to_track)
    count_n1 = counts.get(target_year - 1, 0)
    hist = [counts.get(target_year - 1 - i, 0) for i in range(2, cfg.history_window + 1)]
    mean_hist = float(np.mean(hist)) if hist else 0.0
    growth = (count_n1 / mean_hist) if mean_hist > 0 else float(count_n1)

    texts_n1 = {
        text
        for text, lbl, year in zip(texts, labels, years)
        if local_to_track.get(lbl) == track.track_id and year == target_year - 1
    }
    old_texts = {
        text
        for text, lbl, year in zip(texts, labels, years)
        if local_to_track.get(lbl) == track.track_id and year is not None and year < target_year - 4
    }
    novelty = len(texts_n1 - old_texts) / len(texts_n1) if texts_n1 else 0.0

    total_years = max(len(years), 1)
    freq = sum(counts.get(y, 0) for y in range(target_year - 1)) / total_years
    atypicality = 1.0 - min(freq * max(n_tracks, 1), 1.0)

    if track.prev_centroid is not None:
        sim = float(
            np.dot(track.centroid, track.prev_centroid)
            / (np.linalg.norm(track.centroid) * np.linalg.norm(track.prev_centroid) + 1e-9)
        )
        semantic_shift = 1.0 - sim
    else:
        semantic_shift = 0.0

    emergence = (
        cfg.growth_weight * min(growth, 10) / 10
        + cfg.novelty_weight * novelty
        + cfg.atypicality_weight * atypicality
        + cfg.semantic_shift_weight * semantic_shift
    )
    is_birth = track.first_seen_year >= target_year - 1

    return TrackMetrics(
        track_id=track.track_id,
        growth_rate=growth,
        novelty_score=novelty,
        atypicality_score=atypicality,
        semantic_shift=semantic_shift,
        emergence_score=emergence,
        count_n1=count_n1,
        representative_terms=track.representative_terms,
        is_birth=is_birth,
    )


def update_track_centroids(
    tracks: dict[int, Track],
    local_to_track: dict[int, int],
    centroids: dict[int, np.ndarray],
) -> None:
    for local_id, track_id in local_to_track.items():
        track = tracks[track_id]
        track.prev_centroid = track.centroid.copy()
        track.centroid = centroids[local_id]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_velocity(
    current: np.ndarray,
    previous: np.ndarray | None,
) -> np.ndarray | None:
    if previous is None:
        return None
    return current - previous


def compute_track_pair_dynamics(
    track_a: Track,
    track_b: Track,
    cfg: DynamicTopicsConfig,
) -> TrackPairDynamics:
    proximity = cosine_similarity(track_a.centroid, track_b.centroid)
    convergence = 0.0
    if track_a.prev_centroid is not None and track_b.prev_centroid is not None:
        prev_dist = 1.0 - cosine_similarity(track_a.prev_centroid, track_b.prev_centroid)
        curr_dist = 1.0 - proximity
        convergence = prev_dist - curr_dist

    fusion = 0.0
    if convergence >= cfg.convergence_threshold and proximity >= cfg.fusion_proximity_threshold:
        fusion = convergence * proximity

    return TrackPairDynamics(
        track_id_a=track_a.track_id,
        track_id_b=track_b.track_id,
        proximity=proximity,
        convergence=convergence,
        fusion_score=fusion,
    )


def compute_all_pair_dynamics(
    tracks: dict[int, Track],
    cfg: DynamicTopicsConfig,
) -> list[TrackPairDynamics]:
    ids = sorted(tracks.keys())
    out: list[TrackPairDynamics] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            out.append(compute_track_pair_dynamics(tracks[a], tracks[b], cfg))
    return out


def assign_questions_to_tracks(
    labels: list[int],
    local_to_track: dict[int, int],
) -> dict[int, int]:
    assignments: dict[int, int] = {}
    for idx, lbl in enumerate(labels):
        if lbl < 0:
            continue
        track_id = local_to_track.get(lbl)
        if track_id is not None:
            assignments[idx] = track_id
    return assignments


# ── Topics pipeline step ──────────────────────────────────────────────────────

@dataclass
class TopicsConfig:
    db_path: str = "data/neurolink.db"
    min_questions: int = 5
    n_topics: int = 15
    growth_weight: float = 0.35
    novelty_weight: float = 0.25
    atypicality_weight: float = 0.25
    semantic_shift_weight: float = 0.15
    history_window: int = 5
    match_threshold: float = 0.55
    cluster_method: str = "auto"  # auto | kmeans


def _load_embeddings(conn, year_max: int) -> tuple[list[int], list[str], list[int], np.ndarray]:
    rows = conn.execute(
        """
        SELECT id, question_text, year, embedding FROM questions
        WHERE year IS NOT NULL AND year <= ? AND embedding IS NOT NULL
        ORDER BY id
        """,
        (year_max,),
    ).fetchall()
    ids, texts, years, vecs = [], [], [], []
    for r in rows:
        ids.append(r["id"])
        texts.append(r["question_text"])
        years.append(r["year"])
        vecs.append(pickle.loads(r["embedding"]))
    if not vecs:
        return [], [], [], np.zeros((0, 1))
    return ids, texts, years, np.vstack(vecs)


def _cluster_topics(
    vectors: np.ndarray,
    texts: list[str],
    n_topics: int,
    cluster_method: str = "auto",
) -> list[int]:
    n = len(texts)
    if n < n_topics:
        return list(range(n))
    if cluster_method != "kmeans":
        try:
            from bertopic import BERTopic

            topic_model = BERTopic(nr_topics=min(n_topics, n - 1), verbose=False)
            topics, _ = topic_model.fit_transform(texts, vectors)
            return [int(t) for t in topics]
        except ImportError:
            pass

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=min(n_topics, n), random_state=42, n_init=10)
    return [int(x) for x in km.fit_predict(vectors)]


def _topic_terms(texts: list[str], labels: list[int], topic: int, top_n: int = 8) -> str:
    from sklearn.feature_extraction.text import CountVectorizer

    subset = [t for t, l in zip(texts, labels) if l == topic]
    if not subset:
        return ""
    cv = CountVectorizer(max_features=200, stop_words="english")
    try:
        X = cv.fit_transform(subset)
        sums = np.asarray(X.sum(axis=0)).ravel()
        terms = cv.get_feature_names_out()
        top = sums.argsort()[::-1][:top_n]
        return ", ".join(terms[i] for i in top if sums[i] > 0)
    except ValueError:
        return ""


def _dynamic_cfg(cfg: TopicsConfig) -> DynamicTopicsConfig:
    return DynamicTopicsConfig(
        growth_weight=cfg.growth_weight,
        novelty_weight=cfg.novelty_weight,
        atypicality_weight=cfg.atypicality_weight,
        semantic_shift_weight=cfg.semantic_shift_weight,
        history_window=cfg.history_window,
        match_threshold=cfg.match_threshold,
        n_topics=cfg.n_topics,
    )


def _clear_run_topic_data(conn, run_id: str) -> None:
    conn.execute("DELETE FROM question_topics WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM topic_dynamics WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM topic_centroid_snapshots WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM topic_emergence WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM topic_track_yearly WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM topics WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM topic_tracks WHERE run_id = ?", (run_id,))


def _persist_track(
    conn,
    track: Track,
    target_year: int,
    run_id: str,
    last_seen_year: int,
) -> int:
    row = conn.execute(
        "SELECT id FROM topic_tracks WHERE run_id = ? AND track_label = ?",
        (run_id, f"track_{track.track_id}"),
    ).fetchone()
    centroid_blob = pickle.dumps(track.centroid)
    if row:
        conn.execute(
            """
            UPDATE topic_tracks
            SET last_seen_year = ?, centroid = ?
            WHERE id = ?
            """,
            (last_seen_year, centroid_blob, row["id"]),
        )
        return int(row["id"])

    cur = conn.execute(
        """
        INSERT INTO topic_tracks
        (track_label, first_seen_year, birth_target_year, last_seen_year, centroid, run_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            f"track_{track.track_id}",
            track.first_seen_year,
            target_year,
            last_seen_year,
            centroid_blob,
            run_id,
        ),
    )
    return int(cur.lastrowid)


def run_topics(config_path: str, target_year: int | None = None, run_id: str | None = None) -> int:
    cfg = load_config(config_path, TopicsConfig)
    dyn_cfg = _dynamic_cfg(cfg)
    db = Database(resolve_path(cfg.db_path))
    run_id = run_id or make_run_id("topics")
    total = 0

    with db.connect() as conn:
        years_test = [target_year] if target_year else infer_test_years(conn, list(range(2018, 2025)))
        years_test = sorted(years_test)
        _clear_run_topic_data(conn, run_id)

        tracks: dict[int, Track] = {}
        next_track_id = 1
        topic_db_ids: dict[int, int] = {}

        for N in years_test:
            ids, texts, years, vectors = _load_embeddings(conn, N - 1)
            if len(ids) < cfg.min_questions:
                logger.warning("Year N=%d: not enough questions (%d)", N, len(ids))
                continue

            labels = _cluster_topics(vectors, texts, cfg.n_topics, cfg.cluster_method)
            centroids = compute_centroids(vectors, labels)
            terms_by_local = {local_id: _topic_terms(texts, labels, local_id) for local_id in centroids}

            local_to_track, new_locals = match_clusters_to_tracks(centroids, tracks, cfg.match_threshold)
            new_mapping, next_track_id = register_new_tracks(
                new_locals,
                centroids,
                terms_by_local,
                years,
                labels,
                N,
                tracks,
                next_track_id,
            )
            local_to_track.update(new_mapping)

            for local_id, track_id in local_to_track.items():
                track = tracks[track_id]
                if terms_by_local.get(local_id):
                    track.representative_terms = terms_by_local[local_id]

            update_track_centroids(tracks, local_to_track, centroids)

            conn.execute("DELETE FROM topic_dynamics WHERE target_year = ? AND run_id = ?", (N, run_id))
            conn.execute("DELETE FROM topic_centroid_snapshots WHERE target_year = ? AND run_id = ?", (N, run_id))
            conn.execute("DELETE FROM topic_emergence WHERE target_year = ? AND run_id = ?", (N, run_id))
            conn.execute("DELETE FROM question_topics WHERE target_year = ? AND run_id = ?", (N, run_id))
            conn.execute("DELETE FROM topic_track_yearly WHERE target_year = ? AND run_id = ?", (N, run_id))
            conn.execute(
                "DELETE FROM topics WHERE year = ? AND run_id = ?",
                (N - 1, run_id),
            )

            active_track_ids = set(local_to_track.values())
            yearly_by_track: dict[int, Counter] = defaultdict(Counter)
            for lbl, year in zip(labels, years):
                if lbl < 0 or year is None:
                    continue
                track_id = local_to_track.get(lbl)
                if track_id is not None:
                    yearly_by_track[track_id][int(year)] += 1

            topic_db_ids.clear()
            for track_id in sorted(active_track_ids):
                track = tracks[track_id]
                db_track_id = _persist_track(
                    conn,
                    track,
                    N,
                    run_id,
                    last_seen_year=max(yearly_by_track[track_id]) if yearly_by_track[track_id] else N - 1,
                )

                metrics = compute_track_metrics(
                    track,
                    labels,
                    years,
                    texts,
                    local_to_track,
                    N,
                    dyn_cfg,
                    n_tracks=len(active_track_ids),
                )

                cur = conn.execute(
                    """
                    INSERT INTO topics
                    (topic_label, track_id, year, count, representative_terms, run_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"track_{track_id}",
                        db_track_id,
                        N - 1,
                        metrics.count_n1,
                        metrics.representative_terms,
                        run_id,
                    ),
                )
                topic_row_id = int(cur.lastrowid)
                conn.execute(
                    """
                    INSERT INTO topic_emergence
                    (topic_id, track_id, target_year, growth_rate, novelty_score,
                     atypicality_score, semantic_shift, emergence_score, run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        topic_row_id,
                        db_track_id,
                        N,
                        metrics.growth_rate,
                        metrics.novelty_score,
                        metrics.atypicality_score,
                        metrics.semantic_shift,
                        metrics.emergence_score,
                        run_id,
                    ),
                )

                for calendar_year, count in yearly_by_track[track_id].items():
                    conn.execute(
                        """
                        INSERT INTO topic_track_yearly
                        (track_id, calendar_year, target_year, count, run_id)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (db_track_id, calendar_year, N, count, run_id),
                    )

                topic_db_ids[track_id] = topic_row_id
                total += 1

            pair_dynamics = compute_all_pair_dynamics(
                {tid: tracks[tid] for tid in active_track_ids},
                dyn_cfg,
            )
            for pair in pair_dynamics:
                db_a = conn.execute(
                    "SELECT id FROM topic_tracks WHERE run_id = ? AND track_label = ?",
                    (run_id, f"track_{pair.track_id_a}"),
                ).fetchone()
                db_b = conn.execute(
                    "SELECT id FROM topic_tracks WHERE run_id = ? AND track_label = ?",
                    (run_id, f"track_{pair.track_id_b}"),
                ).fetchone()
                if not db_a or not db_b:
                    continue
                conn.execute(
                    """
                    INSERT INTO topic_dynamics
                    (track_id_a, track_id_b, target_year, proximity, convergence, fusion_score, run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        db_a["id"],
                        db_b["id"],
                        N,
                        pair.proximity,
                        pair.convergence,
                        pair.fusion_score,
                        run_id,
                    ),
                )

            for track_id in sorted(active_track_ids):
                track = tracks[track_id]
                db_track_id = conn.execute(
                    "SELECT id FROM topic_tracks WHERE run_id = ? AND track_label = ?",
                    (run_id, f"track_{track_id}"),
                ).fetchone()
                if not db_track_id:
                    continue
                velocity = compute_velocity(track.centroid, track.prev_centroid)
                micro_count = sum(
                    1 for lbl, tid in local_to_track.items() if tid == track_id
                )
                conn.execute(
                    """
                    INSERT INTO topic_centroid_snapshots
                    (track_id, target_year, centroid, velocity, micro_count, run_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        db_track_id["id"],
                        N,
                        pickle.dumps(track.centroid),
                        pickle.dumps(velocity) if velocity is not None else None,
                        micro_count,
                        run_id,
                    ),
                )

            assignments = assign_questions_to_tracks(labels, local_to_track)
            for q_idx, track_id in assignments.items():
                topic_row_id = topic_db_ids.get(track_id)
                if topic_row_id is None:
                    continue
                conn.execute(
                    """
                    INSERT INTO question_topics (question_id, topic_id, target_year, run_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (ids[q_idx], topic_row_id, N, run_id),
                )

            logger.info(
                "N=%d: %d active tracks (%d new clusters)",
                N,
                len(active_track_ids),
                len(new_locals),
            )

    db.record_run(run_id, "topics", notes=f"{total} dynamic topic-year rows")
    logger.info("Dynamic topics: %d records", total)
    return total
