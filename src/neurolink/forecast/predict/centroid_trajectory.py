"""Centroid approach — LLM generates questions from theme trajectory signals."""

from __future__ import annotations

import logging
import pickle
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from .llm_core import CausalLMConfig, generate_questions

logger = logging.getLogger(__name__)


@dataclass
class CentroidTrajectoryConfig:
    top_tracks: int = 8
    topics_run_id: str | None = None
    llm: CausalLMConfig = field(default_factory=CausalLMConfig)


@dataclass
class TrackTrajectory:
    track_id: int
    track_label: str
    timeline: list[tuple[int, str, float, float]]
    velocity_norm: float
    max_convergence: float
    fusion_partners: list[tuple[int, float]]


def _latest_topics_run(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT run_id FROM runs WHERE stage='topics' ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return row["run_id"] if row else None


def _load_trajectories(
    conn: sqlite3.Connection,
    target_year: int,
    run_id: str,
    top_n: int,
) -> list[TrackTrajectory]:
    rows = conn.execute(
        """
        SELECT s.track_id, s.target_year, s.velocity, s.centroid,
               tt.track_label, t.representative_terms,
               e.emergence_score, e.semantic_shift
        FROM topic_centroid_snapshots s
        JOIN topic_tracks tt ON tt.id = s.track_id
        JOIN topic_emergence e ON e.track_id = s.track_id
            AND e.target_year = s.target_year AND e.run_id = s.run_id
        JOIN topics t ON t.id = e.topic_id
        WHERE s.run_id = ? AND s.target_year <= ?
        ORDER BY s.track_id, s.target_year
        """,
        (run_id, target_year),
    ).fetchall()

    by_track: dict[int, dict] = {}
    for r in rows:
        tid = int(r["track_id"])
        if tid not in by_track:
            by_track[tid] = {
                "label": r["track_label"],
                "timeline": [],
                "velocity_norm": 0.0,
            }
        vel = pickle.loads(r["velocity"]) if r["velocity"] else None
        if vel is not None and r["target_year"] == target_year:
            by_track[tid]["velocity_norm"] = float(np.linalg.norm(vel))
        by_track[tid]["timeline"].append(
            (
                int(r["target_year"]),
                r["representative_terms"] or "",
                float(r["emergence_score"] or 0),
                float(r["semantic_shift"] or 0),
            )
        )

    dyn_rows = conn.execute(
        """
        SELECT track_id_a, track_id_b, convergence, fusion_score
        FROM topic_dynamics WHERE target_year = ? AND run_id = ?
        """,
        (target_year, run_id),
    ).fetchall()
    convergence_by: dict[int, float] = {}
    fusion_by: dict[int, list[tuple[int, float]]] = {}
    for d in dyn_rows:
        a, b = int(d["track_id_a"]), int(d["track_id_b"])
        conv = float(d["convergence"] or 0)
        fusion = float(d["fusion_score"] or 0)
        convergence_by[a] = max(convergence_by.get(a, 0), conv)
        convergence_by[b] = max(convergence_by.get(b, 0), conv)
        if fusion > 0:
            fusion_by.setdefault(a, []).append((b, fusion))
            fusion_by.setdefault(b, []).append((a, fusion))

    trajectories: list[TrackTrajectory] = []
    for tid, data in by_track.items():
        timeline = sorted(data["timeline"], key=lambda x: x[0])
        if not timeline:
            continue
        trajectories.append(
            TrackTrajectory(
                track_id=tid,
                track_label=data["label"],
                timeline=timeline,
                velocity_norm=data["velocity_norm"],
                max_convergence=convergence_by.get(tid, 0.0),
                fusion_partners=fusion_by.get(tid, []),
            )
        )

    trajectories.sort(
        key=lambda t: (
            t.timeline[-1][2] if t.timeline else 0,
            t.velocity_norm,
            t.max_convergence,
        ),
        reverse=True,
    )
    return trajectories[:top_n]


def build_trajectory_prompt(trajectories: list[TrackTrajectory], target_year: int) -> str:
    lines = [
        "You are a neuroscience research forecaster.",
        f"Given theme centroid trajectories observed until {target_year - 1}, "
        f"predict NOVEL research questions likely to emerge in {target_year}.",
        "",
        "Rules:",
        "- Generate genuinely new questions (not verbatim copies from the timeline).",
        "- Some may extend evolving themes; others may arise from converging/fusing themes.",
        "- Each line is one question ending with '?'.",
        "",
        "Theme trajectories (macro-clusters; terms approximate centroid position):",
    ]
    for tr in trajectories:
        lines.append(f"\n## {tr.track_label} (track_id={tr.track_id})")
        for year, terms, emergence, shift in tr.timeline:
            lines.append(
                f"  - {year}: terms=[{terms}] emergence={emergence:.2f} semantic_shift={shift:.2f}"
            )
        lines.append(f"  velocity_norm={tr.velocity_norm:.3f} convergence={tr.max_convergence:.3f}")
        if tr.fusion_partners:
            partners = ", ".join(f"track_{b}(fusion={f:.2f})" for b, f in tr.fusion_partners[:3])
            lines.append(f"  fusing_with: {partners}")

    lines.extend(["", f"Generate research questions for year {target_year}:"])
    return "\n".join(lines)


def predict_centroid_trajectory(
    conn: sqlite3.Connection,
    N: int,
    k: int,
    cfg: CentroidTrajectoryConfig,
) -> list[tuple[str, float]]:
    run_id = cfg.topics_run_id or _latest_topics_run(conn)
    if not run_id:
        logger.warning("centroid_trajectory: no topics run — run the topics step first")
        return []

    trajectories = _load_trajectories(conn, N, run_id, cfg.top_tracks)
    if not trajectories:
        logger.warning("centroid_trajectory: no trajectories for N=%d", N)
        return []

    prompt = build_trajectory_prompt(trajectories, N)
    try:
        generated = generate_questions(prompt, cfg.llm, k, oversample=3)
    except ImportError as e:
        logger.error("centroid_trajectory: %s", e)
        return []

    generated.sort(key=lambda x: x[1], reverse=True)
    logger.info(
        "centroid_trajectory: %d questions for N=%d (%d tracks)",
        len(generated),
        N,
        len(trajectories),
    )
    return generated[:k]


def make_centroid_predictor(cfg: CentroidTrajectoryConfig):
    def _fn(conn: sqlite3.Connection, N: int, k: int, rng) -> list[tuple[str, float]]:
        return predict_centroid_trajectory(conn, N, k, cfg)

    return _fn
