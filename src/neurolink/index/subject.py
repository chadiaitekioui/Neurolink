"""Extract and persist research *directions* from PubMed abstracts (LLM + MiniLM).

Index stage formerly called ``segment``: Mistral-Instruct turns title+abstract into
one short research direction stored in ``article_segments`` (legacy table name).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ..db import Database
from ..utils.config import load_config, make_run_id, resolve_path
from ..utils.pubmed_clean import is_junk_sentence, polish_field
from ..utils.torch_device import resolve_torch_device

logger = logging.getLogger(__name__)

SubjectLabel = Literal["subject", "noise", "methods"]

_RESULTS_VERBS = re.compile(
    r"\b(?:we (?:found|show(?:ed)?|demonstrate(?:d)?|observed|report(?:ed)?)"
    r"|significantly (?:increased|decreased|higher|lower)"
    r"|our (?:results|findings))\b",
    re.IGNORECASE,
)
_AIM_CUES = re.compile(
    r"\b(?:aim(?:ed)?|purpose|objective|investigate|examine|assess|evaluate|"
    r"role of|mechanisms? of|whether|how|effect of|contribution of)\b",
    re.IGNORECASE,
)
_NEURO_CUES = re.compile(
    r"\b(?:cortex|cortical|cerebell(?:um|ar)|neuron|synaptic|plasticity|"
    r"hippocamp|thalamic|motor|cognitive|EEG|fMRI|TMS|spike|axon|glia|"
    r"dopamine|serotonin|GABA|NMDA|LTP|memory|attention|language)\b",
    re.IGNORECASE,
)
_LEADING_FILLER = re.compile(
    r"^(?:(?:here,?\s+)?we (?:aimed to|sought to|set out to|investigated|studied|"
    r"examined|assessed|evaluated)\s+|the (?:aim|purpose|objective) of this "
    r"(?:study|work|paper) (?:was|is) to\s+|this study (?:aimed to|investigates?)\s+)",
    re.IGNORECASE,
)
_TRAILING_CLAUSE = re.compile(r"\s*(?:,|;|:)\s+(?:using|with|in order to|by)\b.*$", re.IGNORECASE)

_PROTOTYPES: dict[SubjectLabel, list[str]] = {
    "subject": [
        "Role of cerebellar nuclei in control of motor cortex via thalamus",
        "Cortical mechanisms of working memory in prefrontal networks",
        "Synaptic plasticity in hippocampal CA1 during spatial learning",
        "Effect of dopaminergic signaling on decision making",
        "Contribution of glial cells to neuroinflammation in cortex",
    ],
    "noise": [
        "Comment in Nature doi 10.1038 contributed equally author information",
        "Update of bioRxiv Collaborators Department of Neurology University",
        "PMID DOI PMCID Electronic address hospital affiliation Korea",
        "Lee HS Suh BC Kim JK Author information Department of Neurology",
    ],
    "methods": [
        "We enrolled sixty patients and recorded EEG during the task",
        "Statistical analysis used Student t-test and ANOVA",
        "Mice were anesthetized with isoflurane and placed in a stereotaxic frame",
        "Data were analyzed using Python and reported as mean plus SEM",
    ],
}


@dataclass
class SubjectLlmConfig:
    """Causal LM settings for index-time direction extraction (instruct, no LoRA)."""

    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_4bit: bool = True
    use_chat_template: bool = True
    max_new_tokens: int = 64
    temperature: float = 0.0
    prompt_max_length: int = 4096
    abstract_max_chars: int = 3500
    llm_subjectness_floor: float = 0.55
    # "topic" = original bench22 noun-phrase prompt; "specific" = precise research direction.
    prompt_style: str = "topic"


@dataclass
class SubjectConfig:
    """Filters / scoring shared by direction extraction and impact."""

    enabled: bool = True
    extraction_mode: str = "llm"
    llm: SubjectLlmConfig = field(default_factory=SubjectLlmConfig)
    min_words: int = 8
    max_words: int = 25
    absolute_min_words: int = 5
    min_subjectness: float = 0.35
    year_min: int = 2000
    year_max: int = 2027
    use_level2_classifier: bool = True
    classifier_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    weight_impact_by_subjectness: bool = True
    impact_subjectness_floor: float = 0.15


@dataclass
class DirectionConfig:
    """Batch LLM extraction of research directions into ``article_segments``."""

    db_path: str = "data/neurolink.db"
    device: str = "auto"  # cpu | cuda | auto
    commit_every: int = 100
    subject: SubjectConfig = field(default_factory=SubjectConfig)


@dataclass
class SubjectResult:
    text: str
    subjectness: float
    label: SubjectLabel
    source_section: str | None = None
    reject_reason: str | None = None


@dataclass
class SubjectClassifier:
    """Prototype classifier (lazy MiniLM). Falls back to heuristics if unavailable."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _centroids: dict[SubjectLabel, object] = field(default_factory=dict, repr=False)
    _model: object | None = field(default=None, repr=False)
    _failed: bool = field(default=False, repr=False)

    def _ensure_loaded(self) -> bool:
        if self._centroids:
            return True
        if self._failed:
            return False
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers unavailable — subject classifier disabled"
            )
            self._failed = True
            return False

        try:
            model = SentenceTransformer(self.model_name)
            self._model = model
            for label, texts in _PROTOTYPES.items():
                emb = model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )
                self._centroids[label] = np.mean(emb, axis=0)
                norm = float(np.linalg.norm(self._centroids[label]))
                if norm > 0:
                    self._centroids[label] = self._centroids[label] / norm
            logger.info("Subject classifier loaded: %s", self.model_name)
            return True
        except Exception as e:
            logger.warning("Subject classifier load failed (%s)", e)
            self._failed = True
            return False

    def classify(self, text: str) -> tuple[SubjectLabel, float]:
        if not text.strip() or not self._ensure_loaded():
            return "subject", 0.0

        import numpy as np

        assert self._model is not None
        emb = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores: dict[SubjectLabel, float] = {}
        for label, centroid in self._centroids.items():
            scores[label] = float(np.dot(emb, centroid))
        ordered = sorted(scores.values(), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        best = max(scores, key=scores.get)
        conf = max(
            0.0,
            min(1.0, (ordered[0] + 1.0) / 2.0 * (0.5 + 0.5 * max(margin, 0.0))),
        )
        return best, conf


_CLASSIFIER_CACHE: dict[str, SubjectClassifier] = {}


def get_subject_classifier(model_name: str) -> SubjectClassifier:
    if model_name not in _CLASSIFIER_CACHE:
        _CLASSIFIER_CACHE[model_name] = SubjectClassifier(model_name=model_name)
    return _CLASSIFIER_CACHE[model_name]


def is_subject_noise(text: str) -> bool:
    return is_junk_sentence(text)


def compress_to_subject_span(
    sentence: str,
    *,
    min_words: int = 8,
    max_words: int = 25,
    absolute_min_words: int = 5,
) -> str | None:
    s = polish_field(sentence)
    s = _LEADING_FILLER.sub("", s)
    s = _TRAILING_CLAUSE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" .;:")
    if not s:
        return None
    if s.endswith("?"):
        s = s[:-1].rstrip()
    words = s.split()
    if len(words) < absolute_min_words:
        return None
    if len(words) > max_words:
        s = " ".join(words[:max_words])
        words = s.split()
    if len(words) < absolute_min_words:
        return None
    return s


def heuristic_subjectness(text: str) -> float:
    if not text or is_subject_noise(text):
        return 0.0
    words = text.split()
    n = len(words)
    score = 0.35
    if 8 <= n <= 25:
        score += 0.25
    elif 5 <= n < 8 or 25 < n <= 40:
        score += 0.10
    else:
        score -= 0.15
    if _AIM_CUES.search(text):
        score += 0.15
    if _NEURO_CUES.search(text):
        score += 0.15
    if _RESULTS_VERBS.search(text):
        score -= 0.25
    if text[:1].islower():
        score -= 0.05
    if text.endswith("?"):
        score -= 0.05
    return float(max(0.0, min(1.0, score)))


def extract_subject(
    text: str,
    *,
    cfg: SubjectConfig | None = None,
    title: str | None = None,
    classifier: SubjectClassifier | None = None,
) -> SubjectResult | None:
    """Extract one research direction via LLM."""
    cfg = cfg or SubjectConfig()
    if not cfg.enabled:
        return None
    if not (text or "").strip() and not (title or "").strip():
        return None
    from .subject_llm import extract_subject_llm

    return extract_subject_llm(title, text, cfg=cfg, classifier=classifier)


def year_in_range(year: int | None, cfg: SubjectConfig) -> bool:
    if year is None:
        return False
    return cfg.year_min <= int(year) <= cfg.year_max


def weighted_impact(impact_score: float | None, subjectness: float, cfg: SubjectConfig) -> float:
    base = float(impact_score or 0.0)
    if not cfg.weight_impact_by_subjectness:
        return base
    weight = max(cfg.impact_subjectness_floor, float(subjectness))
    return base * weight


def load_pmids_file(path: str | Path) -> list[str]:
    """Load PMIDs from JSON or plain text."""
    p = resolve_path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("[") or text.startswith("{"):
        data = json.loads(text)
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return [
                    str(row["pmid"])
                    for row in data
                    if isinstance(row, dict) and row.get("pmid")
                ]
            return [str(x) for x in data if str(x).strip()]
        if isinstance(data, dict):
            if "pmids" in data:
                return [str(x) for x in data["pmids"] if str(x).strip()]
            if "samples" in data:
                return [
                    str(row["pmid"])
                    for row in data["samples"]
                    if isinstance(row, dict) and row.get("pmid")
                ]
        raise ValueError(f"Unsupported PMID JSON in {p}")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _fetch_articles_for_directions(
    conn,
    *,
    limit: int | None = None,
    pmids: list[str] | None = None,
    force: bool = False,
) -> tuple[list, int]:
    """Return (worklist rows, already_done_count). Table: ``article_segments``."""
    if pmids:
        placeholders = ",".join("?" * len(pmids))
        if force:
            rows = conn.execute(
                f"SELECT pmid, title, abstract, text_work FROM articles WHERE pmid IN ({placeholders})",
                pmids,
            ).fetchall()
            return list(rows), 0
        rows = conn.execute(
            f"""
            SELECT a.pmid, a.title, a.abstract, a.text_work
            FROM articles a
            LEFT JOIN article_segments s ON a.pmid = s.pmid
            WHERE a.pmid IN ({placeholders}) AND s.pmid IS NULL
            """,
            pmids,
        ).fetchall()
        return list(rows), len(pmids) - len(rows)

    already = conn.execute("SELECT COUNT(*) FROM article_segments").fetchone()[0]
    if force:
        q = """
            SELECT pmid, title, abstract, text_work
            FROM articles
            ORDER BY pmid
        """
        if limit is not None and limit > 0:
            rows = conn.execute(q + " LIMIT ?", (limit,)).fetchall()
        else:
            rows = conn.execute(q).fetchall()
        return list(rows), 0

    q = """
        SELECT a.pmid, a.title, a.abstract, a.text_work
        FROM articles a
        LEFT JOIN article_segments s ON a.pmid = s.pmid
        WHERE s.pmid IS NULL
        ORDER BY a.pmid
    """
    if limit is not None and limit > 0:
        rows = conn.execute(q + " LIMIT ?", (limit,)).fetchall()
    else:
        rows = conn.execute(q).fetchall()
    return list(rows), int(already)


def extract_direction(
    abstract: str,
    *,
    title: str | None = None,
    subject_cfg: SubjectConfig | None = None,
    classifier: SubjectClassifier | None = None,
) -> tuple[str | None, str, float | None]:
    """One abstract → (direction text, results bucket, subjectness)."""
    from .subject_llm import extract_subject_llm, results_from_sections

    cfg = subject_cfg or SubjectConfig()
    text = (abstract or "").strip()
    extracted = extract_subject_llm(title, text, cfg=cfg, classifier=classifier)
    results = results_from_sections(text) if text else ""
    if extracted is None or extracted.reject_reason:
        return None, results, 0.0 if extracted is None else extracted.subjectness
    return extracted.text, results, extracted.subjectness


def run_directions(
    config_path: str | DirectionConfig,
    run_id: str | None = None,
    *,
    limit: int | None = None,
    pmids: list[str] | None = None,
    force: bool = False,
) -> int:
    """Batch-extract research directions into ``article_segments``.

    With ``force=True``, re-extracts all articles (or the PMID list / limit),
    upserting existing rows. Without ``force``, only articles missing a segment.
    """
    cfg = (
        load_config(config_path, DirectionConfig)
        if isinstance(config_path, str)
        else config_path
    )
    if cfg.subject.extraction_mode != "llm":
        logger.warning(
            "subject.extraction_mode=%r ignored — directions always use LLM extraction",
            cfg.subject.extraction_mode,
        )
    db = Database(resolve_path(cfg.db_path))
    run_id = run_id or make_run_id("direction")
    now = datetime.now(timezone.utc).isoformat()
    commit_every = max(1, int(cfg.commit_every))

    with db.connect() as conn:
        total_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        articles, already = _fetch_articles_for_directions(
            conn, limit=limit, pmids=pmids, force=force
        )

    if not articles:
        logger.info("Directions: nothing to do (%d already extracted)", already)
        db.record_run(run_id, "direction", notes="0 new")
        return 0

    scope = ""
    if pmids:
        scope = f", pmids={len(pmids)}"
    elif limit:
        scope = f", limit={limit}"
    logger.info(
        "Directions: %d/%d done (%d remaining%s) — LLM",
        already,
        total_articles,
        len(articles),
        scope,
    )

    device = resolve_torch_device(cfg.device)
    logger.info(
        "Direction extraction: model=%s device=%s",
        cfg.subject.llm.base_model,
        device,
    )

    classifier = None
    if cfg.subject.enabled and cfg.subject.use_level2_classifier:
        classifier = get_subject_classifier(cfg.subject.classifier_model)

    n = 0
    n_ok = 0
    with db.connect() as conn:
        for row in articles:
            text = (row["text_work"] or row["abstract"] or "").strip()
            direction, results, qc = extract_direction(
                text,
                title=row["title"],
                subject_cfg=cfg.subject,
                classifier=classifier,
            )
            if qc is not None and qc > 0 and direction:
                n_ok += 1
            conn.execute(
                """
                INSERT INTO article_segments
                    (pmid, question, results, segmentation_method, qc_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pmid) DO UPDATE SET
                    question=excluded.question, results=excluded.results,
                    segmentation_method=excluded.segmentation_method,
                    qc_score=excluded.qc_score, updated_at=excluded.updated_at
                """,
                (row["pmid"], direction, results, "llm+direction", qc, now),
            )
            n += 1
            if n % commit_every == 0:
                conn.commit()
                done = already + n
                logger.info(
                    "Directions: %d/%d (%.1f%%)",
                    done,
                    total_articles,
                    100.0 * done / max(1, total_articles),
                )

    try:
        from ..forecast.predict.llm_core import release_gpu_memory

        release_gpu_memory()
        logger.info("Released LLM VRAM after direction stage")
    except ImportError:
        pass

    db.record_run(
        run_id,
        "direction",
        notes=f"{n} new / {already + n} total; ok={n_ok}",
    )
    logger.info(
        "Directions: %d new articles (%d total); extracted=%d",
        n,
        already + n,
        n_ok,
    )
    return n
