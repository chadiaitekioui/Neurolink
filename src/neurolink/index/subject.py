"""Extract short research-direction subjects from segmented abstracts.

Store research directions (subject spans), not interrogative questions.
Level 1: rules (section priority, noise filters, 8–25 word span, subjectness heuristics).
Level 2: light embedding classifier (subject vs noise vs methods) via MiniLM prototypes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from ..utils.pubmed_clean import (
    Bucket,
    is_junk_sentence,
    polish_segment_field,
    structure_abstract_sections,
)

logger = logging.getLogger(__name__)

SubjectLabel = Literal["subject", "noise", "methods"]

# Prefer aim/objective sections over historical background.
_SECTION_PRIORITY: dict[str, int] = {
    "OBJECTIVES": 0,
    "AIM": 1,
    "AIMS": 1,
    "PURPOSE": 2,
    "INTRODUCTION": 3,
    "BACKGROUND": 4,
}

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


ExtractionMode = Literal["rules", "llm", "hybrid"]


@dataclass
class SubjectLlmConfig:
    """Causal LM settings for index-time direction extraction (Mistral base, no LoRA)."""

    base_model: str = "mistralai/Mistral-7B-v0.1"
    use_4bit: bool = True
    max_new_tokens: int = 64
    temperature: float = 0.0
    prompt_max_length: int = 4096
    abstract_max_chars: int = 3500
    llm_subjectness_floor: float = 0.55


@dataclass
class SubjectConfig:
    """Subject extraction / filtering for index stages."""

    enabled: bool = True
    # rules = L1+L2 span pick; llm = Mistral base extraction; hybrid = llm then rules fallback.
    extraction_mode: ExtractionMode = "rules"
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
    # Soft floor so very clean but low-impact items are not zeroed.
    impact_subjectness_floor: float = 0.15


@dataclass
class SubjectResult:
    text: str
    subjectness: float
    label: SubjectLabel
    source_section: str | None = None
    reject_reason: str | None = None


@dataclass
class SubjectClassifier:
    """Level-2 prototype classifier (lazy MiniLM). Falls back to heuristics if unavailable."""

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
                "sentence-transformers unavailable — subject classifier disabled (level-1 only)"
            )
            self._failed = True
            return False

        try:
            model = SentenceTransformer(self.model_name)
            self._model = model
            for label, texts in _PROTOTYPES.items():
                emb = model.encode(texts, normalize_embeddings=True)
                self._centroids[label] = np.mean(emb, axis=0)
                norm = float(np.linalg.norm(self._centroids[label]))
                if norm > 0:
                    self._centroids[label] = self._centroids[label] / norm
            logger.info("Subject classifier loaded: %s", self.model_name)
            return True
        except Exception as e:
            logger.warning("Subject classifier load failed (%s) — level-1 only", e)
            self._failed = True
            return False

    def classify(self, text: str) -> tuple[SubjectLabel, float]:
        """Return (label, confidence in [0, 1])."""
        if not text.strip() or not self._ensure_loaded():
            return "subject", 0.0

        import numpy as np

        assert self._model is not None
        emb = self._model.encode([text], normalize_embeddings=True)[0]
        scores: dict[SubjectLabel, float] = {}
        for label, centroid in self._centroids.items():
            scores[label] = float(np.dot(emb, centroid))
        best = max(scores, key=scores.get)
        # Map cosine [-1,1] → [0,1] confidence vs runner-up.
        ordered = sorted(scores.values(), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        conf = max(0.0, min(1.0, (ordered[0] + 1.0) / 2.0 * (0.5 + 0.5 * max(margin, 0.0))))
        return best, conf


_CLASSIFIER_CACHE: dict[str, SubjectClassifier] = {}


def get_subject_classifier(model_name: str) -> SubjectClassifier:
    if model_name not in _CLASSIFIER_CACHE:
        _CLASSIFIER_CACHE[model_name] = SubjectClassifier(model_name=model_name)
    return _CLASSIFIER_CACHE[model_name]


def split_sentences(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", s).strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]


def is_subject_noise(text: str) -> bool:
    """Aggressive metadata / junk filter for subject candidates."""
    return is_junk_sentence(text)


def compress_to_subject_span(
    sentence: str,
    *,
    min_words: int = 8,
    max_words: int = 25,
    absolute_min_words: int = 5,
) -> str | None:
    """Trim a sentence to a short research-direction span."""
    s = polish_segment_field(sentence)
    s = _LEADING_FILLER.sub("", s)
    s = _TRAILING_CLAUSE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" .;:")
    if not s:
        return None
    # Drop trailing question mark — subjects are not interrogatives.
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
    # Prefer spans reaching min_words when possible; keep shorter if already compact.
    if len(words) < min_words and len(words) < absolute_min_words:
        return None
    return s


def heuristic_subjectness(text: str) -> float:
    """Level-1 subjectness score in [0, 1]."""
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


def _priority_for_section(section: str | None) -> int:
    if not section:
        return 50
    return _SECTION_PRIORITY.get(section.upper(), 40)


def _candidate_sentences_from_sections(
    sections: list[tuple[str, Bucket | None, str | None]],
) -> list[tuple[str, str | None]]:
    """Yield (sentence, section) ordered by section priority then appearance."""
    scored: list[tuple[int, int, str, str | None]] = []
    order = 0
    for content, bucket, section in sections:
        if bucket == "results":
            continue
        for sent in split_sentences(content):
            if is_subject_noise(sent):
                continue
            scored.append((_priority_for_section(section), order, sent, section))
            order += 1
    scored.sort(key=lambda x: (x[0], x[1]))
    return [(sent, section) for _, _, sent, section in scored]


def _extract_subject_rules(
    text: str,
    *,
    cfg: SubjectConfig,
    title: str | None = None,
    classifier: SubjectClassifier | None = None,
) -> SubjectResult | None:
    sections = structure_abstract_sections(text)
    if len(sections) == 1 and sections[0][1] is None and sections[0][2] is None:
        # Unstructured blob (already a question bucket): treat whole as one section.
        candidates = [
            (sent, None)
            for sent in split_sentences(text)
            if not is_subject_noise(sent)
        ]
    else:
        candidates = _candidate_sentences_from_sections(sections)

    if title and not candidates:
        span = compress_to_subject_span(
            title,
            min_words=cfg.min_words,
            max_words=cfg.max_words,
            absolute_min_words=cfg.absolute_min_words,
        )
        if span:
            candidates = [(span, "TITLE")]

    clf = classifier
    if cfg.use_level2_classifier and clf is None:
        clf = get_subject_classifier(cfg.classifier_model)

    best: SubjectResult | None = None
    for sent, section in candidates:
        span = compress_to_subject_span(
            sent,
            min_words=cfg.min_words,
            max_words=cfg.max_words,
            absolute_min_words=cfg.absolute_min_words,
        )
        if not span:
            continue
        score = heuristic_subjectness(span)
        label: SubjectLabel = "subject"
        if clf is not None and cfg.use_level2_classifier:
            label, conf = clf.classify(span)
            if label == "noise":
                score *= 0.15
            elif label == "methods":
                score *= 0.35
            else:
                score = min(1.0, score + 0.15 * conf)
        if score < cfg.min_subjectness:
            continue
        if label != "subject" and score < cfg.min_subjectness + 0.15:
            continue
        cand = SubjectResult(text=span, subjectness=score, label=label, source_section=section)
        if best is None or cand.subjectness > best.subjectness:
            best = cand

    if best is None and title:
        span = compress_to_subject_span(
            title,
            min_words=cfg.min_words,
            max_words=cfg.max_words,
            absolute_min_words=cfg.absolute_min_words,
        )
        if span and not is_subject_noise(span):
            score = heuristic_subjectness(span)
            if score >= cfg.min_subjectness:
                best = SubjectResult(
                    text=span,
                    subjectness=score,
                    label="subject",
                    source_section="TITLE",
                )
    return best


def extract_subject(
    text: str,
    *,
    cfg: SubjectConfig | None = None,
    title: str | None = None,
    classifier: SubjectClassifier | None = None,
) -> SubjectResult | None:
    """Extract one research-direction subject from an abstract or question-bucket text."""
    cfg = cfg or SubjectConfig()
    if not cfg.enabled:
        return None
    if not (text or "").strip() and not (title or "").strip():
        return None

    if cfg.extraction_mode in ("llm", "hybrid"):
        from .subject_llm import extract_subject_llm

        llm_result = extract_subject_llm(
            title,
            text,
            cfg=cfg,
            classifier=classifier,
        )
        if llm_result is not None and not llm_result.reject_reason:
            return llm_result
        if cfg.extraction_mode == "llm":
            return llm_result

    if not (text or "").strip():
        return None
    return _extract_subject_rules(
        text,
        cfg=cfg,
        title=title,
        classifier=classifier,
    )


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
