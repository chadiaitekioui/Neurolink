"""Layer 1 — indexing: collect → segment → impact → embed."""

from .collect import (
    CollectConfig,
    build_search_term,
    collect_pubmed,
    esearch_pmids,
    import_pubmed_text_file,
    run_collect,
)
from .embed import EmbedConfig, run_embed
from .impact import ImpactConfig, citation_rate, fetch_citation_count, run_impact, years_since_publication
from .pipeline import (
    IndexCounts,
    IndexPipelineConfig,
    check_index_ready,
    get_index_counts,
    is_index_ready,
    run_index,
)
from .segment import SegmentConfig, run_segment
from .subject import (
    SubjectConfig,
    SubjectResult,
    compress_to_subject_span,
    extract_subject,
    heuristic_subjectness,
)
from ..utils.pubmed_clean import clean_abstract, structure_abstract, structure_abstract_sections
from ..utils.pubmed_parse import ParsedArticle, parse_pubmed_text

__all__ = [
    "CollectConfig",
    "EmbedConfig",
    "ImpactConfig",
    "IndexCounts",
    "IndexPipelineConfig",
    "ParsedArticle",
    "SegmentConfig",
    "SubjectConfig",
    "SubjectResult",
    "build_search_term",
    "check_index_ready",
    "clean_abstract",
    "collect_pubmed",
    "compress_to_subject_span",
    "esearch_pmids",
    "extract_subject",
    "citation_rate",
    "fetch_citation_count",
    "get_index_counts",
    "heuristic_subjectness",
    "is_index_ready",
    "years_since_publication",
    "import_pubmed_text_file",
    "parse_pubmed_text",
    "run_collect",
    "run_embed",
    "run_impact",
    "run_index",
    "run_segment",
    "structure_abstract",
    "structure_abstract_sections",
]
