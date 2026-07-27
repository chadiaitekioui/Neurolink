"""Structured audit logging for literature LM generation (raw → parse → filter → keep)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Grep-friendly prefix for cluster log audits.
AUDIT_TAG = "GEN_AUDIT"

_MAX_PREVIEW = 600
_MAX_SAMPLES = 5


@dataclass
class GenerationAudit:
    """Per (model, target_year) generation funnel — filled during predict."""

    model: str = ""
    target_year: int = 0
    requested_k: int = 0
    generation_mode: str = "batch"
    filter_outputs: bool = True
    adapter: str | None = None
    temperature: float = 0.0

    # Prompt
    prompt_chars: int = 0
    prompt_tokens_full: int = 0
    prompt_tokens_used: int = 0
    prompt_truncated: bool = False
    context_lines: int = 0
    context_year: int | None = None

    # Raw decode (completion only, after prompt)
    raw_sequences: int = 0
    raw_chars: int = 0
    max_new_tokens: int = 0
    num_return_sequences: int = 0
    do_sample: bool = False

    # Parse
    parsed_candidates: int = 0
    parse_fallback_blob: bool = False

    # Filter (batch path uses filter_directions_audited; iterative aggregates here)
    rejection_counts: dict[str, int] = field(default_factory=dict)
    rejected_samples: list[dict[str, str]] = field(default_factory=list)

    # Output
    after_filter: int = 0
    returned: int = 0
    kept_samples: list[str] = field(default_factory=list)

    # Iterative-only
    attempts_budget: int = 0
    attempts_used: int = 0

    error: str | None = None

    def record_rejection(self, reason: str, text: str) -> None:
        self.rejection_counts[reason] = self.rejection_counts.get(reason, 0) + 1
        if len(self.rejected_samples) < _MAX_SAMPLES:
            self.rejected_samples.append(
                {"reason": reason, "text": truncate_preview(text, 200)}
            )

    def record_kept(self, text: str) -> None:
        if len(self.kept_samples) < _MAX_SAMPLES:
            self.kept_samples.append(truncate_preview(text, 200))

    def set_raw_outputs(self, outputs: list[str]) -> None:
        self.raw_sequences = len(outputs)
        self.raw_chars = sum(len(o) for o in outputs)

    def summary_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Compact nested blobs for one-line JSON log.
        d["raw_preview"] = getattr(self, "_raw_preview", "")
        return d

    def log(self) -> None:
        """Emit audit lines at INFO (summary + previews)."""
        payload = {
            "tag": AUDIT_TAG,
            "model": self.model,
            "year": self.target_year,
            "mode": self.generation_mode,
            "k": self.requested_k,
            "adapter": self.adapter,
            "filter": self.filter_outputs,
            "prompt_chars": self.prompt_chars,
            "prompt_tokens": f"{self.prompt_tokens_used}/{self.prompt_tokens_full}",
            "prompt_truncated": self.prompt_truncated,
            "context_lines": self.context_lines,
            "raw_seqs": self.raw_sequences,
            "raw_chars": self.raw_chars,
            "max_new_tokens": self.max_new_tokens,
            "parsed": self.parsed_candidates,
            "parse_fallback": self.parse_fallback_blob,
            "after_filter": self.after_filter,
            "returned": self.returned,
            "rejections": self.rejection_counts,
            "attempts": f"{self.attempts_used}/{self.attempts_budget}"
            if self.attempts_budget
            else None,
            "error": self.error,
        }
        logger.info("%s %s", AUDIT_TAG, json.dumps(payload, ensure_ascii=False, sort_keys=True))

        raw_preview = getattr(self, "_raw_preview", "")
        if raw_preview:
            logger.info(
                "%s_RAW model=%s year=%s chars=%d preview=%r",
                AUDIT_TAG,
                self.model,
                self.target_year,
                self.raw_chars,
                raw_preview,
            )
        for sample in self.rejected_samples:
            logger.info(
                "%s_REJECT model=%s year=%s reason=%s text=%r",
                AUDIT_TAG,
                self.model,
                self.target_year,
                sample["reason"],
                sample["text"],
            )
        for i, text in enumerate(self.kept_samples, start=1):
            logger.info(
                "%s_KEPT model=%s year=%s rank=%d text=%r",
                AUDIT_TAG,
                self.model,
                self.target_year,
                i,
                text,
            )
        if self.returned == 0 and not self.error:
            logger.warning(
                "%s_EMPTY model=%s year=%s parsed=%d rejections=%s — check RAW/REJECT lines",
                AUDIT_TAG,
                self.model,
                self.target_year,
                self.parsed_candidates,
                self.rejection_counts or "{}",
            )


def truncate_preview(text: str, max_len: int = _MAX_PREVIEW) -> str:
    s = (text or "").replace("\r\n", "\n").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
