"""Hugging Face token detection for local Mistral / transformers downloads."""

from __future__ import annotations

import os
from pathlib import Path


def hf_token_from_env() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return None


def hf_token_from_cache() -> str | None:
    for path in (
        Path.home() / ".cache" / "huggingface" / "token",
        Path(os.environ.get("HF_HOME", "")) / "token" if os.environ.get("HF_HOME") else None,
    ):
        if path and path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def hf_token_available() -> str | None:
    """Return token if found in env or Hugging Face cache."""
    return hf_token_from_env() or hf_token_from_cache()


def set_hf_token(token: str, *, validate: bool = False) -> None:
    token = token.strip()
    if not token:
        return
    os.environ["HF_TOKEN"] = token
    if not validate:
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except ImportError:
        pass
