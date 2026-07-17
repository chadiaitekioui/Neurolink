#!/usr/bin/env python3
"""Verify HuggingFace models are cached for offline GPU jobs on Jean Zay."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REQUIRED_MODELS = [
    "mistralai/Mistral-7B-v0.1",
    "ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section",
    "BrainGPT/BrainGPT-7B-v0.2",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def _hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))


def is_model_cached(model_id: str, hf_home: Path | None = None) -> bool:
    """True when at least one snapshot exists in the HF hub cache."""
    root = hf_home or _hf_home()
    slug = "models--" + model_id.replace("/", "--")
    markers = ("config.json", "adapter_config.json")
    for cache_root in (root / "hub" / slug, root / slug):
        snapshots = cache_root / "snapshots"
        if not snapshots.is_dir():
            continue
        for snap in snapshots.iterdir():
            if snap.is_dir() and any((snap / name).is_file() for name in markers):
                return True
    return False


def check_4bit_cuda() -> tuple[bool, str]:
    """On GPU nodes, confirm bitsandbytes can initialize 4-bit loading."""
    if not os.environ.get("SLURM_JOB_ID"):
        return True, "skipped (login node)"
    try:
        import torch
    except ImportError:
        return False, "torch not importable"
    if not torch.cuda.is_available():
        return False, "CUDA unavailable on GPU job"
    try:
        from transformers import BitsAndBytesConfig

        BitsAndBytesConfig(load_in_4bit=True)
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        return False, f"bitsandbytes/4-bit init failed: {exc}"
    return True, "ok"


def main() -> int:
    hf_home = _hf_home()
    print(f"HF_HOME: {hf_home}\n")

    missing = [model for model in REQUIRED_MODELS if not is_model_cached(model, hf_home)]
    if missing:
        print("=== Missing models (required offline) ===")
        for model in missing:
            print(f"  !! {model}")
        print("\nRe-run: bash scripts/cluster/setup_login.sh (login node, HTTP)")
        return 1

    print("=== Cached models ===")
    for model in REQUIRED_MODELS:
        print(f"  OK {model}")

    ok, msg = check_4bit_cuda()
    print(f"\n=== 4-bit CUDA ===")
    if ok:
        print(f"  OK ({msg})")
    else:
        print(f"  !! {msg}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
