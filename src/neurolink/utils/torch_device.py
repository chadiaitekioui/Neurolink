"""Torch device selection (CUDA auto-detect, explicit CPU/CUDA)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEVICE_AUTO = "auto"
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def cuda_device_name() -> str | None:
    if not cuda_available():
        return None
    import torch

    return torch.cuda.get_device_name(0)


def resolve_torch_device(preference: str = DEVICE_CPU) -> str:
    """Map config preference to a concrete torch device string."""
    pref = (preference or DEVICE_CPU).lower()
    if pref == DEVICE_AUTO:
        if cuda_available():
            device = DEVICE_CUDA
            name = cuda_device_name()
            logger.info("CUDA GPU detected%s — using cuda", f" ({name})" if name else "")
        else:
            device = DEVICE_CPU
            logger.info("No CUDA GPU detected — using cpu")
        return device
    if pref == DEVICE_CUDA:
        if cuda_available():
            name = cuda_device_name()
            logger.info("Using CUDA GPU%s", f" ({name})" if name else "")
            return DEVICE_CUDA
        logger.warning("CUDA requested but unavailable — falling back to cpu")
        return DEVICE_CPU
    return DEVICE_CPU
