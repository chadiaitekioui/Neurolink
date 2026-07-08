#!/usr/bin/env python3
"""Verify Neurolink imports on the cluster (after module load)."""

from __future__ import annotations

import importlib
import sys

CHECKS: list[tuple[str, str, bool]] = [
    ("yaml", "pyyaml", True),
    ("dacite", "dacite", True),
    ("numpy", "numpy (module IDRIS)", True),
    ("scipy", "scipy (module IDRIS)", True),
    ("sklearn", "scikit-learn (module IDRIS)", True),
    ("requests", "requests", True),
    ("rich", "rich (optional)", False),
    ("torch", "torch (module IDRIS)", True),
    ("transformers", "transformers (module IDRIS)", True),
    ("peft", "peft", True),
    ("bitsandbytes", "bitsandbytes (4-bit LoRA)", True),
]

FORBIDDEN_PIP = frozenset({"torch", "numpy", "scipy", "sklearn", "transformers"})


def main() -> int:
    print(f"Python: {sys.executable}\n")
    missing_required: list[str] = []
    missing_optional: list[str] = []
    ok: list[str] = []

    for mod, label, required in CHECKS:
        try:
            importlib.import_module(mod)
            ok.append(label)
        except ImportError:
            (missing_required if required else missing_optional).append(label)

    print("=== OK ===")
    for name in ok:
        print(f"  {name}")

    if missing_required:
        print("\n=== Missing (required) ===")
        for name in missing_required:
            print(f"  !! {name}")

    if missing_optional:
        print("\n=== Missing (optional) ===")
        for name in missing_optional:
            print(f"  -- {name}")

    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
