#!/usr/bin/env python3
"""Short generation smoke using the real predict_literature_lm path.

Runs year=2023, k=10 with current compare settings so you can eyeball coherence
before launching the full Job-2 benchmark. Writes eval/gen_smoke_*.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neurolink.db import Database
from neurolink.forecast.predict.direction_filter import format_compliance
from neurolink.forecast.predict.literature_lora import (
    LiteratureLoraConfig,
    adapter_exists,
    predict_literature_lm,
)
from neurolink.forecast.predict.llm_core import release_gpu_memory
from neurolink.utils.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("gen_smoke")


@dataclass
class GenSmokeConfig:
    db_path: str = "data/neurolink.db"
    test_years: list[int] = field(default_factory=lambda: [2023])
    top_k: int = 10
    models: list[str] = field(
        default_factory=lambda: ["literature_lora", "mistral_base", "braingpt"]
    )
    include_instruct_if_present: bool = True
    output_dir: str = "eval"
    literature: LiteratureLoraConfig = field(default_factory=LiteratureLoraConfig)


def _coherence_notes(preds: list[str]) -> dict:
    fc = format_compliance(preds) if preds else None
    unique = len({p.lower() for p in preds})
    return {
        "n": len(preds),
        "n_unique": unique,
        "frac_unique": (unique / len(preds)) if preds else 0.0,
        "format_frac_valid": fc.frac_valid if fc else None,
        "format_frac_word_len_ok": fc.frac_word_len_ok if fc else None,
        "format_mean_words": fc.mean_words if fc else None,
        "ok_enough": bool(preds) and unique >= max(3, len(preds) // 2) and (fc.frac_valid >= 0.7 if fc else False),
    }


def run_smoke(cfg: GenSmokeConfig) -> Path:
    db = Database(resolve_path(cfg.db_path))
    out_dir = resolve_path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"gen_smoke_{stamp}.json"

    runs: list[dict] = []
    lit = cfg.literature
    if lit.benchmark_lora_year_max is None and cfg.test_years:
        lit = replace(lit, benchmark_lora_year_max=min(cfg.test_years) - 1)

    jobs: list[tuple[str, LiteratureLoraConfig, str]] = []
    for model in cfg.models:
        label = f"{model}@base_lora" if model == "literature_lora" else model
        jobs.append((model, lit, label))

    if cfg.include_instruct_if_present:
        inst = replace(
            lit,
            adapter_dir="data/models/literature_instruct",
            base_model="mistralai/Mistral-7B-Instruct-v0.2",
            prompt_style="instruct",
        )
        if adapter_exists(inst, lit.benchmark_lora_year_max or 2022):
            jobs.append(("literature_lora", inst, "literature_lora@instruct"))
            logger.info("Also smoking Instruct adapter under literature_instruct/")
        else:
            logger.info("No Instruct adapter — skip literature_lora@instruct")

    with db.connect(readonly=True) as conn:
        for model, model_lit, label in jobs:
            for year in cfg.test_years:
                logger.info("=== gen-smoke %s year=%d k=%d ===", label, year, cfg.top_k)
                preds = predict_literature_lm(
                    conn, year, cfg.top_k, model_lit, model=model
                )
                texts = [t for t, _ in preds]
                notes = _coherence_notes(texts)
                runs.append(
                    {
                        "label": label,
                        "model": model,
                        "year": year,
                        "adapter_dir": model_lit.adapter_dir,
                        "prompt_style": model_lit.prompt_style,
                        "predictions": [
                            {"rank": i + 1, "text": t, "score": float(s)}
                            for i, (t, s) in enumerate(preds)
                        ],
                        "coherence": notes,
                    }
                )
                logger.info(
                    "%s year=%d kept=%d unique=%d ok_enough=%s",
                    label,
                    year,
                    notes["n"],
                    notes["n_unique"],
                    notes["ok_enough"],
                )
                for i, (t, s) in enumerate(preds[:10], start=1):
                    logger.info("  %2d. (%.3f) %s", i, s, t)
                release_gpu_memory()

    payload = {
        "created_at": stamp,
        "config": {
            "db_path": cfg.db_path,
            "test_years": cfg.test_years,
            "top_k": cfg.top_k,
            "temperature": lit.llm.temperature,
            "pool_factor": lit.pool_factor,
            "near_duplicate_threshold": lit.near_duplicate_threshold,
        },
        "runs": runs,
        "verdict": {
            "all_ok_enough": all(r["coherence"]["ok_enough"] for r in runs) if runs else False,
            "by_label": {r["label"]: r["coherence"]["ok_enough"] for r in runs},
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    logger.info("verdict=%s", json.dumps(payload["verdict"], indent=2))
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generation smoke before full benchmark")
    p.add_argument("--config", default="config/forecast/gen_smoke.yaml")
    args = p.parse_args(argv)
    cfg = load_config(args.config, GenSmokeConfig)
    path = run_smoke(cfg)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
