"""Config loading, paths, run IDs, and temporal helpers."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union, cast

import yaml
from dacite import from_dict

T = TypeVar("T")

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def load_config(yaml_path_or_dict: Union[str, Path, Dict, T], config_class: Type[T]) -> T:
    if isinstance(yaml_path_or_dict, config_class):
        return yaml_path_or_dict
    if isinstance(yaml_path_or_dict, (str, Path)):
        path = resolve_path(yaml_path_or_dict)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        data = yaml_path_or_dict
    data = expand_env_vars(data)
    return from_dict(config_class, cast(Dict, data))


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def make_run_id(stage: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stage}_{ts}"


def available_question_years(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        "SELECT DISTINCT year FROM questions WHERE year IS NOT NULL ORDER BY year"
    ).fetchall()
    return [int(r[0]) for r in rows]


def infer_test_years(conn: sqlite3.Connection, configured: list[int] | None = None) -> list[int]:
    years = available_question_years(conn)
    if not years:
        return configured or []
    if configured:
        valid = [y for y in configured if y in years]
        if valid:
            return valid
    if len(years) >= 2:
        return years[1:]
    return [years[-1]]
