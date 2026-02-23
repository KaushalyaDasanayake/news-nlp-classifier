# Load YAML + override via NEWSCLF_CONFIG
# switching configs (dev/prod/experiments) without code changes

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "configs/base.yaml"


class ConfigError(RuntimeError):
    pass


def _repo_root_from_file() -> Path:
    """
    We assume this file is at: src/newsclf/utils/config.py
    Repo root is 4 parents up:
      config.py -> utils -> newsclf -> src -> <repo_root>
    """
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return p.parents[4]


def load_config(path: str | None = None) -> dict[str, Any]:
    """
    Load YAML config with this priority:
      1) explicit argument 'path'
      2) env var NEWSCLF_CONFIG
      3) DEFAULT_CONFIG_PATH

    Returns a plain dict.
    """
    repo_root = _repo_root_from_file()

    cfg_path = path or os.getenv("NEWSCLF_CONFIG") or DEFAULT_CONFIG_PATH
    cfg_file = (repo_root / cfg_path).resolve()

    if not cfg_file.exists():
        raise ConfigError(f"Config file not found: {cfg_file}")

    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ConfigError("Config must be a YAML mapping (top-level dict).")

    cfg["_meta"] = {
        "config_path": str(cfg_file),
        "repo_root": str(repo_root),
    }
    return cfg