"""Load the YAML config. All tunable values live in configs/default.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = REPO_ROOT
    return cfg


def apply_overrides(cfg: dict, items) -> dict:
    """Apply `--set a.b.c=value` overrides (value parsed as YAML: numbers, true/false, null, lists)."""
    for item in items or []:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override '{item}' must look like key=value")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg
