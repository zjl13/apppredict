from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def _merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}

    defaults = config.pop("defaults", [])
    merged: dict = {}

    for default_name in defaults:
        default_path = config_path.parent / f"{default_name}.yaml"
        with default_path.open("r", encoding="utf-8") as fp:
            default_config = yaml.safe_load(fp) or {}
        merged = _merge_dicts(merged, default_config)

    return _merge_dicts(merged, config)
