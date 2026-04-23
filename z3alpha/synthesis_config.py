"""Load and merge synthesis configuration: package defaults, JSON file, CLI overrides."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

_DEFAULTS_PATH = Path(__file__).resolve().with_name("synthesis_defaults.json")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_synthesis_defaults() -> dict[str, Any]:
    with open(_DEFAULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def merge_synthesis_config(user_json: dict[str, Any]) -> dict[str, Any]:
    """Defaults first, then keys from the experiment JSON (user overrides)."""
    return _deep_merge(load_synthesis_defaults(), user_json)
