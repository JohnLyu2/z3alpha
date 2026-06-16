"""Per–SMT-logic tactic lists under ``tactics/logic_configs/``; names map via :mod:`z3alpha.tactics.catalog`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from z3alpha.tactics.catalog import NAME_TO_ID

_LOGIC_CONFIGS_DIR = Path(__file__).resolve().parent / "logic_configs"


def _parse(path: Path) -> dict[str, Any]:
    """Parse a logic config JSON: ``{tactic_name: {"solver": bool, "params": {...}}}``,
    where each param entry is ``{"default": ..., "values": [...]}``.

    Both ``solver`` (default ``False``) and ``params`` (default ``{}``) are optional.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    solver_tactics: list[int] = []
    preprocess_tactics: list[int] = []
    params: dict[int, dict[str, dict[str, Any]]] = {}
    for name, entry in raw.items():
        tactic_id = NAME_TO_ID[name]
        if entry.get("solver", False):
            solver_tactics.append(tactic_id)
        else:
            preprocess_tactics.append(tactic_id)
        if entry.get("params"):
            params[tactic_id] = entry["params"]
    return {
        "solver_tactics": solver_tactics,
        "preprocess_tactics": preprocess_tactics,
        "params": params,
    }


def load_logic_config(logic: str, config_dir: str | None = None) -> dict[str, Any]:
    if config_dir and (override := Path(config_dir) / f"{logic}.json").exists():
        return _parse(override)
    builtin = _LOGIC_CONFIGS_DIR / f"{logic}.json"
    if not builtin.exists():
        raise FileNotFoundError(
            f"No logic config for {logic!r} (tried {builtin})"
        )
    return _parse(builtin)
