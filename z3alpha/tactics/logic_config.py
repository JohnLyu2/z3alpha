"""Per–SMT-logic tactic lists under ``tactics/logic_configs/``; names map via :mod:`z3alpha.tactics.catalog`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from z3alpha.tactics.catalog import PREPROCESS_NAME_TO_ID, SOLVER_NAME_TO_ID

_LOGIC_CONFIGS_DIR = Path(__file__).resolve().parent / "logic_configs"


def _parse(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    solvers = [SOLVER_NAME_TO_ID[n] for n in raw["solver_tactics"]]
    pre = [PREPROCESS_NAME_TO_ID[n] for n in raw["preprocess_tactics"]]
    return {
        "solver_tactics": solvers,
        "preprocess_tactics": pre,
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
