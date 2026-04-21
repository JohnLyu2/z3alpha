import json
from pathlib import Path

SOLVER_CATALOG = {
    10: "smt",
    11: "qfnra-nlsat",
    12: "sat",
    13: "qfbv",
    14: "qfnia",
    15: "qfnra",
    16: "qflia",
    17: "qflra",
}

PREPROCESS_CATALOG = {
    5: "nla2bv",
    7: "bit-blast",
    8: "pb2bv",
    20: "simplify",
    21: "propagate-values",
    22: "ctx-simplify",
    23: "elim-uncnstr",
    24: "solve-eqs",
    25: "purify-arith",
    26: "max-bv-sharing",
    27: "aig",
    28: "reduce-bv-size",
    29: "ackermannize_bv",
    32: "lia2card",
    33: "card2bv",
    34: "cofactor-term-ite",
    35: "propagate-ineqs",
    36: "add-bounds",
    37: "normalize-bounds",
    38: "lia2pb",
}

SOLVER_NAME_TO_ID = {name: id for id, name in SOLVER_CATALOG.items()}
PREPROCESS_NAME_TO_ID = {name: id for id, name in PREPROCESS_CATALOG.items()}
NAME_TO_ID = {**SOLVER_NAME_TO_ID, **PREPROCESS_NAME_TO_ID}

_BUILTIN_CONFIG_DIR = Path(__file__).parent / "logic_configs"


def _parse_logic_config(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    solver_ids = [SOLVER_NAME_TO_ID[n] for n in raw["solver_tactics"]]
    preprocess_ids = [PREPROCESS_NAME_TO_ID[n] for n in raw["preprocess_tactics"]]

    params = {}
    for tac_name, param_grid in raw.get("params", {}).items():
        tac_id = NAME_TO_ID[tac_name]
        params[tac_id] = param_grid

    return {
        "solver_tactics": solver_ids,
        "preprocess_tactics": preprocess_ids,
        "params": params,
    }


def load_logic_config(logic, config_dir=None):
    if config_dir:
        override = Path(config_dir) / f"{logic}.json"
        if override.exists():
            return _parse_logic_config(override)

    builtin = _BUILTIN_CONFIG_DIR / f"{logic}.json"
    if not builtin.exists():
        raise FileNotFoundError(
            f"No logic config found for '{logic}' "
            f"(looked in {_BUILTIN_CONFIG_DIR})"
        )
    return _parse_logic_config(builtin)


def create_params_dict(logic, config_dir=None):
    return load_logic_config(logic, config_dir)["params"]
