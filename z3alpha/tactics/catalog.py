"""Linear MCTS: tactic name ↔ int maps (logic JSON lists names; see :mod:`z3alpha.tactics.logic_config`).

Branched shortlist path ids: :mod:`z3alpha.stage2.utils`. Probes: :class:`z3alpha.stage2.strategy_tree.ProbeAction`.
"""

from __future__ import annotations

from typing import NewType

# --- id → SMT2 tactic name (linear search action space)

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
    6: "bv1-blast",
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

SOLVER_TACTICS = list(SOLVER_CATALOG.values())
PREPROCESS_TACTICS = list(PREPROCESS_CATALOG.values())

SUPPORTED_TACTIC_PARAMS = [
    "inline_vars",
    "seed",
    "factor",
    "elim_and",
    "som",
    "blast_distinct",
    "flat",
    "hi_div0",
    "local_ctx",
    "hoist_mul",
    "push_ite_bv",
    "pull_cheap_ite",
    "nla2bv_max_bv_size",
    "add_bound_lower",
    "add_bound_upper",
    "pb2bv_all_clauses_limit",
    "lia2pb_max_bits",
    "random_seed",
    "push_ite_arith",
    "hoist_ite",
    "arith_lhs",
]

SOLVER_NAME_TO_ID = {n: i for i, n in SOLVER_CATALOG.items()}
PREPROCESS_NAME_TO_ID = {n: i for i, n in PREPROCESS_CATALOG.items()}
NAME_TO_ID = {**SOLVER_NAME_TO_ID, **PREPROCESS_NAME_TO_ID}

CatalogTacticId = NewType("CatalogTacticId", int)

__all__ = [
    "CatalogTacticId",
    "NAME_TO_ID",
    "PREPROCESS_CATALOG",
    "PREPROCESS_NAME_TO_ID",
    "PREPROCESS_TACTICS",
    "SOLVER_CATALOG",
    "SOLVER_NAME_TO_ID",
    "SOLVER_TACTICS",
    "SUPPORTED_TACTIC_PARAMS",
]
