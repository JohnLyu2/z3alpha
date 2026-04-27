"""Branched MCTS helpers: shortlist path encoding and prefix queries.

**Path ids:** bases ``1000`` / ``2000`` and :class:`BranchedPathSegment` (typing-only) must stay disjoint
from linear :mod:`z3alpha.tactics.catalog` and from :class:`~z3alpha.stage2.strategy_tree.ProbeAction` (50–52).
"""

from __future__ import annotations

from typing import NewType

from z3alpha.parser import parse_linear_strategy
from z3alpha.tactics.catalog import PREPROCESS_TACTICS, SOLVER_TACTICS

SOLVER_INSTANCE_ID_BASE = 1000
PREPROCESS_INSTANCE_ID_BASE = 2000

BranchedPathSegment = NewType("BranchedPathSegment", int)

ActionId = int
ActionPath = list[BranchedPathSegment]


def encode_linear_strategies(
    linear_strategies: list[str],
) -> tuple[
    list[list[BranchedPathSegment]],
    dict[int, tuple],
    dict[int, tuple],
    dict[str, tuple[BranchedPathSegment, ...]],
]:
    tactic_dict = {}
    solver_id = SOLVER_INSTANCE_ID_BASE
    act2solver = {}
    preprocess_id = PREPROCESS_INSTANCE_ID_BASE
    act2preprocess = {}
    strat_act_lst = []
    linear_strategy_to_actions = {}

    for linear_strategy in linear_strategies:
        tac_lst = parse_linear_strategy(linear_strategy)
        act_lst = []
        for tac in tac_lst:
            # parameter order may change after parsing; string key normalizes identity.
            tac_str = str(tac)
            if tac_str not in tactic_dict:
                if tac[0] in SOLVER_TACTICS:
                    tactic_dict[tac_str] = solver_id
                    act2solver[solver_id] = tac
                    solver_id += 1
                elif tac[0] in PREPROCESS_TACTICS:
                    tactic_dict[tac_str] = preprocess_id
                    act2preprocess[preprocess_id] = tac
                    preprocess_id += 1
                else:
                    raise Exception(f"Unknown tactic {tac}")
            act_lst.append(BranchedPathSegment(tactic_dict[tac_str]))
        strat_act_lst.append(act_lst)
        linear_strategy_to_actions[linear_strategy] = tuple(act_lst)

    return strat_act_lst, act2solver, act2preprocess, linear_strategy_to_actions


def is_strict_prefix(lst1: ActionPath, lst2: ActionPath) -> bool:
    if len(lst1) >= len(lst2):
        return False
    for i in range(len(lst1)):
        if lst1[i] != lst2[i]:
            return False
    return True


def next_actions_from_prefix(
    cur_act_path: list[BranchedPathSegment] | list[int],
    strat_act_lst: list[list[BranchedPathSegment]],
) -> list[ActionId]:
    action_set = set()
    for strategy in strat_act_lst:
        if is_strict_prefix(cur_act_path, strategy):
            action_set.add(strategy[len(cur_act_path)])
    return list(action_set)
