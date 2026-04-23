from pathlib import Path
from typing import Callable

from z3alpha.parser import parse_linear_strategy
from z3alpha.tactics.catalog import PREPROCESS_TACTICS, SOLVER_TACTICS
from z3alpha.utils import par_n_reward, solved_num_reward

ActionId = int
ActionPath = list[ActionId]


def encode_linear_strategies(linear_strategies: list[str]):
    tactic_dict = {}
    solver_id = 1000
    act2solver = {}
    preprocess_id = 2000
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
            act_lst.append(tactic_dict[tac_str])
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
    cur_act_path: ActionPath, strat_act_lst: list[ActionPath]
) -> list[ActionId]:
    action_set = set()
    for strategy in strat_act_lst:
        if is_strict_prefix(cur_act_path, strategy):
            action_set.add(strategy[len(cur_act_path)])
    return list(action_set)


def create_benchmark_list(benchmark_directories: list[str]) -> list[str]:
    benchmark_lst = []
    for bench_dir in benchmark_directories:
        assert Path(bench_dir).exists()
        benchmark_lst += [str(p) for p in sorted(list(Path(bench_dir).rglob("*.smt2")))]
    benchmark_lst.sort()
    return benchmark_lst


def reward_dispatcher(timeout: int) -> dict[str, Callable[[list], float]]:
    return {
        "#solved": solved_num_reward,
        "par2": lambda results: par_n_reward(results, 2, timeout),
        "par10": lambda results: par_n_reward(results, 10, timeout),
    }
