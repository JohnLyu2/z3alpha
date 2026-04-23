import random
from typing import Callable

from z3alpha.evaluator import SolverEvaluator
from z3alpha.mcts import BaseMCTSRun
from z3alpha.stage2.strategy_tree import Stage2Context
from z3alpha.stage2.utils import reward_dispatcher
from z3alpha.strategy_tree import StrategyTree


def solve_with_cache(
    bench_id: int,
    linear_strategies: list[tuple[list[int], int]],
    cache_database: dict[tuple[int, ...], list[tuple[bool, float, str]]],
    timeout: int,
) -> tuple[bool, float]:
    time_remain = timeout
    solved = False
    time_used = 0
    for strategy, strategy_timeout in linear_strategies:
        cache_solved, cache_time, _ = cache_database[tuple(strategy)][bench_id]
        if strategy_timeout < cache_time:
            time_remain -= strategy_timeout
            time_used += strategy_timeout
        elif time_remain < cache_time:
            time_used += time_remain
            time_remain = 0
        else:
            if cache_solved:
                solved = True
                time_used += cache_time
                break
            time_remain -= cache_time
            time_used += cache_time
        if time_used >= timeout:
            break
    return solved, time_used


def evaluate_stage2_with_cache(
    benchmark_count: int,
    get_linear_strategies_for_bench,
    cache_database: dict[tuple[int, ...], list[tuple[bool, float, str]]],
    timeout: int,
) -> list[tuple[bool, float]]:
    results = []
    for bench_id in range(benchmark_count):
        linear_strategies = get_linear_strategies_for_bench(bench_id)
        results.append(
            solve_with_cache(bench_id, linear_strategies, cache_database, timeout)
        )
    return results


class Stage2StrategyGame:
    def __init__(
        self,
        training_lst,
        logic,
        timeout,
        sconfig,
        batch_size,
        z3path,
    ):
        self.stage = 2
        self.benchmarks = training_lst
        self.strat_ast = StrategyTree(2, logic, timeout, sconfig)
        self.stage2_context: Stage2Context = sconfig["stage2_context"]
        self.probe_records = self.stage2_context.probe_records
        self.simulator = SolverEvaluator(
            z3path,
            training_lst,
            timeout,
            batch_size,
        )
        self.timeout = timeout

    def __str__(self) -> str:
        return str(self.strat_ast)

    def is_terminal(self):
        return self.strat_ast.is_terminal()

    def legal_actions(self, rollout=False):
        return self.strat_ast.legal_actions(rollout)

    def step(self, action, params):
        self.strat_ast.apply_rule(action, params)

    def rollout(self):
        assert not self.is_terminal()
        while not self.is_terminal():
            actions = self.legal_actions(rollout=True)
            action = random.choice(actions)
            self.step(action, None)

    def _get_linear_strategies(self, bench_id):
        probe_record = self.probe_records[bench_id]
        return self.strat_ast.get_linear_strategies(probe_record)

    def get_s2_res_list(self, database):
        return evaluate_stage2_with_cache(
            len(self.benchmarks),
            self._get_linear_strategies,
            database,
            self.timeout,
        )

    def get_value(self, database: dict, reward_type: str) -> float:
        assert self.is_terminal()
        res_lst = self.get_s2_res_list(database)
        reward_fn_by_type: dict[str, Callable[[list], float]] = reward_dispatcher(
            self.timeout
        )
        if reward_type not in reward_fn_by_type:
            raise Exception(f"Unknown value type {reward_type}")
        return reward_fn_by_type[reward_type](res_lst)


class Stage2MCTSRun(BaseMCTSRun):
    stage = 2

    def _init_stage_state(self, log_folder, bench_lst):
        self.c_ucb = None
        self.res_database = self.config["stage2_context"].result_cache

    def _create_env(self):
        return Stage2StrategyGame(
            self.training_list,
            self.logic,
            self.timeout,
            self.config,
            self.batch_size,
            z3path=self.z3path,
        )
