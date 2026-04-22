import random
from typing import Callable
from z3alpha.strat_tree import StrategyAST
from z3alpha.evaluator import SolverEvaluator
from z3alpha.utils import solvedNumReward, parNReward


class StrategyGame:
    def __init__(
        self, stage, training_lst, logic, timeout, sconfig, batch_size, z3path,
        logic_config=None,
    ):
        self.stage = stage
        self.benchmarks = training_lst
        if stage == 1:
            self.strat_ast = StrategyAST(
                1, logic, timeout,
                logic_config=logic_config,
            )
        else:
            self.strat_ast = StrategyAST(2, logic, timeout, sconfig)
            self.probe_records = sconfig["s2dict"]["probe_records"]
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
            params = None
            self.step(action, params)

    # every entry in the resLst is (solved, time, res)
    def get_s1_res_list(self, database):
        strat_str = str(self)
        if strat_str in database:  # does not account for nondeterministism now
            return database[strat_str]
        res_list = self.simulator.getResLst(strat_str)
        database[strat_str] = res_list
        return res_list

    def _get_linear_strategies(self, bench_id):
        probe_record = self.probe_records[bench_id]
        return self.strat_ast.get_linear_strategies(probe_record)

    # this function to be tested
    @staticmethod
    def solve_with_cache(
        bench_id: int,
        ln_strats: list[tuple[list[int], int]],
        database: dict,
        timeout: int,
    ) -> tuple[bool, int]:
        time_remain = timeout
        solved = False
        time_used = 0
        for strat, st_timeout in ln_strats:
            cache_solved, cache_time, _ = database[tuple(strat)][bench_id]
            if st_timeout < cache_time:
                time_remain -= st_timeout
                time_used += st_timeout
            elif time_remain < cache_time:
                # Cache says this run exceeds remaining budget; consume all remaining time.
                time_used += time_remain
                time_remain = 0
            else:
                if cache_solved == True:
                    solved = True
                    time_used += cache_time
                    break
                else:
                    time_remain -= cache_time
                    time_used += cache_time
            if time_used >= timeout:
                break
        return solved, time_used

    def get_s2_res_list(self, database):
        res_lst = []
        for bench_id in range(len(self.benchmarks)):
            # later add bench-specific ln strats
            ln_strats = self._get_linear_strategies(bench_id)
            res_tuple = StrategyGame.solve_with_cache(
                bench_id, ln_strats, database, self.timeout
            )
            res_lst.append(res_tuple)
        return res_lst

    # return a total reward of [0,1] according to the reward type
    def get_value(self, database: dict, reward_type: str) -> float:
        assert self.is_terminal()
        if self.stage == 1:
            res_lst = self.get_s1_res_list(database)
        else:
            res_lst = self.get_s2_res_list(database)
        reward_dispatcher: dict[str, Callable[[list], float]] = {
            "#solved": solvedNumReward,
            "par2": lambda results: parNReward(results, 2, self.timeout),
            "par10": lambda results: parNReward(results, 10, self.timeout),
        }
        if reward_type not in reward_dispatcher:
            raise Exception(f"Unknown value type {reward_type}")
        return reward_dispatcher[reward_type](res_lst)

