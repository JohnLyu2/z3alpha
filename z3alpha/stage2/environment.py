import random
from typing import Callable

from z3alpha.evaluator import SolverEvaluator
from z3alpha.stage2.context import Stage2Context
from z3alpha.stage2.evaluator import evaluate_stage2_with_cache
from z3alpha.strat_tree import StrategyAST
from z3alpha.utils import parNReward, solvedNumReward


class Stage2StrategyGame:
    def __init__(
        self, stage, training_lst, logic, timeout, sconfig, batch_size, z3path,
        logic_config=None,
    ):
        assert stage == 2
        self.stage = 2
        self.benchmarks = training_lst
        self.strat_ast = StrategyAST(2, logic, timeout, sconfig)
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
            params = None
            self.step(action, params)

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

    # return a total reward of [0,1] according to the reward type
    def get_value(self, database: dict, reward_type: str) -> float:
        assert self.is_terminal()
        res_lst = self.get_s2_res_list(database)
        reward_dispatcher: dict[str, Callable[[list], float]] = {
            "#solved": solvedNumReward,
            "par2": lambda results: parNReward(results, 2, self.timeout),
            "par10": lambda results: parNReward(results, 10, self.timeout),
        }
        if reward_type not in reward_dispatcher:
            raise Exception(f"Unknown value type {reward_type}")
        return reward_dispatcher[reward_type](res_lst)
