import random
from typing import Callable
from z3alpha.strategy_tree import StrategyTree
from z3alpha.evaluator import SolverEvaluator
from z3alpha.utils import par_n_reward, solved_num_reward


class LinearStrategyGame:
    def __init__(
        self,
        training_lst,
        logic,
        timeout,
        batch_size,
        z3path,
        logic_config=None,
    ):
        self.benchmarks = training_lst
        self.strat_ast = StrategyTree(
            1, logic, timeout,
            logic_config=logic_config,
        )
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
        res_list = self.simulator.get_res_list(strat_str)
        database[strat_str] = res_list
        return res_list

    # return a total reward of [0,1] according to the reward type
    def get_value(self, database: dict, reward_type: str) -> float:
        assert self.is_terminal()
        res_lst = self.get_s1_res_list(database)
        reward_dispatcher: dict[str, Callable[[list], float]] = {
            "#solved": solved_num_reward,
            "par2": lambda results: par_n_reward(results, 2, self.timeout),
            "par10": lambda results: par_n_reward(results, 10, self.timeout),
        }
        if reward_type not in reward_dispatcher:
            raise Exception(f"Unknown value type {reward_type}")
        return reward_dispatcher[reward_type](res_lst)

