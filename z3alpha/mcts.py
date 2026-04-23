import copy
import csv
import logging
import math
from pathlib import Path
from typing import Any

from z3alpha.environment import LinearStrategyGame
from z3alpha.logging_config import attach_file_logger

logger = logging.getLogger(__name__)

INIT_Q = 0
IS_MEAN_EST = False


class MCTSNode:
    def __init__(self, logic_config, is_mean, trace_log, c_ucb, action_history=None):
        self.param_dict: dict[int, dict[str, list[Any]]] = (
            logic_config["params"] if logic_config else {}
        )
        self.is_mean: bool = is_mean
        self.c_ucb: float | None = c_ucb
        self.visit_count: int = 0
        self.action_history: list[Any] = (
            [] if action_history is None else list(action_history)
        )
        self.value_est: float = 0
        self.children: dict[Any, "MCTSNode"] = {}
        self.reward: float = 0
        self._set_param_mabs()
        self.trace_log = trace_log

    def __str__(self):
        return str(self.action_history)

    def is_expanded(self):
        return bool(self.children)

    def has_param_mabs(self):
        if len(self.action_history) == 0:
            return False
        latest_action = self.action_history[-1]
        return latest_action in self.param_dict.keys()

    def _set_param_mabs(self):
        if not self.has_param_mabs():
            return
        latest_action = self.action_history[-1]
        self.params = self.param_dict[latest_action]
        self.MABs = {}
        self.selected = {}
        for param in self.params.keys():
            mab_dict = {}
            for param_value in self.params[param]:
                mab_dict[param_value] = [0, INIT_Q]  # (visit count, q estimation)
            self.MABs[param] = mab_dict
            self.selected[param] = None

    def _ucb(self, action_pair, action):
        visit_count, q_score = action_pair
        explore_score = self.c_ucb * math.sqrt(
            math.log(self.visit_count + 1) / (visit_count + 0.001)
        )
        ucb = q_score + explore_score
        self.trace_log.debug(
            f"  Value of {action}: Q value: {q_score:.05f}; Exp: {explore_score:.05f} ({visit_count}/{self.visit_count}); UCB: {ucb:.05f}"
        )
        return ucb

    def _select_mab(self, param):
        MABdict = self.MABs[param]
        selected = None
        best_ucb = -1
        for value_candidate, pair in MABdict.items():
            ucb = self._ucb(pair, value_candidate)
            if ucb > best_ucb:
                best_ucb = ucb
                selected = value_candidate
        assert best_ucb >= 0
        return selected

    def select_mabs(self):
        for param in self.params.keys():
            self.trace_log.debug(f"\n  Select MAB of {param}")
            selected_value = self._select_mab(param)
            self.trace_log.debug(f"  Selected value: {selected_value}\n")
            self.selected[param] = selected_value
        return self.selected

    def backup_mabs(self, reward):
        for param in self.params.keys():
            mab_dict = self.MABs[param]
            selected_value = self.selected[param]

            if self.is_mean:
                mab_dict[selected_value][1] = (
                    mab_dict[selected_value][1] * mab_dict[selected_value][0] + reward
                ) / (mab_dict[selected_value][0] + 1)
            else:
                mab_dict[selected_value][1] = max(mab_dict[selected_value][1], reward)
            mab_dict[selected_value][0] += 1
            self.selected[param] = None


class BaseMCTSRun:
    def __init__(
        self,
        config,
        bench_lst,
        logic,
        z3path,
        value_type,
        log_folder,
        batch_size=1,
        root=None,
        logic_config=None,
    ):
        self.z3path = z3path
        self.config = config
        self.logic_config = logic_config
        self.num_simulations = config["sim_num"]
        self.is_mean = IS_MEAN_EST
        self.discount = 1
        self.c_uct = config["c_uct"]
        self.training_list = bench_lst
        self.logic = logic
        self.timeout = config["timeout"]
        self.value_type = value_type
        self.batch_size = batch_size
        self._init_stage_state(log_folder, bench_lst)

        self.trace_log = attach_file_logger(
            f"z3alpha.s{self.stage}mcts",
            Path(log_folder) / f"stage{self.stage}_mcts_trace.log",
        )

        if not root:
            root = MCTSNode(self.logic_config, self.is_mean, self.trace_log, self.c_ucb)
        self.root = root
        self.best_reward = -1
        self.top_strategies = [None, None, None]  # top 3 strategies
        self.top_rewards = [-1, -1, -1]

    def _init_stage_state(self, log_folder, bench_lst):
        raise NotImplementedError

    def _create_env(self):
        raise NotImplementedError

    def _after_simulation(self):
        pass

    def _uct(self, child_node, parent_node, action):
        value_score = child_node.reward + self.discount * child_node.value_est
        explore_score = self.c_uct * math.sqrt(
            math.log(parent_node.visit_count) / (child_node.visit_count + 0.001)
        )
        uct = value_score + explore_score
        self.trace_log.debug(
            f"  Value of {action}: Q value: {value_score:.05f}; Exp: {explore_score:.05f} ({child_node.visit_count}/{parent_node.visit_count}); UCT: {uct:.05f}"
        )
        return uct

    def _select(self):
        search_path = [self.root]
        node = self.root
        while node.is_expanded() and not self.env.is_terminal():
            self.trace_log.debug(f"\n  Select at {node}")

            selected = None
            best_uct = -1
            next_node = None
            for action, child_node in node.children.items():
                uct = self._uct(child_node, node, action)
                if uct > best_uct:
                    selected = action
                    best_uct = uct
                    next_node = child_node
            assert best_uct >= 0
            node = next_node
            self.trace_log.debug(f"  Selected action {selected}")
            params = node.select_mabs() if node.has_param_mabs() else None
            search_path.append(node)
            self.env.step(selected, params)
        return node, search_path

    def _expand_node(self, node, actions, reward):
        node.reward = reward
        for action in actions:
            history = copy.deepcopy(node.action_history)
            history.append(action)
            node.children[action] = MCTSNode(
                self.logic_config, self.is_mean, self.trace_log, self.c_ucb, history
            )

    def _rollout(self):
        self.env.rollout()

    def _backup(self, search_path, sim_value):
        value = sim_value
        for node in reversed(search_path):
            if self.is_mean:
                node.value_est = (node.value_est * node.visit_count + value) / (
                    node.visit_count + 1
                )
            else:
                node.value_est = max(node.value_est, value)
            node.visit_count += 1
            value = node.reward + self.discount * value
            if node.has_param_mabs():
                node.backup_mabs(value)

    def _one_simulation(self):
        self.env = self._create_env()
        selected_node, search_path = self._select()
        self.trace_log.info("Selected Node: " + str(selected_node))
        self.trace_log.info("Selected Strategy ParseTree: " + str(self.env))
        if self.env.is_terminal():
            self.trace_log.info("Terminal Strategy: no rollout")
            value = self.env.get_value(self.res_database, self.value_type)
        else:
            actions = self.env.legal_actions()
            self._expand_node(selected_node, actions, 0)
            self._rollout()
            self.trace_log.info(f"Rollout Strategy: {self.env}")
            value = self.env.get_value(self.res_database, self.value_type)
        self._update_top_strategies(value, str(self.env))
        self.trace_log.info(f"Final Return: {value}\n")
        self._backup(search_path, value)

    def _update_top_strategies(self, value, strategy):
        for i in range(3):
            if value > self.top_rewards[i]:
                if i == 0:
                    msg = f"At sim {self.num_sim}, new best reward found: {value:.5f}"
                    logger.info(msg)
                    self.trace_log.info(msg)
                self.top_rewards.insert(i, value)
                self.top_rewards.pop()
                self.top_strategies.insert(i, strategy)
                self.top_strategies.pop()
                break

    def start(self):
        for i in range(self.num_simulations):
            self.num_sim = i
            if self.stage == 1:
                logger.info(f"Simulation {i} starts")
            self.trace_log.info(f"Simulation {i} starts")
            self._one_simulation()
            self._after_simulation()

    def get_strategy_stat(self, strat):
        return self.res_database[strat]

    def get_res_dict(self):
        return self.res_database

    def get_best_strat(self):
        return self.top_strategies[0]

    def get_best_3_strats(self):
        return self.top_strategies


class LinearStrategySearchRun(BaseMCTSRun):
    stage = 1

    def _write_new_results(self):
        """Append any newly evaluated strategies to linear results CSV."""
        new_strats = set(self.res_database.keys()) - self._written_strats
        if not new_strats:
            return
        with open(self._s1_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for strat in new_strats:
                res = self.res_database[strat]
                row = [strat] + [r[1] if r[0] else -r[1] for r in res]
                writer.writerow(row)
        self._written_strats.update(new_strats)

    def _init_stage_state(self, log_folder, bench_lst):
        self.c_ucb = self.config["c_ucb"]
        self.res_database = {}
        self._s1_csv_path = Path(log_folder) / "linear_strategy_results.csv"
        self._written_strats = set()
        with open(self._s1_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["strat"] + bench_lst)

    def _create_env(self):
        return LinearStrategyGame(
            self.stage,
            self.training_list,
            self.logic,
            self.timeout,
            self.config,
            self.batch_size,
            z3path=self.z3path,
            logic_config=self.logic_config,
        )

    def _after_simulation(self):
        self._write_new_results()

