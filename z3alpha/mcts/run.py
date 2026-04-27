"""Shared MCTS engine: ``MctsConfig``, ``BaseMCTSRun`` (PUCT selection)."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from z3alpha.config.logging import attach_file_logger
from z3alpha.mcts.node import MCTSNode

logger = logging.getLogger(__name__)


# Default for ``MctsConfig.is_mean``. ``False`` (running max) preserves
# the historical behavior; flipping to ``True`` switches to running mean
# value-estimate updates in :meth:`BaseMCTSRun._backup`.
DEFAULT_IS_MEAN = False


@dataclass(frozen=True)
class MctsConfig:
    """All MCTS knobs for one run.

    ``is_mean`` toggles between max-based and mean-based value-estimate
    updates; default ``False`` (max) keeps the prior behavior. Flip to
    ``True`` to use running-mean estimates instead of running-max.
    """

    sim_num: int
    timeout: int
    c_uct: float
    random_seed: int
    is_mean: bool = DEFAULT_IS_MEAN


# PUCT uniform prior: P=1 for every tactic child.
UNIFORM_PUCT_PRIOR = 1.0


class BaseMCTSRun:
    """Abstract MCTS run.

    Subclasses implement :meth:`_init_stage_state` (any stage-specific state),
    :meth:`_create_env` (build the per-simulation game env), and may override
    :meth:`params_for` to inject a tactic-parameter heuristic.
    """

    stage: int

    def __init__(
        self,
        config: MctsConfig,
        bench_lst,
        logic,
        z3path,
        value_type,
        log_folder,
        batch_size: int = 1,
    ) -> None:
        self.z3path = z3path
        self.config = config
        self.num_simulations = config.sim_num
        self.is_mean = config.is_mean
        self.c_uct = config.c_uct
        self.training_list = bench_lst
        self.logic = logic
        self.timeout = config.timeout
        self.value_type = value_type
        self.batch_size = batch_size
        self.log_folder = Path(log_folder)
        self._init_stage_state()

        self.trace_log = attach_file_logger(
            f"z3alpha.s{self.stage}mcts",
            self.log_folder / f"stage{self.stage}_mcts_trace.log",
        )

        self.root = MCTSNode()
        self.top_strategies: list[Any] = [None, None, None]
        self.top_rewards: list[float] = [-1, -1, -1]

    def _init_stage_state(self) -> None:
        raise NotImplementedError

    def _create_env(self):
        raise NotImplementedError

    def _after_simulation(self) -> None:
        pass

    def _per_simulation_log(self, sim_index: int) -> None:
        """Hook for subclass-specific logging at the start of each simulation."""

    def params_for(self, action):
        """Placeholder param heuristic. Override to set tactic params; default = no using-params."""
        return None

    def _puct(self, child_node: MCTSNode, parent_node: MCTSNode, action) -> float:
        """PUCT: V + c * P * sqrt(N_parent) / (1 + N_child), with uniform prior."""
        value_score = child_node.value_est
        parent_n = max(1, parent_node.visit_count)
        explore_score = (
            self.c_uct
            * UNIFORM_PUCT_PRIOR
            * math.sqrt(parent_n)
            / (1 + child_node.visit_count)
        )
        puct = value_score + explore_score
        self.trace_log.debug(
            f"  Value of {action}: V est: {value_score:.05f}; Exp: {explore_score:.05f} "
            f"({child_node.visit_count}/{parent_n}); P={UNIFORM_PUCT_PRIOR}; PUCT: {puct:.05f}"
        )
        return puct

    def _select(self):
        search_path = [self.root]
        node = self.root
        while node.is_expanded() and not self.env.is_terminal():
            self.trace_log.debug(f"\n  Select at {node}")

            selected = None
            best_puct = float("-inf")
            next_node: MCTSNode | None = None
            for action, child_node in node.children.items():
                puct = self._puct(child_node, node, action)
                if puct > best_puct:
                    selected = action
                    best_puct = puct
                    next_node = child_node
            assert best_puct > float("-inf")
            assert next_node is not None
            node = next_node
            self.trace_log.debug(f"  Selected action {selected}")
            search_path.append(node)
            self.env.step(selected)
        return node, search_path

    def _expand_node(self, node: MCTSNode, actions) -> None:
        for action in actions:
            history = copy.deepcopy(node.action_history)
            history.append(action)
            node.children[action] = MCTSNode(history)

    def _rollout(self) -> None:
        self.env.rollout()

    def _backup(self, search_path, sim_value: float) -> None:
        value = sim_value
        for node in reversed(search_path):
            if self.is_mean:
                node.value_est = (node.value_est * node.visit_count + value) / (
                    node.visit_count + 1
                )
            else:
                node.value_est = max(node.value_est, value)
            node.visit_count += 1

    def _one_simulation(self) -> None:
        self.env = self._create_env()
        selected_node, search_path = self._select()
        self.trace_log.info("Selected Node: " + str(selected_node))
        self.trace_log.info("Selected Strategy ParseTree: " + str(self.env))
        if self.env.is_terminal():
            self.trace_log.info("Terminal Strategy: no rollout")
            value = self.env.get_value(self.res_database, self.value_type)
        else:
            actions = self.env.legal_actions()
            self._expand_node(selected_node, actions)
            self._rollout()
            self.trace_log.info(f"Rollout Strategy: {self.env}")
            value = self.env.get_value(self.res_database, self.value_type)
        self._update_top_strategies(value, str(self.env))
        self.trace_log.info(f"Final Return: {value}\n")
        self._backup(search_path, value)

    def _update_top_strategies(self, value: float, strategy: str) -> None:
        for i in range(3):
            if value > self.top_rewards[i]:
                if i == 0:
                    msg = f"At sim {self.num_sim}, new best value found: {value:.5f}"
                    logger.info(msg)
                    self.trace_log.info(msg)
                self.top_rewards.insert(i, value)
                self.top_rewards.pop()
                self.top_strategies.insert(i, strategy)
                self.top_strategies.pop()
                break

    def start(self) -> None:
        for i in range(self.num_simulations):
            self.num_sim = i
            self._per_simulation_log(i)
            self.trace_log.info(f"Simulation {i} starts")
            self._one_simulation()
            self._after_simulation()

    def get_res_dict(self) -> dict:
        return self.res_database

    def get_best_strat(self):
        return self.top_strategies[0]
