"""Shared MCTS engine: ``MctsConfig``, ``BaseMCTSRun`` (PUCT selection)."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from z3alpha.config.logging import attach_file_logger
from z3alpha.mcts.llm_prior import LLMPriorConfig
from z3alpha.mcts.node import MCTSNode
from z3alpha.tactics.catalog import (
    PREPROCESS_CATALOG,
    SOLVER_CATALOG,
    tactic_name_for_action,
)

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

    ``llm_prior`` is optional; when set and enabled, stage-1 MCTS uses LLM
    scores as PUCT priors (see :class:`z3alpha.mcts.llm_prior.LLMPriorConfig`).
    """

    sim_num: int
    timeout: int
    c_uct: float
    random_seed: int
    is_mean: bool = DEFAULT_IS_MEAN
    llm_prior: LLMPriorConfig | None = None


class BaseMCTSRun:
    """Abstract MCTS run.

    Subclasses implement :meth:`_init_stage_state` (any stage-specific state),
    :meth:`_create_env` (build the per-simulation game env), and may override
    :meth:`params_for` to inject a tactic-parameter heuristic.
    """

    stage: int
    #: If set, trace file name under ``log_folder`` instead of ``stage{N}_mcts_trace.log``.
    TRACE_LOG_FILENAME: ClassVar[str | None] = None

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

        trace_path = (
            self.log_folder / self.TRACE_LOG_FILENAME
            if self.TRACE_LOG_FILENAME
            else self.log_folder / f"stage{self.stage}_mcts_trace.log"
        )
        self.trace_log = attach_file_logger(f"z3alpha.s{self.stage}mcts", trace_path)

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

    def _priors_for(self, actions: list) -> dict[Any, float] | None:
        """Return mapping action -> prior P for expansion; ``None`` means uniform (1.0 per child)."""
        return None

    def _seed_expanded_children(self, node: MCTSNode, value: float) -> None:
        """Optional post-rollout seeding for freshly expanded children."""
        return

    def _trace_format_action(self, action: Any) -> str:
        """Pretty-print ``action`` for MCTS trace logs (linear SMT tactic ids vs stage-2 values)."""
        if isinstance(action, int):
            return f"{tactic_name_for_action(action)} ({action})"
        return str(action)

    def _trace_format_node(self, node: MCTSNode) -> str:
        inner = ", ".join(self._trace_format_action(a) for a in node.action_history)
        return f"[{inner}]"

    def _candidate_row_bucket(self, action: Any) -> int:
        """Order key for grouping: 0 solver, 1 preprocessing, 2 other."""
        if isinstance(action, int):
            if action in SOLVER_CATALOG:
                return 0
            if action in PREPROCESS_CATALOG:
                return 1
        return 2

    def _trace_group_candidate_rows(
        self, rows: list[tuple[Any, MCTSNode, float, float, float, int, float, int]]
    ) -> list[tuple[str, list[tuple[Any, MCTSNode, float, float, float, int, float, int]]]]:
        bucket_label = ["Solver tactics", "Preprocessing tactics", "Other actions"]
        buckets: dict[int, list] = {0: [], 1: [], 2: []}
        label_key = lambda r: self._trace_format_action(r[0])
        for row in rows:
            buckets[self._candidate_row_bucket(row[0])].append(row)
        out: list[tuple[str, list]] = []
        for i in range(3):
            grp = buckets[i]
            if not grp:
                continue
            grp.sort(key=label_key)
            out.append((bucket_label[i], grp))
        return out

    def _trace_log_selection_candidates(
        self,
        node: MCTSNode,
        rows: list[tuple[Any, MCTSNode, float, float, float, int, float, int]],
    ) -> None:
        """Emit one grouped, aligned DEBUG block instead of per-action lines."""
        label_w = 42
        groups = self._trace_group_candidate_rows(rows)
        lines = [f"\nSelect at {self._trace_format_node(node)}"]
        hdr = (
            f"{'action':<{label_w}}"
            f"{'PUCT':>9} {'Exploit':>9} {'Explore':>9} {'N_c/N_p':>9} {'Prior':>9}"
        )
        ruler = "-" * len(hdr.expandtabs())
        for title, grp in groups:
            lines.append(f"  {title}:")
            lines.append(f"    {hdr}")
            lines.append(f"    {ruler}")
            for (
                action,
                _cn,
                puct,
                v_est,
                exp,
                p_n,
                prior,
                n_ch,
            ) in grp:
                lab_full = self._trace_format_action(action)
                lab = (
                    lab_full
                    if len(lab_full) <= label_w
                    else lab_full[: label_w - 3] + "..."
                )
                vis = f"{n_ch}/{p_n}"
                lines.append(
                    f"    {lab:<{label_w}}"
                    f"{puct:9.3f} {v_est:9.3f} {exp:9.3f} {vis:>9} {prior:9.3f}"
                )
        self.trace_log.debug("\n".join(lines))

    def _select(self):
        search_path = [self.root]
        node = self.root
        while node.is_expanded() and not self.env.is_terminal():
            rows: list[tuple[Any, MCTSNode, float, float, float, int, float, int]] = []
            for action, child_node in node.children.items():
                value_score = child_node.value_est
                parent_n = max(1, node.visit_count)
                p = child_node.prior
                n_ch = child_node.visit_count
                explore_score = (
                    self.c_uct * p * math.sqrt(parent_n) / (1 + n_ch)
                )
                puct = value_score + explore_score
                rows.append(
                    (action, child_node, puct, value_score, explore_score, parent_n, p, n_ch)
                )

            selected = None
            best_puct = float("-inf")
            next_node: MCTSNode | None = None
            for action, child_node, puct, *_rest in rows:
                if puct > best_puct:
                    selected = action
                    best_puct = puct
                    next_node = child_node
            assert selected is not None
            assert next_node is not None
            assert best_puct > float("-inf")

            self._trace_log_selection_candidates(node, rows)

            node = next_node
            self.trace_log.debug(f"  Selected action {self._trace_format_action(selected)}")
            search_path.append(node)
            self.env.step(selected)
        return node, search_path

    def _expand_node(
        self, node: MCTSNode, actions, priors: dict[Any, float] | None = None
    ) -> None:
        for action in actions:
            history = copy.deepcopy(node.action_history)
            history.append(action)
            child = MCTSNode(history)
            if priors is not None and action in priors:
                child.prior = float(priors[action])
            node.children[action] = child

    def _rollout(self) -> None:
        self.env.rollout()

    def _backup(self, search_path, sim_value: float) -> None:
        value = sim_value
        for node in reversed(search_path):
            # If this is the first actual visit, overwrite any optimistic/seeded
            # initialization so later estimation is based on real evaluations.
            if node.visit_count == 0:
                node.value_est = value
            elif self.is_mean:
                node.value_est = (node.value_est * node.visit_count + value) / (
                    node.visit_count + 1
                )
            else:
                node.value_est = max(node.value_est, value)
            node.visit_count += 1

    def _one_simulation(self) -> None:
        self.env = self._create_env()
        selected_node, search_path = self._select()
        self.trace_log.info(f"Selected Node: {self._trace_format_node(selected_node)}")
        self.trace_log.info("Selected Strategy ParseTree: " + str(self.env))
        if self.env.is_terminal():
            self.trace_log.info("Terminal Strategy: no rollout")
            value = self.env.get_value(self.res_database, self.value_type)
        else:
            actions = self.env.legal_actions()
            priors = self._priors_for(actions)
            self._expand_node(selected_node, actions, priors)
            self._rollout()
            self.trace_log.info(f"Rollout Strategy: {self.env}")
            value = self.env.get_value(self.res_database, self.value_type)
            self._seed_expanded_children(selected_node, value)
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
