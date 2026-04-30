"""Stage-1 (linear) MCTS run: ``LinearStrategySearchRun``."""

from __future__ import annotations

import csv
import logging
from dataclasses import replace
from typing import Any

from z3alpha.environment import LinearStrategyGame
from z3alpha.mcts.llm_prior import LLMPriorScorer
from z3alpha.mcts.run import BaseMCTSRun, MctsConfig
from z3alpha.tactics.catalog import tactic_name_for_action
from z3alpha.utils import encode_strat_row

logger = logging.getLogger(__name__)


class LinearStrategySearchRun(BaseMCTSRun):
    stage = 1

    def __init__(
        self,
        config: MctsConfig,
        bench_lst,
        logic,
        z3path,
        value_type,
        log_folder,
        batch_size: int = 1,
        logic_config=None,
    ) -> None:
        self.logic_config = logic_config
        super().__init__(
            config,
            bench_lst,
            logic,
            z3path,
            value_type,
            log_folder,
            batch_size=batch_size,
        )

    def _init_stage_state(self) -> None:
        self.res_database: dict = {}
        self._s1_csv_path = self.log_folder / "linear_strategy_results.csv"
        self._written_strats: set = set()
        with open(self._s1_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["strat"] + list(self.training_list))
        self._scorer: LLMPriorScorer | None = None
        lp = self.config.llm_prior
        if lp is not None and lp.enabled:
            cfg = replace(lp, cache_path=self.log_folder / "llm_priors.json")
            self._scorer = LLMPriorScorer(cfg)

    def _priors_for(self, actions: list) -> dict[Any, float] | None:
        if self._scorer is None:
            return None
        names = [tactic_name_for_action(int(a)) for a in actions]
        by_name = self._scorer.score(self.logic, str(self.env), names)
        return {a: float(by_name[names[i]]) for i, a in enumerate(actions)}

    def _create_env(self):
        return LinearStrategyGame(
            self.training_list,
            self.logic,
            self.timeout,
            self.batch_size,
            z3path=self.z3path,
            logic_config=self.logic_config,
            params_for=self.params_for,
        )

    def _per_simulation_log(self, sim_index: int) -> None:
        logger.info(f"Simulation {sim_index} starts")

    def _after_simulation(self) -> None:
        self._write_new_results()

    def _write_new_results(self) -> None:
        """Append any newly evaluated strategies to linear results CSV."""
        new_strats = set(self.res_database.keys()) - self._written_strats
        if not new_strats:
            return
        with open(self._s1_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for strat in new_strats:
                writer.writerow(encode_strat_row(strat, self.res_database[strat]))
        self._written_strats.update(new_strats)
