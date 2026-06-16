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
from z3alpha.utils import par_n, solved_num

logger = logging.getLogger(__name__)


class LinearStrategySearchRun(BaseMCTSRun):
    stage = 1
    TRACE_LOG_FILENAME = "linear_strategy_mcts.log"

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
        self._strat_first_sim: dict[str, int] = {}
        self._summary_csv_path = self.log_folder / "linear_strategy_summary.csv"
        self._per_bench_csv_path = self.log_folder / "linear_strategy_per_benchmark.csv"
        self._written_strats: set = set()
        with open(self._summary_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["id", "strategy", "n_solved", "par2_avg", "par10_avg"]
            )
        with open(self._per_bench_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["strat", "benchmark", "status", "time_s", "solved"])
        self._scorer: LLMPriorScorer | None = None
        lp = self.config.llm_prior
        if lp is not None and lp.enabled:
            llm_log_path = str(self.log_folder / "llm_prior_qa.log")
            self._scorer = LLMPriorScorer(replace(lp, qa_log_path=llm_log_path))

    def _priors_for(self, actions: list) -> dict[Any, float] | None:
        if self._scorer is None:
            return None
        names = [tactic_name_for_action(int(a)) for a in actions]
        by_name = self._scorer.score(
            self.logic, str(self.env), names, sim_id=self.num_sim
        )
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
        for strat in self.res_database:
            if strat not in self._strat_first_sim:
                self._strat_first_sim[strat] = self.num_sim
        self._write_new_results()

    def _write_summary_table(self) -> None:
        with open(self._summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["id", "strategy", "n_solved", "par2_avg", "par10_avg"]
            )
            for strat in sorted(self.res_database.keys(), key=lambda s: (self._strat_first_sim[s], s)):
                res = self.res_database[strat]
                nb = len(res)
                ns = solved_num(res)
                par2_avg = par_n(res, 2, self.timeout) / nb
                par10_avg = par_n(res, 10, self.timeout) / nb
                writer.writerow(
                    [self._strat_first_sim[strat], strat, ns, par2_avg, par10_avg]
                )

    def _write_new_results(self) -> None:
        """Append per-benchmark rows for new strategies; refresh summary table."""
        new_strats = set(self.res_database.keys()) - self._written_strats
        if not new_strats:
            return
        with open(self._per_bench_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for strat in sorted(new_strats, key=lambda s: (self._strat_first_sim[s], s)):
                row_list = self.res_database[strat]
                for bench, row in zip(self.training_list, row_list):
                    solved_b, time_s, status = row[0], row[1], row[2]
                    writer.writerow([strat, bench, status, time_s, solved_b])
        self._written_strats.update(new_strats)
        self._write_summary_table()
