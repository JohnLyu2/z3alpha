from z3alpha.mcts import BaseMCTSRun
from z3alpha.stage2.environment import Stage2StrategyGame


class Stage2MCTSRun(BaseMCTSRun):
    stage = 2

    def _init_stage_state(self, log_folder, bench_lst):
        self.c_ucb = None
        self.res_database = self.config["stage2_context"].result_cache

    def _create_env(self):
        return Stage2StrategyGame(
            self.stage,
            self.training_list,
            self.logic,
            self.timeout,
            self.config,
            self.batch_size,
            z3path=self.z3path,
        )
