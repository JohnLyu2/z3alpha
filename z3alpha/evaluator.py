import subprocess
import time
import logging
import csv
import concurrent.futures

from z3alpha.utils import par_n, solved_num

logger = logging.getLogger(__name__)

__all__ = ["SolverEvaluator", "SolverRunner"]


class SolverRunner:
    """Runner which executes a solver on a single formula via subprocess.

    If z3_strategy is not None, the solver is assumed to be Z3 and the
    strategy is passed via the tactic.default_tactic parameter.
    """

    def __init__(self, solver_path, smt_file, timeout, run_id, z3_strategy=None):
        self.solver_path = solver_path
        self.smt_file = smt_file
        self.timeout = timeout
        self.z3_strategy = z3_strategy
        self.run_id = run_id

    def _build_cmd(self) -> list[str]:
        cmd = [self.solver_path]
        if self.z3_strategy is not None:
            cmd.append(f"tactic.default_tactic={self.z3_strategy}")
        cmd.append(self.smt_file)
        return cmd

    def _parse_output(self, out: bytes | None, runtime: float):
        """Parse solver stdout into a result tuple (run_id, result_str, runtime, smt_file)."""
        if not out:
            logger.warning(
                f"Empty output from solver: {self.solver_path}\n strategy: {self.z3_strategy}\ninstance: {self.smt_file}"
            )
            return self.run_id, "error", runtime, self.smt_file

        try:
            text = out.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning(
                f"Failed to decode solver output: {self.solver_path}\ninstance: {self.smt_file}\n{e}"
            )
            return self.run_id, "error", runtime, self.smt_file

        lines = text.rstrip("\n").split("\n")
        if not lines or not lines[0].strip():
            logger.warning(
                f"No lines in solver output: {self.solver_path}\ninstance: {self.smt_file}"
            )
            return self.run_id, "error", runtime, self.smt_file

        res = lines[0]
        if res.startswith("(error"):
            logger.warning(
                f"Error occurred when solver: {self.solver_path}\n strategy: {self.z3_strategy}\ninstance: {self.smt_file}\nMessage: {res}"
            )
            return self.run_id, "error", runtime, self.smt_file

        return self.run_id, res, runtime, self.smt_file

    def execute(self):
        """Run solver synchronously with self-managed timeout.

        Returns (run_id, result_str, runtime, smt_file).
        Suitable for use inside a ThreadPoolExecutor.
        """
        time_before = time.time()
        p = subprocess.Popen(self._build_cmd(), stdout=subprocess.PIPE)
        try:
            out, _ = p.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate()
            return self.run_id, "timeout", self.timeout, self.smt_file
        return self._parse_output(out, time.time() - time_before)


class SolverEvaluator:
    """Evaluates a solver on a benchmark list.

    Timeout (seconds) is enforced per instance via subprocess and used for
    PAR scoring.
    """

    def __init__(
        self,
        solver_path,
        benchmark_lst,
        timeout,
        batch_size,
        is_write_res=False,
        res_path=None,
    ):
        self.solver_path = solver_path
        self.benchmark_list = benchmark_lst
        assert self.get_benchmark_size() > 0
        self.timeout = timeout
        assert self.timeout > 0
        self.batch_size = batch_size
        self.is_write_res = is_write_res
        self.res_path = res_path

    def get_benchmark_size(self):
        return len(self.benchmark_list)

    def _run_single(self, run_id, strat_str):
        """Create a SolverRunner and execute it synchronously (for pool use)."""
        runner = SolverRunner(
            self.solver_path,
            self.benchmark_list[run_id],
            self.timeout,
            run_id,
            strat_str,
        )
        return runner.execute()

    def get_res_list(self, strat_str):
        size = self.get_benchmark_size()
        results = [None] * size

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = {
                executor.submit(self._run_single, i, strat_str): i
                for i in range(size)
            }
            for future in concurrent.futures.as_completed(futures):
                run_id, resTask, timeTask, pathTask = future.result()
                solved = resTask == "sat" or resTask == "unsat"
                results[run_id] = (solved, timeTask, resTask)

        for i in range(size):
            assert results[i] is not None
        return results

    def evaluate(self, strat_str):
        results = self.get_res_list(strat_str)
        if self.is_write_res:
            with open(self.res_path, "w") as f:
                writer = csv.writer(f)
                # write header
                writer.writerow(["id", "path", "solved", "time", "result"])
                for i in range(len(self.benchmark_list)):
                    writer.writerow(
                        [
                            i,
                            self.benchmark_list[i],
                            results[i][0],
                            results[i][1],
                            results[i][2],
                        ]
                    )
        solved = solved_num(results)
        par2 = par_n(results, 2, self.timeout)
        par10 = par_n(results, 10, self.timeout)
        return (solved, par2, par10)
