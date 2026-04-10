import os
import threading
import subprocess
import time
import logging
import csv
import concurrent.futures

from z3alpha.utils import solvedNum, parN

logger = logging.getLogger(__name__)

__all__ = ["SolverEvaluator", "SolverRunner"]


class SolverRunner(threading.Thread):
    """Runner which executes a solver on a single formula via subprocess.

    If z3_strategy is not None, the solver is assumed to be Z3 and the input
    file is rewritten to use (check-sat-using <z3_strategy>).
    """

    def __init__(
        self,
        solver_path,
        smt_file,
        timeout,
        run_id,
        z3_strategy=None,
        tmp_dir="/tmp/",
    ):
        super().__init__()
        self.solver_path = solver_path
        self.smt_file = smt_file
        self.timeout = timeout
        self.z3_strategy = z3_strategy
        self.run_id = run_id
        self.tmpDir = tmp_dir

        if self.z3_strategy is not None:
            os.makedirs(self.tmpDir, exist_ok=True)
            unique_id = f"{os.getpid()}_{int(time.time() * 1000)}_{run_id}"
            self.new_file_name = os.path.join(self.tmpDir, f"tmp_{unique_id}.smt2")
            with open(self.new_file_name, "w") as tmp_file, open(self.smt_file, "r") as f:
                for line in f:
                    if "check-sat" in line:
                        tmp_file.write(f"(check-sat-using {z3_strategy})\n")
                    else:
                        tmp_file.write(line)
        else:
            self.new_file_name = self.smt_file

    def _build_cmd(self) -> list[str]:
        return [self.solver_path, self.new_file_name]

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

    def run(self):
        self.time_before = time.time()
        self.p = subprocess.Popen(self._build_cmd(), stdout=subprocess.PIPE)
        self.p.wait()
        self.time_after = time.time()

    def _remove_tmp_file(self) -> None:
        """Remove temp SMT file if we created one (z3_strategy was not None)."""
        if self.z3_strategy is not None and os.path.isfile(self.new_file_name):
            try:
                os.remove(self.new_file_name)
            except OSError:
                pass

    def execute(self):
        """Run solver synchronously with self-managed timeout.

        Returns (run_id, result_str, runtime, smt_file).
        Suitable for use inside a ThreadPoolExecutor.
        """
        time_before = time.time()
        try:
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
        finally:
            self._remove_tmp_file()

    def collect(self):
        if self.is_alive():
            try:
                self.p.terminate()
                self.join()
            except OSError:
                pass
            self._remove_tmp_file()
            return self.run_id, "timeout", self.timeout, self.smt_file

        try:
            out, _ = self.p.communicate()
            return self._parse_output(out, self.time_after - self.time_before)
        finally:
            self._remove_tmp_file()


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
        tmp_dir="/tmp/",
        is_write_res=False,
        res_path=None,
    ):
        self.solverPath = solver_path
        self.benchmarkLst = benchmark_lst
        assert self.getBenchmarkSize() > 0
        self.timeout = timeout
        assert self.timeout > 0
        self.batchSize = batch_size
        self.tmpDir = tmp_dir
        self.isWriteRes = is_write_res
        self.resPath = res_path

    def getBenchmarkSize(self):
        return len(self.benchmarkLst)

    def _run_single(self, run_id, strat_str):
        """Create a SolverRunner and execute it synchronously (for pool use)."""
        runner = SolverRunner(
            self.solverPath,
            self.benchmarkLst[run_id],
            self.timeout,
            run_id,
            strat_str,
            self.tmpDir,
        )
        return runner.execute()

    def getResLst(self, strat_str):
        size = self.getBenchmarkSize()
        results = [None] * size

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.batchSize) as executor:
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
        results = self.getResLst(strat_str)
        if self.isWriteRes:
            with open(self.resPath, "w") as f:
                writer = csv.writer(f)
                # write header
                writer.writerow(["id", "path", "solved", "time", "result"])
                for i in range(len(self.benchmarkLst)):
                    writer.writerow(
                        [
                            i,
                            self.benchmarkLst[i],
                            results[i][0],
                            results[i][1],
                            results[i][2],
                        ]
                    )
        solved = solvedNum(results)
        par2 = parN(results, 2, self.timeout)
        par10 = parN(results, 10, self.timeout)
        return (solved, par2, par10)
