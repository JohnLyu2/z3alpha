import os
import threading
import subprocess
import shlex
import time
import logging
import csv

from z3alpha.utils import solvedNum, parN

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
log.addHandler(log_handler)


class SolverRunner(threading.Thread):
    """Runner which executes (a single strategy) on a single formula by calling a solver in shell"""

    def __init__(
        self, solver_path, smt_file, timeout, id, strategy=None, tmp_dir="/tmp/"
    ):
        threading.Thread.__init__(self)
        self.solver_path = solver_path
        self.smt_file = smt_file
        self.timeout = timeout  # used only for output
        self.strategy = strategy
        self.id = id
        self.tmpDir = tmp_dir

        if self.strategy is not None:
            if not os.path.exists(self.tmpDir):
                os.makedirs(self.tmpDir)
            self.new_file_name = os.path.join(self.tmpDir, f"tmp_{id}.smt2")
            self.tmp_file = open(self.new_file_name, "w")
            with open(self.smt_file, "r") as f:
                for line in f:
                    new_line = line
                    if "check-sat" in line:
                        new_line = f"(check-sat-using {strategy})\n"
                    self.tmp_file.write(new_line)
            self.tmp_file.close()
        else:
            self.new_file_name = self.smt_file

    def run(self):
        self.time_before = time.time()
        safe_path = shlex.quote(self.new_file_name)
        cmd = f"{self.solver_path} {safe_path}"
        self.p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)
        self.p.wait()
        self.time_after = time.time()

    def collect(self):
        if self.is_alive():
            try:
                self.p.terminate()
                self.join()
            except OSError:
                pass
            return self.id, "timeout", self.timeout, self.smt_file

        out, err = self.p.communicate()

        lines = out[:-1].decode("utf-8").split("\n")
        res = lines[0]
        runtime = self.time_after - self.time_before

        # this error format may not be the same with solvers other than z3
        if res.startswith("(error"):
            log.warn(
                f"Error occurred when solver: {self.solver_path}\n strategy: {self.strategy}\ninstance: {self.smt_file}\nMessage: {res}"
            )
            return self.id, "error", runtime, self.smt_file

        # rlimit = None
        # for line in lines:
        #     if 'rlimit' in line:
        #         tokens = line.split(' ')
        #         for token in tokens:
        #             if token.isdigit():
        #                 rlimit = int(token)

        return self.id, res, runtime, self.smt_file


class SolverEvaluator:
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

    # now returns a list of (solved, timeTask, resTask) for each instance
    def getResLst(self, strat_str):
        size = self.getBenchmarkSize()
        results = [None] * size
        for i in range(0, size, self.batchSize):
            batch_instance_ids = range(i, min(i + self.batchSize, size))
            threads = []
            for id in batch_instance_ids:
                smtfile = self.benchmarkLst[id]
                runnerThread = SolverRunner(
                    self.solverPath, smtfile, self.timeout, id, strat_str, self.tmpDir
                )
                runnerThread.start()
                threads.append(runnerThread)
            time_start = time.time()
            for task in threads:
                time_left = max(0, self.timeout - (time.time() - time_start))
                task.join(time_left)
                id, resTask, timeTask, pathTask = task.collect()
                solved = True if (resTask == "sat" or resTask == "unsat") else False
                results[id] = (solved, timeTask, resTask)
        # assert no entries in results is still -1
        for i in range(size):
            assert results[i] != None
        return results

    # returns a tuple (#solved, par2, par10)
    def testing(self, strat_str):
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
