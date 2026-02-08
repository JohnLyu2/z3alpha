import argparse
import csv
import json
import os
import pathlib
import sys
from datetime import datetime

from z3alpha.evaluator import SolverEvaluator, _z3_timeout_arg

_REQUIRED_KEYS = ("solvers", "timeout", "batch_size", "res_dir", "test_dir")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate solvers and Z3 strategies on a benchmark set.",
    )
    parser.add_argument(
        "json_config",
        type=str,
        help="Path to the experiment configuration JSON file",
    )
    args = parser.parse_args()

    with open(args.json_config) as f:
        config = json.load(f)

    missing = [k for k in _REQUIRED_KEYS if k not in config]
    if missing:
        sys.exit(f"Config missing required keys: {missing}")

    solvers = config["solvers"]
    z3strats = config.get("strat_files")
    timeout = config["timeout"]
    batch_size = config["batch_size"]
    res_dir = config["res_dir"]
    test_dir = config["test_dir"]
    tmp_dir = config.get("tmp_dir", "/tmp/")

    test_dir_path = pathlib.Path(test_dir)
    if not test_dir_path.is_dir():
        sys.exit(f"test_dir does not exist or is not a directory: {test_dir}")

    test_lst = sorted(str(p.resolve()) for p in test_dir_path.rglob("*.smt2"))
    if not test_lst:
        sys.exit(f"No .smt2 files found under test_dir: {test_dir}")

    os.makedirs(res_dir, exist_ok=True)

    test_solvers = {}
    for solver in solvers:
        solver_path = solvers[solver]
        if solver == "z3":
            if z3strats:
                for z3strat in z3strats:
                    test_solvers[z3strat] = (solver_path, z3strats[z3strat])
            else:
                test_solvers["z3"] = (solver_path, None)
        else:
            test_solvers[solver] = (solver_path, None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_csv = os.path.join(res_dir, f"res_{timestamp}.csv")
    with open(res_csv, "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["solver", "solved", "par2", "par10"])  # write header
        for solver in test_solvers.keys():
            solver_path, strat_path = test_solvers[solver]
            strat = (
                pathlib.Path(strat_path).read_text() if strat_path else None
            )
            timeout_solver_arg = (
                _z3_timeout_arg
                if (solver == "z3" or (z3strats and solver in z3strats))
                else None
            )
            csv_path = os.path.join(res_dir, f"{solver}.csv")
            evaluator = SolverEvaluator(
                solver_path,
                test_lst,
                timeout,
                batch_size,
                is_write_res=True,
                res_path=csv_path,
                tmp_dir=tmp_dir,
                timeout_solver_arg=timeout_solver_arg,
            )
            solved, par2, par10 = evaluator.testing(strat)
            csvwriter.writerow([solver, solved, par2, par10])
            print(
                f"{solver} test results: solved {solved} instances with par2 {par2} and par10 {par10:.2f}"
            )


if __name__ == "__main__":
    main()
