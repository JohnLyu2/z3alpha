import os
import csv
import pathlib
import sys
import argparse
import json
sys.path.append('./')
sys.path.append('../')

from alphasmt.evaluator import SolverEvaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_config', type=str, help='The experiment testing configuration file in json')
    configJsonPath = parser.parse_args()
    config = json.load(open(configJsonPath.json_config, 'r'))

    solvers = config['solvers']
    z3strats = config['strat_files'] if 'strat_files' in config else None
    timeout = config['timeout']
    batchSize = config['batch_size']
    res_dir = config['res_dir']
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    test_dir = config['test_dir']
    tmp_dir = config['tmp_dir'] if 'tmp_dir' in config else "/tmp/"

    test_solvers = {}

    for solver in solvers:
        solver_path = solvers[solver]
        if solver == 'z3':
            if z3strats:
                for z3stat in z3strats:
                    test_solvers[z3stat] = (solver_path, z3strats[z3stat])
            else:
                test_solvers['z3'] = (solver_path, None)
        else:
            test_solvers[solver] = (solver_path, None)

    test_lst = sorted(str(p.resolve()) for p in pathlib.Path(test_dir).rglob("*.smt2"))
    res_csv = os.path.join(res_dir, "res.csv")
    with open(res_csv, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["solver", "solved", "par2", "par10"]) # write header
        for solver in test_solvers.keys():
            solver_path, strat_path = test_solvers[solver]
            strat = open(strat_path, 'r').read() if strat_path else None
            csv_path = os.path.join(res_dir, f"{solver}.csv")
            testEvaluator = SolverEvaluator(solver_path, test_lst, timeout, batchSize, is_write_res=True, res_path=csv_path, tmp_dir=tmp_dir)
            resTuple = testEvaluator.testing(strat)
            csvwriter.writerow([solver, resTuple[0], resTuple[1], resTuple[2]])
            print(f"{solver} test results: solved {resTuple[0]} instances with par2 {resTuple[1]} and par10 {resTuple[2]:.2f}")

if __name__ == "__main__":
    main()