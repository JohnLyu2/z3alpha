import time
import random
import csv
from pathlib import Path
import argparse
import logging
import json
import datetime


from z3 import *

from z3alpha.evaluator import SolverEvaluator
from z3alpha.mcts import MCTS_RUN
from z3alpha.selector import linear_strategy_select, convert_strats_to_act_lists
from z3alpha.utils import calculatePercentile, write_strat_res_to_csv

from z3alpha.strat_tree import PERCENTILES

VALUE_TYPE = "par10"  # hard code for now


def createBenchmarkList(
    benchmark_directories, timeout, batchSize, tmp_folder, z3path, is_sorted
):
    # benchmarkLst = [str(p) for p in sorted(list(pathlib.Path(benchmark_directory).rglob(f"*.smt2")))]
    benchmarkLst = []
    for dir in benchmark_directories:
        assert Path(dir).exists()
        benchmarkLst += [
            str(p) for p in sorted(list(Path(dir).rglob(f"*.smt2")))
        ]
    benchmarkLst.sort()
    if not is_sorted:
        return benchmarkLst
    evaluator = SolverEvaluator(
        z3path, benchmarkLst, timeout, batchSize, tmp_dir=tmp_folder
    )
    resLst = evaluator.getResLst(None)
    # par2 list from resLst; for each entry (solved, time) in resLst, if solved, return time; else return 2 * timeout
    par2Lst = [2 * timeout if not res[0] else res[1] for res in resLst]
    # # sort benchmarkLst resLst into a ascending list by par2Lst
    # benchmarkLst = [x for _, x in sorted(zip(par2Lst, benchmarkLst))]
    # sort benchmarkLst resLst into a descending list by par2Lst
    benchmarkLst = [x for _, x in sorted(zip(par2Lst, benchmarkLst), reverse=True)]
    return benchmarkLst


def createProbeStats(bench_lst):
    numConstsLst = []
    numExprsLst = []
    sizeLst = []
    probeRecords = []
    for smt_path in bench_lst:
        instanceDict = {}
        formula = parse_smt2_file(smt_path)
        constProbe = Probe("num-consts")
        exprProbe = Probe("num-exprs")
        sizeProbe = Probe("size")
        goal = Goal()
        goal.add(formula)
        numConsts = constProbe(goal)
        numConstsLst.append(numConsts)
        instanceDict["num-consts"] = numConsts
        numExprs = exprProbe(goal)
        numExprsLst.append(numExprs)
        instanceDict["num-exprs"] = numExprs
        size = sizeProbe(goal)
        sizeLst.append(size)
        instanceDict["size"] = size
        probeRecords.append(instanceDict)
    # get 90 percentile, 70 percentile and median from lists
    probeStats = {}
    # contents of probeStats['num-consts'] is another dict, key is the percentile and value is the value
    probeStats["num-consts"] = {
        percentile: calculatePercentile(numConstsLst, percentile)
        for percentile in PERCENTILES
    }
    probeStats["num-exprs"] = {
        percentile: calculatePercentile(numExprsLst, percentile)
        for percentile in PERCENTILES
    }
    probeStats["size"] = {
        percentile: calculatePercentile(sizeLst, percentile)
        for percentile in PERCENTILES
    }
    return probeStats, probeRecords


def stage1_synthesize(config, stream_logger, log_folder):
    startTime = time.time()
    logic = config["logic"]
    z3path = config["z3path"] if "z3path" in config else "z3"
    tmp_folder = config["temp_folder"]
    batch_size = config["batch_size"]
    s1config = config["s1config"]
    num_ln_strat = config["ln_strat_num"]
    tmp_folder = config["temp_folder"]
    random_seed = config["random_seed"]
    random.seed(random_seed)

    # Stage 1
    s1_bench_dirs = s1config["bench_dirs"]
    s1BenchLst = createBenchmarkList(
        s1_bench_dirs,
        s1config["timeout"],
        batch_size,
        tmp_folder,
        z3path,
        is_sorted=True,
    )
    stream_logger.info("S1 MCTS Simulations Start")
    run1 = MCTS_RUN(
        1,
        s1config,
        s1BenchLst,
        logic,
        z3path,
        VALUE_TYPE,
        log_folder,
        tmp_folder=tmp_folder,
        batch_size=batch_size,
    )
    run1.start()
    s1_res_dict = run1.getResDict()

    selected_strat, ln_select_logs = linear_strategy_select(
        num_ln_strat, s1_res_dict, s1config["timeout"]
    )
    stream_logger.info(ln_select_logs)
    lnStratCandidatsPath = Path(log_folder) / "ln_strat_candidates.csv"
    with open(lnStratCandidatsPath, "w") as f:
        # write one strategy per line as a csv file
        cwriter = csv.writer(f)
        # header (one column "strat")
        cwriter.writerow(["strat"])
        for strat in selected_strat:
            cwriter.writerow([strat])
    stream_logger.info(
        f"Selected {len(selected_strat)} strategies: {selected_strat}, saved to {lnStratCandidatsPath}"
    )

    endTime = time.time()
    s1time = endTime - startTime
    stream_logger.info(f"Stage 1 Time: {s1time:.0f}")
    return selected_strat

def add_fail_if_undecided(strat):
    return f"(then {strat} fail-if-undecided)"

def parallel_linear_strategies(ln_strat_lst, fail_if_undecided=True):
    assert len(ln_strat_lst) > 0, "No linear strategies provided"
    if len(ln_strat_lst) == 1:
        return ln_strat_lst[0]
    parallel_strats = "(par-or"
    for strat in ln_strat_lst:
        if fail_if_undecided:
            strat = add_fail_if_undecided(strat)
        parallel_strats += f" {strat}"
    parallel_strats += ")"
    return parallel_strats

def cache4stage2(selected_strat, config, stream_logger, log_folder, benchlst=None):
    startTime = time.time()
    num_strat = config["ln_strat_num"]
    # assert len(selected_strat) >= num_strat
    selected_strat = selected_strat[:num_strat]
    z3path = config["z3path"] if "z3path" in config else "z3"
    s2config = config["s2config"]
    s2benchDirs = s2config["bench_dirs"]
    s2timeout = s2config["timeout"]
    batch_size = config["batch_size"]
    tmp_folder = config["temp_folder"]
    s2benchLst = (
        benchlst
        if benchlst
        else createBenchmarkList(
            s2benchDirs, s2timeout, batch_size, tmp_folder, z3path, is_sorted=True
        )
    )
    # s2benchLst = createBenchmarkList(s2benchDirs, s2timeout, batch_size, tmp_folder, z3path, is_sorted=True)
    s2_res_lst = []
    s2evaluator = SolverEvaluator(
        z3path, s2benchLst, s2timeout, batch_size, tmp_dir=tmp_folder
    )
    for i in range(len(selected_strat)):
        strat = selected_strat[i]
        stream_logger.info(f"Stage 2 Caching: {i + 1}/{len(selected_strat)}")
        strat_res = s2evaluator.getResLst(strat)
        s2_res_lst.append((strat, strat_res))
    ln_res_csv = Path(log_folder) / "ln_res.csv"
    write_strat_res_to_csv(s2_res_lst, ln_res_csv, s2benchLst)
    stream_logger.info(f"Cached results saved to {ln_res_csv}")
    endTime = time.time()
    cacheTime = endTime - startTime
    stream_logger.info(f"Stage 2 Cache Time: {cacheTime:.0f}")
    return s2_res_lst, s2benchLst


def stage2_synthesize(results, bench_lst, config, stream_logger, log_folder):
    num_strat = config["ln_strat_num"]
    # assert len(results) >= num_strat
    # results is a list of (strat, res_lst); get the first num_strat strat
    results = results[:num_strat]
    res_dict = {}
    for strat, res_lst in results:
        res_dict[strat] = res_lst
    selected_strat = list(res_dict.keys())
    act_lst, solver_dict, preprocess_dict, s1strat2acts = convert_strats_to_act_lists(
        selected_strat
    )
    stream_logger.info(f"preprocess dict: {preprocess_dict}")
    stream_logger.info(f"solver dict: {solver_dict}")
    stream_logger.info(f"converted selected strategies: {act_lst}")

    s2_res_dict_acts = {}
    for strat in s1strat2acts:
        s2_res_dict_acts[s1strat2acts[strat]] = res_dict[strat]

    s2dict = {}
    s2dict["s1_strats"] = act_lst
    s2dict["solver_dict"] = solver_dict
    s2dict["preprocess_dict"] = preprocess_dict
    s2dict["res_cache"] = s2_res_dict_acts
    logic = config["logic"]
    z3path = "z3"
    if "z3path" in config:
        z3path = config["z3path"]
    tmp_folder = config["temp_folder"]

    s2startTime = time.time()
    stream_logger.info("S2 MCTS Simulations Start")

    probeStats, probeRecords = createProbeStats(bench_lst)
    s2dict["probe_stats"] = probeStats
    s2dict["probe_records"] = probeRecords
    s2config = config["s2config"]
    s2config["s2dict"] = s2dict

    run2 = MCTS_RUN(
        2,
        s2config,
        bench_lst,
        logic,
        z3path,
        VALUE_TYPE,
        log_folder,
        tmp_folder=tmp_folder,
    )
    run2.start()
    best3_s2 = run2.getBest3Strats()

    # Only output the best strategy
    best_strategy = best3_s2[0]
    stratPath = Path(log_folder) / "final_strategy.txt"
    with open(stratPath, "w") as f:
        f.write(best_strategy)
    stream_logger.info(f"Best final strategy saved to: {stratPath}")

    s2endTime = time.time()
    s2time = s2endTime - s2startTime
    stream_logger.info(f"Stage 2 MCTS Time: {s2time:.0f}")
    return best3_s2


def branched_synthesize(config, log_folder=None, stream_logger=None):
    """
    Perform the complete synthesis for branched strategy [IJCAI 2024].
    
    Args:
        config: Configuration dictionary containing synthesis parameters
        log_folder: Optional log folder path. If None, creates a timestamped folder
        stream_logger: Optional logger. If None, creates a default logger
    
    """
    start_time = time.time()
    
    # Setup logger if not provided
    if stream_logger is None:
        stream_logger = logging.getLogger(__name__)
        stream_logger.setLevel(logging.INFO)
        log_handler = logging.StreamHandler()
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S")
        )
        stream_logger.addHandler(log_handler)
    
    # Setup log folder if not provided
    if log_folder is None:
        log_folder = f"experiments/synthesis/out-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}/"
        assert not Path(log_folder).exists()
        Path(log_folder).mkdir(parents=True, exist_ok=False)
    
    # Stage 1: Initial strategy synthesis
    selected_strats = stage1_synthesize(config, stream_logger, log_folder)
    
    # Stage 2: Cache results for selected strategies
    res_dict, bench_lst = cache4stage2(
        selected_strats, config, stream_logger, log_folder
    )
    
    # Stage 3: Final strategy synthesis
    bst_strats = stage2_synthesize(
        res_dict, bench_lst, config, stream_logger, log_folder
    )
    
    # Log results
    order = 1
    for bst_strat in bst_strats:
        stream_logger.info(f"Best strategy {order}: {bst_strat}")
        order += 1
    
    total_time = time.time() - start_time
    stream_logger.info(f"Total synthesis time: {total_time:.0f} seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_config", type=str, help="The experiment configuration file in json"
    )
    args = parser.parse_args()
    config = json.load(open(args.json_config, "r"))
    
    # Use the new full_synthesize function
    branched_synthesize(config)


if __name__ == "__main__":
    main()
