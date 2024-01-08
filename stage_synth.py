# import os
# import logging 
# import argparse
# import json
# import datetime
# import pathlib
# import random
# import time
# from z3 import *

# from alphasmt.mcts import MCTS_RUN
# from alphasmt.evaluator import Z3StrategyEvaluator
# from alphasmt.selector import * 

# log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)
# log_handler = logging.StreamHandler()
# log_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(message)s','%Y-%m-%d %H:%M:%S'))
# log.addHandler(log_handler)

# import functools
# print = functools.partial(print, flush=True)

# def calculatePercentile(lst, percentile):
#     assert (len(lst) > 0)
#     sortedLst = sorted(lst)
#     index = int(len(sortedLst) * percentile)
#     return sortedLst[index]

# def createProbeStatDict(benchmark_directory):
#     numConstsLst = []
#     numExprsLst = []
#     sizeLst = []
#     benchmarkLst = [str(p) for p in sorted(list(pathlib.Path(benchmark_directory).rglob(f"*.smt2")))]
#     # no parllel now 
#     for smt_path in benchmarkLst:
#         formula = parse_smt2_file(smt_path)
#         constProbe = Probe('num-consts')
#         exprProbe = Probe('num-exprs')
#         sizeProbe = Probe('size')
#         goal = Goal()
#         goal.add(formula)
#         numConstsLst.append(constProbe(goal))
#         numExprsLst.append(exprProbe(goal))
#         sizeLst.append(sizeProbe(goal))
#     # get 90 percentile, 70 percentile and median from lists
#     percentileLst = [0.9, 0.7, 0.5]
#     probeStats = {}
#     # to-do: make probe as list
#     probeStats[50] = {"value": [calculatePercentile(numConstsLst, p) for p in percentileLst]} # action 50: probe num-consts
#     probeStats[51] = {"value": [calculatePercentile(numExprsLst, p) for p in percentileLst]} # action 51: probe num-exprs
#     probeStats[52] = {"value": [calculatePercentile(sizeLst, p) for p in percentileLst]} # action 52: probe size
#     return probeStats

# # is_sorted indicates whether the benchmark list is asecendingly sorted by solving time
# def createBenchmarkList(benchmark_directory, timeout, batchSize, tmp_folder, is_sorted):
#     benchmarkLst = [str(p) for p in sorted(list(pathlib.Path(benchmark_directory).rglob(f"*.smt2")))]
#     if not is_sorted:
#         return benchmarkLst
#     evaluator = Z3StrategyEvaluator(benchmarkLst, timeout, batchSize, tmp_dir=tmp_folder)
#     resLst = evaluator.getResLst(None)
#     # par2 list from resLst; for each entry (solved, time) in resLst, if solved, return time; else return 2 * timeout
#     par2Lst = [2 * timeout if not res[0] else res[1] for res in resLst]
#     # sort benchmarkLst resLst into a ascending list by par2Lst
#     benchmarkLst = [x for _, x in sorted(zip(par2Lst, benchmarkLst))]
#     return benchmarkLst

# def main():
#     s1startTime = time.time()
#     parser = argparse.ArgumentParser()
#     parser.add_argument('json_config', type=str, help='The experiment configuration file in json')
#     configJsonPath = parser.parse_args()
#     config = json.load(open(configJsonPath.json_config, 'r'))

#     logic = config['logic']
#     batch_size = config['batch_size']
#     s1config = config['s1config']
#     s2config = config['s2config']
#     num_ln_strat = config['ln_strat_num']
#     tmp_folder = config['temp_folder']
#     random_seed = config['random_seed']
#     random.seed(random_seed)

#     log_folder = f"experiments/results/out-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}/"
#     assert(not os.path.exists(log_folder))
#     os.makedirs(log_folder)

#     # trainProbeStatDict = createProbeStatDict(train_path)
#     # log.info(f"Probe Stats: {trainProbeStatDict}")
    
#     value_type = 'par10' # hard code for now

#     # Stage 1
#     s1_bench_dir = s1config['bench_dir']
#     s1BenchLst = createBenchmarkList(s1_bench_dir, s1config["timeout"], batch_size, tmp_folder, is_sorted=True)

#     log.info("S1 MCTS Simulations Start")
#     run1 = MCTS_RUN(1, s1config, s1BenchLst, logic, value_type, log_folder, tmp_folder=tmp_folder, batch_size = batch_size)
#     run1.start()
#     s1_res_dict = run1.getResDict()

#     selected_strat = linear_strategy_select(num_ln_strat, s1_res_dict, s1config["timeout"])
#     lnStratCandidatsPath = f"{log_folder}/ln_strat_candidates.txt"
#     with open(lnStratCandidatsPath, 'w') as f:
#        for strat in selected_strat:
#             f.write(strat + '\n')
#     log.info(f"Selected {len(selected_strat)} strategies: {selected_strat}, saved to {lnStratCandidatsPath}")

#     s2startTime = time.time()
#     log.info(f"Stage 1 Time: {s2startTime - s1startTime:.0f}")

#     # Stage 2
#     act_lst, solver_dict, preprocess_dict, s1strat2acts = convert_strats_to_act_lists(selected_strat)
#     log.info(f"solver dict: {solver_dict}")
#     log.info(f"preprocess dict: {preprocess_dict}")
#     log.info(f"converted selected strategies: {act_lst}")

#     s2benchDir = s2config['bench_dir']
#     s2timeout = s2config["timeout"]
#     s2benchLst = createBenchmarkList(s2benchDir, s2timeout, batch_size, tmp_folder, is_sorted=True)

#     s2_res_dict = {}
#     s2evaluator = Z3StrategyEvaluator(s2benchLst, s2timeout, batch_size, tmp_dir=tmp_folder)
#     for i in range(len(selected_strat)):
#         strat = selected_strat[i]
#         log.info(f"Stage 2 Caching: {i+1}/{len(selected_strat)}")
#         s2_res_dict[strat] = s2evaluator.getResLst(strat)

#     s2_res_dict_acts = {}
#     for strat in s1strat2acts:
#         s2_res_dict_acts[s1strat2acts[strat]] = s2_res_dict[strat]

#     s2config['s1_strats'] = act_lst
#     s2config['solver_dict'] = solver_dict
#     s2config['preprocess_dict'] = preprocess_dict
#     s2config['res_cache'] = s2_res_dict_acts

#     s2caching_end_time = time.time()
#     log.info(f"Stage 2 Caching Time: {s2caching_end_time - s2startTime:.0f}")
#     log.info("S2 MCTS Simulations Start")

#     run2 = MCTS_RUN(2, s2config, s2benchLst, logic, value_type, log_folder, tmp_folder=tmp_folder)
#     run2.start()
#     best_s2 = run2.getBestStrat()
#     finalStratPath = os.path.join(log_folder, 'final_strategy.txt')
#     with open(finalStratPath, 'w') as f:
#        f.write(best_s2)
#     log.info(f"Final Strategy saved to: {finalStratPath}")

#     s2endTime = time.time()
#     log.info(f"Stage 2 MCTS Time: {s2endTime - s2caching_end_time:.0f}")

#     log.info(f"Total Time: {s2endTime - s1startTime:.0f}, with stage1 {s2startTime - s1startTime:.0f}, stage2 caching {s2caching_end_time - s2startTime:.0f}, stage2 MCTS {s2endTime - s2caching_end_time:.0f}")


# if __name__ == "__main__":
#     main()
