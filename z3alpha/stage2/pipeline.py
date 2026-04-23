import logging
import time
from pathlib import Path

from z3 import Goal, Probe, parse_smt2_file

from z3alpha.evaluator import SolverEvaluator
from z3alpha.stage2.actions import convert_strats_to_act_lists
from z3alpha.stage2.context import Stage2Context
from z3alpha.stage2.tree import PERCENTILES
from z3alpha.utils import calculatePercentile, write_strat_res_to_csv

log = logging.getLogger(__name__)


def create_probe_stats(bench_lst):
    num_consts_lst = []
    num_exprs_lst = []
    size_lst = []
    probe_records = []
    for smt_path in bench_lst:
        instance_dict = {}
        formula = parse_smt2_file(smt_path)
        const_probe = Probe("num-consts")
        expr_probe = Probe("num-exprs")
        size_probe = Probe("size")
        goal = Goal()
        goal.add(formula)
        num_consts = const_probe(goal)
        num_consts_lst.append(num_consts)
        instance_dict["num-consts"] = num_consts
        num_exprs = expr_probe(goal)
        num_exprs_lst.append(num_exprs)
        instance_dict["num-exprs"] = num_exprs
        size = size_probe(goal)
        size_lst.append(size)
        instance_dict["size"] = size
        probe_records.append(instance_dict)
    probe_stats = {}
    probe_stats["num-consts"] = {
        percentile: calculatePercentile(num_consts_lst, percentile)
        for percentile in PERCENTILES
    }
    probe_stats["num-exprs"] = {
        percentile: calculatePercentile(num_exprs_lst, percentile)
        for percentile in PERCENTILES
    }
    probe_stats["size"] = {
        percentile: calculatePercentile(size_lst, percentile)
        for percentile in PERCENTILES
    }
    return probe_stats, probe_records


def cache_stage2_candidates(selected_strategies, config, log_folder, benchmark_list=None):
    start_time = time.time()
    num_strat = config["ln_strat_num"]
    selected_strategies = selected_strategies[:num_strat]
    z3path = config["z3path"] if "z3path" in config else "z3"
    s2config = config["s2config"]
    s2_bench_dirs = s2config["bench_dirs"]
    s2timeout = s2config["timeout"]
    batch_size = config["batch_size"]
    bench_list = (
        benchmark_list
        if benchmark_list
        else _create_benchmark_list(s2_bench_dirs)
    )
    s2_res_list = []
    s2evaluator = SolverEvaluator(z3path, bench_list, s2timeout, batch_size)
    for i, strategy in enumerate(selected_strategies):
        log.info(f"Stage 2 Caching: {i + 1}/{len(selected_strategies)}")
        strategy_res = s2evaluator.getResLst(strategy)
        s2_res_list.append((strategy, strategy_res))
    cache_csv = Path(log_folder) / "stage2_strategy_cache.csv"
    write_strat_res_to_csv(s2_res_list, cache_csv, bench_list)
    log.info(f"Cached results saved to {cache_csv}")
    elapsed = time.time() - start_time
    log.info(f"Stage 2 Cache Time: {elapsed:.0f}")
    return s2_res_list, bench_list


def build_stage2_context(results, bench_list, num_strategies):
    results = results[:num_strategies]
    result_by_strategy = {strat: res_lst for strat, res_lst in results}
    selected_strategies = list(result_by_strategy.keys())
    action_lists, solver_actions, preprocess_actions, strategy_to_actions = (
        convert_strats_to_act_lists(selected_strategies)
    )
    result_cache = {
        strategy_to_actions[strategy]: result_by_strategy[strategy]
        for strategy in strategy_to_actions
    }
    probe_stats, probe_records = create_probe_stats(bench_list)
    return Stage2Context(
        seed_action_sequences=action_lists,
        solver_actions=solver_actions,
        preprocess_actions=preprocess_actions,
        result_cache=result_cache,
        probe_stats=probe_stats,
        probe_records=probe_records,
    )


def _create_benchmark_list(benchmark_directories):
    benchmark_lst = []
    for bench_dir in benchmark_directories:
        assert Path(bench_dir).exists()
        benchmark_lst += [str(p) for p in sorted(list(Path(bench_dir).rglob("*.smt2")))]
    benchmark_lst.sort()
    return benchmark_lst
