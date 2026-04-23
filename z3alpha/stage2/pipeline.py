from z3 import Goal, Probe, parse_smt2_file

from z3alpha.stage2.strategy_tree import PERCENTILES, Stage2Context
from z3alpha.stage2.utils import encode_linear_strategies
from z3alpha.utils import calculate_percentile


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
        percentile: calculate_percentile(num_consts_lst, percentile)
        for percentile in PERCENTILES
    }
    probe_stats["num-exprs"] = {
        percentile: calculate_percentile(num_exprs_lst, percentile)
        for percentile in PERCENTILES
    }
    probe_stats["size"] = {
        percentile: calculate_percentile(size_lst, percentile)
        for percentile in PERCENTILES
    }
    return probe_stats, probe_records


def build_stage2_context(results, bench_list, num_strategies):
    results = results[:num_strategies]
    result_by_strategy = {strat: res_lst for strat, res_lst in results}
    selected_strategies = list(result_by_strategy.keys())
    action_lists, solver_actions, preprocess_actions, strategy_to_actions = (
        encode_linear_strategies(selected_strategies)
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
