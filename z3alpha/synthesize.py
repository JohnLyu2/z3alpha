import time
import random
import csv
from pathlib import Path
import argparse
import logging
import json
import datetime

from z3alpha.logging_config import setup_logging
from z3alpha.mcts import LinearStrategySearchRun
from z3alpha.stage2.search_runtime import Stage2MCTSRun
from z3alpha.strategy_portfolio import create_greedy_linear_strategy_portfolio
from z3alpha.stage2.pipeline import build_stage2_context
from z3alpha.synthesis_config import merge_synthesis_config
from z3alpha.tactics.logic_config import load_logic_config

log = logging.getLogger(__name__)

VALUE_TYPE = "par10"  # hard code for now


def _train_dir(config) -> str:
    """Top-level config ``train_dir``: one root directory (scanned recursively for ``*.smt2``)."""
    raw = config["train_dir"]
    if not isinstance(raw, str):
        raise TypeError("train_dir must be a string (path to a directory)")
    return raw


def create_benchmark_list(benchmark_directories):
    """Collect all .smt2 files from the given directories."""
    benchmark_lst = []
    for dir in benchmark_directories:
        assert Path(dir).exists()
        benchmark_lst += [
            str(p) for p in sorted(list(Path(dir).rglob("*.smt2")))
        ]
    benchmark_lst.sort()
    return benchmark_lst


def _log_elapsed(start_time, label):
    elapsed = time.time() - start_time
    log.info(f"{label}: {elapsed:.0f}")
    return elapsed


def _linear_mcts_run_config(config: dict) -> dict:
    """Config dict for ``LinearStrategySearchRun`` (expects ``sim_num`` / ``c_ucb`` / UCT / timeout)."""
    m = config["mcts_config"]
    return {
        "sim_num": config["mcts_sims"],
        "timeout": config["timeout"],
        "c_uct": m["c_uct"],
        "c_ucb": m["c_ucb"],
    }


def _stage2_mcts_run_config(config: dict) -> dict:
    """Config dict for ``Stage2MCTSRun`` (same ``timeout`` and ``c_uct`` as linear search)."""
    m = config["mcts_config"]
    return {
        "sim_num": config["s2_sims"],
        "timeout": config["timeout"],
        "c_uct": m["c_uct"],
    }


def synthesize_linear_strategies(config, log_folder):
    start_time = time.time()
    logic = config["logic"]
    z3path = config["z3path"] if "z3path" in config else "z3"
    batch_size = config["batch_size"]
    linear_run_config = _linear_mcts_run_config(config)
    num_ln_strat = config["ln_strat_num"]
    random_seed = config["random_seed"]
    random.seed(random_seed)

    config_dir = config.get("logic_config_dir", None)
    logic_config = load_logic_config(logic, config_dir)

    # Linear strategy search
    s1_bench_lst = create_benchmark_list([_train_dir(config)])
    log.info("Linear strategy search starts")
    run1 = LinearStrategySearchRun(
        linear_run_config,
        s1_bench_lst,
        logic,
        z3path,
        VALUE_TYPE,
        log_folder,
        batch_size=batch_size,
        logic_config=logic_config,
    )
    run1.start()
    s1_res_dict = run1.get_res_dict()

    selected_strat, ln_select_logs = create_greedy_linear_strategy_portfolio(
        num_ln_strat, s1_res_dict, config["timeout"]
    )
    log.info(ln_select_logs)
    ln_strat_candidates_path = Path(log_folder) / "linear_selected_strategies.csv"
    with open(ln_strat_candidates_path, "w") as f:
        # write one strategy per line as a csv file
        cwriter = csv.writer(f)
        # header (one column "strat")
        cwriter.writerow(["strat"])
        for strat in selected_strat:
            cwriter.writerow([strat])
    log.info(
        f"Selected {len(selected_strat)} strategies: {selected_strat}, saved to {ln_strat_candidates_path}"
    )

    _log_elapsed(start_time, "Linear strategy search time")
    stage1_results_for_s2 = [(s, s1_res_dict[s]) for s in selected_strat]
    return selected_strat, s1_bench_lst, stage1_results_for_s2

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

def stage2_synthesize(results, bench_lst, config, log_folder):
    num_strat = config["ln_strat_num"]
    stage2_context = build_stage2_context(results, bench_lst, num_strat)
    log.info(f"preprocess dict: {stage2_context.preprocess_actions}")
    log.info(f"solver dict: {stage2_context.solver_actions}")
    log.info(
        f"converted selected strategies: {stage2_context.seed_action_sequences}"
    )

    logic = config["logic"]
    z3path = "z3"
    if "z3path" in config:
        z3path = config["z3path"]

    s2startTime = time.time()
    log.info("S2 MCTS Simulations Start")

    run2_config = {**_stage2_mcts_run_config(config), "stage2_context": stage2_context}

    run_stage_two = Stage2MCTSRun(
        run2_config,
        bench_lst,
        logic,
        z3path,
        VALUE_TYPE,
        log_folder,
    )
    run_stage_two.start()
    best_strategy = run_stage_two.get_best_strat()

    strat_path = Path(log_folder) / "synthesized_strategy.txt"
    with open(strat_path, "w") as f:
        f.write(best_strategy)
    log.info(f"Best final strategy saved to: {strat_path}")

    _log_elapsed(s2startTime, "Stage 2 MCTS Time")
    return best_strategy


def parallel_synthesize(config, log_folder):
    """
    Perform parallel strategy synthesis.

    Args:
        config: Configuration dictionary containing synthesis parameters
        log_folder: Log folder path for output files
    """
    start_time = time.time()

    selected_strats, _, _ = synthesize_linear_strategies(config, log_folder)

    parallel_strat = parallel_linear_strategies(selected_strats)

    parallel_strat_path = Path(log_folder) / "synthesized_strategy.txt"
    with open(parallel_strat_path, "w") as f:
        f.write(parallel_strat)
    log.info(f"Final parallel strategy saved to {parallel_strat_path}")

    _log_elapsed(start_time, "Total synthesis time")



def branched_synthesize(config, log_folder):
    """
    Perform the complete synthesis for branched strategy [IJCAI 2024].

    Args:
        config: Configuration dictionary containing synthesis parameters
        log_folder: Log folder path for output files
    """
    start_time = time.time()

    # Step 1: Linear strategy synthesis (Stage 2 uses its per-benchmark results)
    _, s1_bench_lst, stage1_results_for_s2 = synthesize_linear_strategies(
        config, log_folder
    )

    # Step 2: Branched strategy synthesis
    stage2_synthesize(stage1_results_for_s2, s1_bench_lst, config, log_folder)

    _log_elapsed(start_time, "Total synthesis time")

def _apply_synthesis_cli_overrides(config, args) -> None:
    """Apply argparse overrides onto merged config (mutates *config*)."""
    m = config.setdefault("mcts_config", {})
    if args.c_uct is not None:
        m["c_uct"] = args.c_uct
    if args.c_ucb is not None:
        m["c_ucb"] = args.c_ucb
    if args.random_seed is not None:
        config["random_seed"] = args.random_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_config", type=str, help="The experiment configuration file in json"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Synthesize parallel strategy instead of branched strategy"
    )
    parser.add_argument(
        "--c-uct",
        type=float,
        default=None,
        help="Override MCTS UCT constant (default from z3alpha/synthesis_defaults.json)",
    )
    parser.add_argument(
        "--c-ucb",
        type=float,
        default=None,
        help="Override UCB1 constant for linear strategy search (default from synthesis_defaults.json)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        dest="random_seed",
        help="Override random seed (default from synthesis_defaults.json)",
    )
    args = parser.parse_args()
    with open(args.json_config, encoding="utf-8") as f:
        user = json.load(f)
    config = merge_synthesis_config(user)
    _apply_synthesis_cli_overrides(config, args)

    level = config.get("log_level", "INFO")
    setup_logging(level=level)

    # Use log_parent_dir if provided, otherwise use default experiments/synthesis
    parent_dir = Path(config.get("parent_log_dir", "experiments/synthesis"))
    log_folder = parent_dir / f"out-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
    
    assert not log_folder.exists()
    log_folder.mkdir(parents=True)
    

    if args.parallel:
        parallel_synthesize(config, log_folder)
    else:
        branched_synthesize(config, log_folder)


if __name__ == "__main__":
    main()
