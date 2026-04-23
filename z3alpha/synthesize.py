import time
import random
import csv
from pathlib import Path
import argparse
import logging
import json
import datetime

from z3alpha.logging_config import setup_logging
from z3alpha.mcts import LinearStrategySearchRun, MCTSSearchConfig
from z3alpha.stage2.search_runtime import run_branched_synthesis
from z3alpha.strategy_portfolio import create_greedy_linear_strategy_portfolio
from z3alpha.synthesis_config import (
    ExperimentConfig,
    MCTSParams,
    SynthesisRun,
    parse_experiment_config,
    resolve_mcts_params,
)
from z3alpha.tactics.logic_config import load_logic_config

log = logging.getLogger(__name__)


def _mcts_config_linear(e: ExperimentConfig, m: MCTSParams) -> MCTSSearchConfig:
    return MCTSSearchConfig(
        sim_num=e.mcts_sims,
        timeout=e.timeout,
        c_uct=m.c_uct,
        c_ucb=m.c_ucb,
    )


def _train_dir(experiment: ExperimentConfig) -> str:
    raw = experiment.train_dir
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


def synthesize_linear_strategies(run: SynthesisRun, log_folder: Path):
    experiment, m = run.experiment, run.m
    start_time = time.time()
    logic = experiment.logic
    z3path = experiment.z3path if experiment.z3path else "z3"
    batch_size = experiment.batch_size
    search = _mcts_config_linear(experiment, m)
    num_ln_strat = experiment.ln_strat_num
    value_type = experiment.value_type
    random.seed(m.random_seed)

    config_dir = experiment.logic_config_dir
    logic_config = load_logic_config(logic, config_dir)

    s1_bench_lst = create_benchmark_list([_train_dir(experiment)])
    log.info("Linear strategy search starts")
    run1 = LinearStrategySearchRun(
        search,
        s1_bench_lst,
        logic,
        z3path,
        value_type,
        log_folder,
        batch_size=batch_size,
        logic_config=logic_config,
    )
    run1.start()
    s1_res_dict = run1.get_res_dict()

    selected_strat, ln_select_logs = create_greedy_linear_strategy_portfolio(
        num_ln_strat, s1_res_dict, experiment.timeout
    )
    log.info(ln_select_logs)
    ln_strat_candidates_path = Path(log_folder) / "linear_selected_strategies.csv"
    with open(ln_strat_candidates_path, "w") as f:
        cwriter = csv.writer(f)
        cwriter.writerow(["strat"])
        for strat in selected_strat:
            cwriter.writerow([strat])
    log.info(
        f"Selected {len(selected_strat)} strategies: {selected_strat}, saved to {ln_strat_candidates_path}"
    )

    _log_elapsed(start_time, "Linear strategy search time")
    shortlist = [(s, s1_res_dict[s]) for s in selected_strat]
    return selected_strat, s1_bench_lst, shortlist


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


def parallel_synthesize(run: SynthesisRun, log_folder: Path):
    start_time = time.time()

    selected_strats, _, _ = synthesize_linear_strategies(run, log_folder)

    parallel_strat = parallel_linear_strategies(selected_strats)

    parallel_strat_path = Path(log_folder) / "synthesized_strategy.txt"
    with open(parallel_strat_path, "w") as f:
        f.write(parallel_strat)
    log.info(f"Final parallel strategy saved to {parallel_strat_path}")

    _log_elapsed(start_time, "Total synthesis time")


def branched_synthesize(run: SynthesisRun, log_folder: Path):
    start_time = time.time()

    _, bench_lst, shortlist = synthesize_linear_strategies(
        run, log_folder
    )

    run_branched_synthesis(run, shortlist, bench_lst, log_folder)

    _log_elapsed(start_time, "Total synthesis time")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_config", type=str, help="The experiment configuration file in json"
    )
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Synthesize parallel strategy instead of branched strategy",
    )
    parser.add_argument(
        "--c-uct",
        type=float,
        default=None,
        help="Override MCTS UCT (default: z3alpha.synthesis_config.DEFAULT_C_UCT)",
    )
    parser.add_argument(
        "--c-ucb",
        type=float,
        default=None,
        help="Override UCB1 for linear search (default: z3alpha.synthesis_config.DEFAULT_C_UCB)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        dest="random_seed",
        help="Override random seed (default: z3alpha.synthesis_config.DEFAULT_RANDOM_SEED)",
    )
    args = parser.parse_args()
    with open(args.json_config, encoding="utf-8") as f:
        user = json.load(f)
    experiment = parse_experiment_config(user)
    m = resolve_mcts_params(args)
    run = SynthesisRun(experiment=experiment, m=m)

    level = run.experiment.log_level or "INFO"
    setup_logging(level=level)

    parent_dir = Path(
        run.experiment.parent_log_dir
        if run.experiment.parent_log_dir
        else "experiments/synthesis"
    )
    log_folder = parent_dir / f"out-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"

    assert not log_folder.exists()
    log_folder.mkdir(parents=True)

    if args.parallel:
        parallel_synthesize(run, log_folder)
    else:
        branched_synthesize(run, log_folder)


if __name__ == "__main__":
    main()
