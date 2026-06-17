import time
import random
import csv
from pathlib import Path
import argparse
import logging
import json
import datetime

from z3alpha.config import (
    ExperimentConfig,
    SynthesisRun,
    parse_experiment_config,
    resolve_mcts_config,
    setup_logging,
)
from z3alpha.config.env import check_z3_version, load_env_config
from z3alpha.mcts import LinearStrategySearchRun
from z3alpha.smt_select import train_pwc_selector
from z3alpha.stage2.search_runtime import run_branched_synthesis
from z3alpha.strategy_portfolio import create_greedy_linear_strategy_portfolio
from z3alpha.tactics.logic_config import load_logic_config
from z3alpha.utils import create_benchmark_list

log = logging.getLogger(__name__)


def _train_dir(experiment: ExperimentConfig) -> str:
    raw = experiment.train_dir
    if not isinstance(raw, str):
        raise TypeError("train_dir must be a string (path to a directory)")
    return raw


def _log_elapsed(start_time, label):
    elapsed = time.time() - start_time
    log.info(f"{label}: {elapsed:.0f}")
    return elapsed


def synthesize_linear_strategies(run: SynthesisRun, log_folder: Path, env=None):
    if env is None:
        env = load_env_config()
    experiment = run.experiment
    start_time = time.time()
    logic = experiment.logic
    z3path = experiment.z3path if experiment.z3path else (env.z3_path or "z3")
    batch_size = env.workers
    num_ln_strat = experiment.ln_strat_num
    value_type = experiment.value_type
    random.seed(run.mcts.random_seed)

    config_dir = experiment.logic_config_dir
    logic_config = load_logic_config(logic, config_dir)

    s1_bench_lst = create_benchmark_list([_train_dir(experiment)])
    log.info("Linear strategy search starts")
    run1 = LinearStrategySearchRun(
        run.mcts,
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
    return s1_bench_lst, shortlist


def ml_synthesize(run: SynthesisRun, log_folder: Path, env=None):
    if env is None:
        env = load_env_config()
    start_time = time.time()

    bench_lst, shortlist = synthesize_linear_strategies(run, log_folder, env=env)

    selector = train_pwc_selector(
        shortlist=shortlist,
        bench_paths=bench_lst,
        timeout=run.experiment.timeout,
        random_seed=run.mcts.random_seed,
    )

    selector_path = Path(log_folder) / "selector.pkl"
    selector.save(selector_path)
    log.info(f"PWC selector saved to {selector_path}")

    _log_elapsed(start_time, "Total synthesis time")


def branched_synthesize(run: SynthesisRun, log_folder: Path, env=None):
    if env is None:
        env = load_env_config()
    start_time = time.time()

    bench_lst, shortlist = synthesize_linear_strategies(run, log_folder, env=env)

    run_branched_synthesis(run, shortlist, bench_lst, log_folder)

    _log_elapsed(start_time, "Total synthesis time")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "json_config", type=str, help="The experiment configuration file in json"
    )
    parser.add_argument(
        "--c-uct",
        type=float,
        default=None,
        help="Override MCTS PUCT c (tactic tree; default: z3alpha.config.synthesis.DEFAULT_C_UCT)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        dest="random_seed",
        help="Override random seed (default: z3alpha.config.synthesis.DEFAULT_RANDOM_SEED)",
    )
    parser.add_argument(
        "--stage2",
        action="store_true",
        help="Use MCTS branched strategy search (Stage 2) instead of the default PWC ML selector",
    )
    parser.add_argument(
        "--llm-prior",
        action="store_true",
        help="Use OpenAI Chat Completions to score legal tactics as PUCT priors (stage 1 only; needs OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-5.4-mini",
        help="OpenAI model id for --llm-prior (default: gpt-5.4-mini)",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=None,
        help="OpenAI API base URL for --llm-prior (default: https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=None,
        help="LLM HTTP timeout in seconds for --llm-prior (default: 30)",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="Sampling temperature for --llm-prior (default: 0)",
    )
    args = parser.parse_args()
    env = load_env_config()
    check_z3_version(env)

    with open(args.json_config, encoding="utf-8") as f:
        user = json.load(f)
    experiment = parse_experiment_config(user)
    mcts_config = resolve_mcts_config(args, experiment)
    run = SynthesisRun(experiment=experiment, mcts=mcts_config)

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

    if args.stage2:
        branched_synthesize(run, log_folder, env=env)
    else:
        ml_synthesize(run, log_folder, env=env)


if __name__ == "__main__":
    main()
