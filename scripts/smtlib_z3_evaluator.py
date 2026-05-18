import argparse
import csv
import json
import logging
import os
import pathlib
import subprocess
import sys
from datetime import datetime

from z3alpha.config import setup_logging
from z3alpha.evaluator import SolverEvaluator
from z3alpha.utils import par_n, solved_num

log = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = pathlib.Path(__file__).with_name("smtlib_z3_evaluator_config.json")


def _evaluate_and_write(evaluator, strategy_str, csv_path):
    """Run ``evaluator.evaluate`` and dump per-instance results + PAR summary."""
    total = evaluator.get_benchmark_size()
    report_every = min(max(1, total // 20), 50)

    def _progress(completed: int, size: int, solved: int) -> None:
        if completed % report_every == 0 or completed == size:
            pct = (completed * 100.0) / size
            solve_rate = (solved * 100.0) / completed
            log.info(
                "Evaluation progress: %d/%d (%.1f%%) | solved %d (%.1f%% solve rate)",
                completed,
                size,
                pct,
                solved,
                solve_rate,
            )

    results = evaluator.evaluate(strategy_str, progress_callback=_progress)
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "path", "solved", "time", "result"])
        for i, bench in enumerate(evaluator.benchmark_list):
            solved, runtime, result_str = results[i]
            writer.writerow([i, bench, solved, runtime, result_str])
    return (
        solved_num(results),
        par_n(results, 2, evaluator.timeout),
        par_n(results, 10, evaluator.timeout),
    )


def _load_default_config(config_path: pathlib.Path) -> dict:
    if not config_path.is_file():
        sys.exit(f"Default config does not exist: {config_path}")
    try:
        return json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        sys.exit(f"Invalid JSON in config {config_path}: {exc}")


def _resolve_eval_list(eval_list_file: str, smtlib_root: str | None) -> list[str]:
    list_path = pathlib.Path(eval_list_file)
    if not list_path.is_file():
        sys.exit(f"eval_list_file does not exist or is not a file: {eval_list_file}")

    eval_lst = []
    for line in list_path.read_text().splitlines():
        p = line.strip()
        if not p or p.startswith("#"):
            continue
        rel_path = pathlib.Path(p)
        if rel_path.is_absolute():
            path = rel_path
        elif smtlib_root:
            path = pathlib.Path(smtlib_root) / rel_path
        else:
            path = rel_path.resolve()
        if path.is_file():
            eval_lst.append(str(path.resolve()))
        else:
            sys.exit(f"Path in eval_list_file does not exist or is not a file: {p}")

    eval_lst = sorted(eval_lst)
    if not eval_lst:
        sys.exit(f"No valid paths in eval_list_file: {eval_list_file}")
    return eval_lst


def _resolve_eval_dir(eval_dir: str) -> list[str]:
    eval_dir_path = pathlib.Path(eval_dir)
    if not eval_dir_path.is_dir():
        sys.exit(f"eval_dir does not exist or is not a directory: {eval_dir}")
    eval_lst = sorted(str(p.resolve()) for p in eval_dir_path.rglob("*.smt2"))
    if not eval_lst:
        sys.exit(f"No .smt2 files found under eval_dir: {eval_dir}")
    return eval_lst


_CLI_ONLY_CONFIG_KEYS = frozenset(
    {"strategy", "strategy_file", "z3_extra_params"},
)


def _reject_cli_only_config_keys(config: dict) -> None:
    forbidden = sorted(k for k in _CLI_ONLY_CONFIG_KEYS if k in config)
    if forbidden:
        sys.exit(
            "Set via CLI only (not config JSON): "
            + ", ".join(forbidden)
            + ". Use --strategy/--strategy-file and/or --z3-param."
        )


def _check_z3_version(z3_path: str, expected_version: str | None) -> None:
    if expected_version is None:
        return
    expected = expected_version.strip()
    if not expected:
        return

    try:
        proc = subprocess.run(
            [z3_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        sys.exit(f"z3 executable not found: {z3_path}")
    except subprocess.CalledProcessError as exc:
        msg = (exc.stdout or exc.stderr or "").strip()
        sys.exit(f"failed to get z3 version from {z3_path}: {msg}")

    actual = (proc.stdout or proc.stderr).strip().splitlines()
    actual_line = actual[0].strip() if actual else ""
    if expected not in actual_line:
        sys.exit(
            f"z3 version mismatch: expected to contain '{expected}', got '{actual_line}'"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Z3 on SMT-LIB benchmarks using local default config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_DEFAULT_CONFIG_PATH),
        help="Path to local default config JSON (default: scripts/smtlib_z3_evaluator_config.json).",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        help="Directory to recursively search for .smt2 benchmarks.",
    )
    parser.add_argument(
        "--eval-list-file",
        type=str,
        help="Text file containing one benchmark path per line.",
    )
    parser.add_argument("--z3-path", type=str, help="Override Z3 executable path.")
    parser.add_argument(
        "--z3-version",
        type=str,
        help="Override expected Z3 version string from config.",
    )
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument(
        "--timeout",
        type=int,
        required=True,
        help="Timeout in seconds (required CLI argument).",
    )
    parser.add_argument("--res-dir", type=str, help="Override result output directory.")
    strategy_group = parser.add_mutually_exclusive_group(required=False)
    strategy_group.add_argument(
        "--strategy-file",
        type=str,
        help="Path to Z3 strategy file; content passed to tactic.default_tactic.",
    )
    strategy_group.add_argument(
        "--strategy",
        type=str,
        help="Z3 strategy string passed to tactic.default_tactic.",
    )
    parser.add_argument(
        "--z3-param",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "Extra Z3 command-line argument (repeatable), e.g. smt.random_seed=1 "
            "or -v:10."
        ),
    )
    args = parser.parse_args()

    setup_logging()
    config = _load_default_config(pathlib.Path(args.config))
    _reject_cli_only_config_keys(config)

    z3_path = args.z3_path if args.z3_path is not None else config.get("z3_path", "z3")
    z3_version = (
        args.z3_version
        if args.z3_version is not None
        else config.get("z3_version")
    )
    batch_size = (
        args.batch_size if args.batch_size is not None else config.get("batch_size", 4)
    )
    smtlib_root = config.get("smtlib_root")
    z3_extra_params = [p.strip() for p in args.z3_param if p and p.strip()]
    timeout = args.timeout
    res_dir = (
        pathlib.Path(args.res_dir)
        if args.res_dir is not None
        else pathlib.Path(config.get("res_dir", "experiments/evaluation/smtlib_z3"))
    )

    if batch_size <= 0:
        sys.exit("batch_size must be > 0")
    if timeout <= 0:
        sys.exit("timeout must be > 0")

    _check_z3_version(z3_path, z3_version)

    if (args.eval_dir is None) == (args.eval_list_file is None):
        sys.exit("Provide exactly one of --eval-dir or --eval-list-file.")

    eval_lst = (
        _resolve_eval_dir(args.eval_dir)
        if args.eval_dir is not None
        else _resolve_eval_list(args.eval_list_file, smtlib_root)
    )

    if args.strategy_file is not None:
        strategy_path = pathlib.Path(args.strategy_file)
        if not strategy_path.is_file():
            sys.exit(
                f"strategy_file does not exist or is not a file: {args.strategy_file}"
            )
        strategy = strategy_path.read_text()
    elif args.strategy is not None:
        strategy = args.strategy
    else:
        strategy = None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = res_dir / timestamp
    os.makedirs(run_dir, exist_ok=True)

    evaluator = SolverEvaluator(
        z3_path, eval_lst, timeout, batch_size, z3_extra_params=z3_extra_params
    )

    detail_csv = run_dir / "instance_results.csv"
    summary_json = run_dir / "summary.json"
    solved, par2, par10 = _evaluate_and_write(evaluator, strategy, str(detail_csv))
    instance_count = len(eval_lst)
    par2_avg = par2 / instance_count
    par10_avg = par10 / instance_count

    summary_payload = {
        "timestamp": timestamp,
        "environment": {
            "machine_name": config.get("machine_name"),
            "z3_path": z3_path,
            "z3_version": z3_version,
            "batch_size": batch_size,
            "smtlib_root": smtlib_root,
        },
        "run_config": {
            "mode": "eval_dir" if args.eval_dir is not None else "eval_list_file",
            "source": args.eval_dir if args.eval_dir is not None else args.eval_list_file,
            "timeout": timeout,
            "strategy": strategy,
            "z3_extra_params": z3_extra_params,
            "instances_total": instance_count,
        },
        "results": {
            "solved": solved,
            "par2": par2_avg,
            "par10": par10_avg,
            "detail_csv": str(detail_csv),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2))

    log.info(
        "z3 test results: solved %s instances with par2 %.2f and par10 %.2f",
        solved,
        par2,
        par10,
    )
    log.info("Run directory: %s", run_dir)
    log.info("Wrote detail results to %s", detail_csv)
    log.info("Wrote summary results to %s", summary_json)


if __name__ == "__main__":
    main()
