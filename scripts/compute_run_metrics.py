#!/usr/bin/env python3
"""Compute or backfill stage-1 run metrics from a synthesis output folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from z3alpha.experiment_metrics import (  # noqa: E402
    append_run_metrics_row,
    compute_run_metrics,
    llm_calls_from_qa_log,
    res_database_from_per_benchmark_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute coverage and best-single PAR metrics from a synthesis run directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run folder containing linear_strategy_per_benchmark.csv",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        required=True,
        help="Per-instance timeout used for the run (required for PAR metrics)",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=None,
        help="MCTS sim count (required with --append)",
    )
    parser.add_argument(
        "--wall-time",
        type=float,
        default=None,
        dest="wall_time",
        help="Wall time in seconds (required with --append)",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("experiments/run_metrics.csv"),
        help="Ledger CSV path (default: experiments/run_metrics.csv)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append a row to the metrics ledger CSV",
    )
    args = parser.parse_args()

    per_bench = args.run_dir / "linear_strategy_per_benchmark.csv"
    res_database = res_database_from_per_benchmark_csv(per_bench)
    metrics = compute_run_metrics(res_database, args.timeout)
    llm_calls = llm_calls_from_qa_log(args.run_dir / "llm_prior_qa.log")

    out = {
        "run_name": args.run_dir.name,
        "sims": args.sims,
        **metrics,
        "wall_time_s": args.wall_time,
        "llm_calls": llm_calls,
    }

    if args.append:
        if args.sims is None or args.wall_time is None:
            raise SystemExit("--append requires --sims and --wall-time")
        append_run_metrics_row(
            args.metrics_csv,
            {
                "run_name": args.run_dir.name,
                "sims": args.sims,
                "num_strategies": metrics["num_strategies"],
                "union": metrics["union"],
                "best_single": metrics["best_single"],
                "union_gap": metrics["union_gap"],
                "best_single_par2": metrics["best_single_par2"],
                "best_single_par10": metrics["best_single_par10"],
                "k_union": metrics["k_union"],
                "wall_time_s": int(args.wall_time),
                "llm_calls": llm_calls,
            },
        )
        print(f"Appended row to {args.metrics_csv}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
