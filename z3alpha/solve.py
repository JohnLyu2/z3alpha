"""
Inference entry point: select a Z3 tactic strategy for a benchmark using a
trained PWC selector, then optionally run Z3 with that strategy.

Programmatic use:
    from z3alpha.solve import select_strategy, run_z3
    strategy = select_strategy("selector.pkl", "benchmark.smt2")
    result, elapsed = run_z3("z3", "benchmark.smt2", strategy, timeout=30)

CLI use:
    # Print selected strategy only
    python -m z3alpha.solve selector.pkl benchmark.smt2 --strategy-only

    # Select and run Z3
    python -m z3alpha.solve selector.pkl benchmark.smt2 --timeout 30 --z3 /path/to/z3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from z3alpha.evaluator import SolverRunner
from z3alpha.ml_selector import PwcSelector


def select_strategy(selector_path: str | Path, benchmark_path: str | Path) -> str:
    """Load a trained selector and return the predicted strategy for a benchmark."""
    selector = PwcSelector.load(selector_path)
    return selector.select(benchmark_path)


def run_z3(
    z3_path: str,
    benchmark_path: str | Path,
    strategy: str,
    timeout: Optional[float] = None,
) -> tuple[str, float]:
    """Run Z3 with the given strategy on a benchmark.

    Returns:
        (result, elapsed_seconds) where result is the solver verdict
        ("sat", "unsat", "unknown", "timeout", or "error").
    """
    runner = SolverRunner(
        solver_path=z3_path,
        smt_file=str(benchmark_path),
        timeout=timeout,
        run_id=0,
        z3_strategy=strategy,
    )
    _, result, elapsed, _ = runner.execute()
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Select a Z3 strategy via PWC selector and optionally run Z3"
    )
    parser.add_argument("selector", type=str, help="Path to trained selector .pkl file")
    parser.add_argument("benchmark", type=str, help="Path to .smt2 benchmark file")
    parser.add_argument(
        "--strategy-only",
        action="store_true",
        help="Print the selected strategy and exit without running Z3",
    )
    parser.add_argument(
        "--z3",
        type=str,
        default="z3",
        help="Path to Z3 binary (default: z3)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Solver timeout in seconds (default: no limit)",
    )
    args = parser.parse_args()

    strategy = select_strategy(args.selector, args.benchmark)

    if args.strategy_only:
        print(strategy)
        return

    try:
        result, elapsed = run_z3(args.z3, args.benchmark, strategy, timeout=args.timeout)
        print(result)
        sys.stderr.write(f"; time: {elapsed:.3f}s\n")
    except FileNotFoundError:
        print(f"Error: Z3 binary not found at '{args.z3}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
