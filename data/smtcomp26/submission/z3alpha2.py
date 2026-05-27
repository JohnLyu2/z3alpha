#!/usr/bin/env python3
"""
z3-alpha2 submission entry point for SMT-COMP 2026.

Detects the logic of the input benchmark, loads the corresponding trained
SMT-Select algorithm selector, selects the best Z3 strategy, and invokes Z3.

Usage (as called by SMT-COMP infrastructure):
    ./z3alpha2.py <benchmark.smt2>
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parent
Z3             = SUBMISSION_DIR / "bin" / "z3"
SELECTORS_DIR  = SUBMISSION_DIR / "selectors"

DEBUG = False


def dbg(msg: str) -> None:
    if DEBUG:
        print(f"[z3alpha2] {msg}", file=sys.stderr)


sys.path.insert(0, str(SUBMISSION_DIR / "vendor"))
sys.path.insert(0, str(SUBMISSION_DIR / "lib"))

# Map from SMT-LIB logic name -> selector group directory
LOGIC_TO_GROUP: dict[str, str] = {
    "QF_DT":   "QF_Datatypes",
    "QF_UFDT": "QF_Datatypes",
}


def detect_logic(benchmark: Path) -> str | None:
    """Scan the first 4KB of the benchmark for (set-logic ...)."""
    try:
        header = benchmark.read_bytes()[:4096].decode(errors="ignore")
        m = re.search(r'\(\s*set-logic\s+(\S+)\s*\)', header)
        return m.group(1) if m else None
    except OSError:
        return None


def run_z3(z3_params: list[str], benchmark: Path, timeout: float | None = None) -> int:
    """Run Z3 and return its exit code.

    stdout/stderr are inherited so SMT-COMP sees Z3's output directly.
    timeout (seconds) raises subprocess.TimeoutExpired if Z3 does not finish in time.
    """
    result = subprocess.run(
        [str(Z3)] + z3_params + [str(benchmark)],
        timeout=timeout,
    )
    return result.returncode


def solve(z3_params: list[str], benchmark: Path) -> None:
    """Run the solving schedule and exit with Z3's exit code.

    Currently a single-strategy run.
    """
    rc = run_z3(z3_params, benchmark)
    sys.exit(rc)


def main():
    global DEBUG

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("benchmark")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    DEBUG = args.debug

    benchmark = Path(args.benchmark)
    if not benchmark.is_file():
        dbg(f"Benchmark not found: {benchmark}")
        sys.exit(1)

    logic = detect_logic(benchmark)
    group = LOGIC_TO_GROUP.get(logic) if logic else None

    if group:
        selector_dir = SELECTORS_DIR / group
        try:
            from smt_select import PairwiseSelector
            selector = PairwiseSelector.load(selector_dir / "selector.joblib")
            meta     = json.loads((selector_dir / "meta.json").read_text())
            label    = selector.select(benchmark)
            z3_params = meta["strategy_cli"].get(label, [])
            dbg(f"logic={logic} label={label} params={z3_params}")
        except Exception as e:
            dbg(f"selector failed ({e}), using plain Z3")
            z3_params = []
    else:
        dbg(f"no selector for logic {logic!r}, using plain Z3")
        z3_params = []

    solve(z3_params, benchmark)


if __name__ == "__main__":
    main()
