#!/usr/bin/env python3
"""
z3-alpha2 submission entry point for SMT-COMP 2026.

Detects the logic of the input benchmark, loads the corresponding trained
SMT-Select algorithm selector, selects the best Z3 strategy, and invokes Z3.

Usage (as called by SMT-COMP infrastructure):
    ./z3alpha2.py <benchmark.smt2>
"""

import json
import re
import sys
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parent
Z3             = SUBMISSION_DIR / "bin" / "z3"
SELECTORS_DIR  = SUBMISSION_DIR / "selectors"

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


def run_z3(z3_params: list[str], benchmark: Path) -> None:
    """Exec Z3 — replaces this process so SMT-COMP sees Z3's exit code."""
    import os
    os.execv(str(Z3), [str(Z3)] + z3_params + [str(benchmark)])


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: z3alpha2.py <benchmark.smt2>")

    benchmark = Path(sys.argv[1])
    if not benchmark.is_file():
        sys.exit(f"Benchmark not found: {benchmark}")

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
        except Exception as e:
            sys.stderr.write(f"; selector failed ({e}), using plain Z3\n")
            z3_params = []
    else:
        sys.stderr.write(f"; no selector for logic {logic!r}, using plain Z3\n")
        z3_params = []

    run_z3(z3_params, benchmark)


if __name__ == "__main__":
    main()
