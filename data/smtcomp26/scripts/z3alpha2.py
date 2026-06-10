#!/usr/bin/env python3
"""
z3-alpha2 submission entry point for SMT-COMP 2026.

Detects the logic of the input benchmark, loads the corresponding trained
SMT-Select algorithm selector, selects the best Z3 strategy, and invokes Z3.

Selector routing and runtime options live in selectors/<group>/meta.json
(logic, presolver_seconds, scheduler, scheduler_timeout, strategy_cli).

Logics with fixed_strategy in meta.json run that Z3 tactic directly (no
presolver, no trained selector). Example: LIA → (then qe_rec smt).

Usage (as called by SMT-COMP infrastructure):
    ./z3alpha2.py <benchmark.smt2>
"""

import argparse
import json
import re
import subprocess
import sys
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


def load_selector_registry(selectors_dir: Path) -> dict[str, tuple[Path, dict]]:
    """Map SMT-LIB logic name -> (selector_dir, meta.json contents).

    Entries need selector.joblib (ML routing) or fixed_strategy (single tactic).
    """
    registry: dict[str, tuple[Path, dict]] = {}
    if not selectors_dir.is_dir():
        return registry

    for group_dir in sorted(selectors_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        meta_path = group_dir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        logic = meta.get("logic")
        if not logic:
            dbg(f"skipping {group_dir.name}: meta.json missing 'logic'")
            continue
        has_selector = (group_dir / "selector.joblib").is_file()
        has_fixed = "fixed_strategy" in meta
        if not has_selector and not has_fixed:
            dbg(f"skipping {group_dir.name}: need selector.joblib or fixed_strategy")
            continue
        if logic in registry:
            dbg(f"warning: duplicate logic {logic!r} ({group_dir.name})")
        registry[logic] = (group_dir, meta)

    return registry


def validate_selector_meta(selector, meta: dict) -> None:
    """Log meta/selector mismatches in debug mode only."""
    strategy_cli = meta.get("strategy_cli", {})
    missing = set(selector.strategies) - set(strategy_cli)
    if missing:
        dbg(f"strategy_cli missing entries for: {sorted(missing)}")

    shortlist = meta.get("shortlist")
    if shortlist is not None and set(shortlist) != set(selector.strategies):
        dbg(
            "shortlist does not match selector.strategies: "
            f"shortlist={shortlist!r}, selector={selector.strategies!r}"
        )


def detect_logic(benchmark: Path) -> str | None:
    """Scan the first 4KB of the benchmark for (set-logic ...)."""
    try:
        header = benchmark.read_bytes()[:4096].decode(errors="ignore")
        m = re.search(r'\(\s*set-logic\s+(\S+)\s*\)', header)
        return m.group(1) if m else None
    except OSError:
        return None


def run_z3(
    z3_params: list[str],
    benchmark: Path,
    timeout: float | None = None,
    *,
    capture: bool = False,
) -> bool | None:
    """Run Z3 on benchmark.

    capture=False: stdout/stderr inherited (returns None).
    capture=True: capture stdout; print and return True on sat/unsat, else False.
    """
    cmd = [str(Z3)] + z3_params + [str(benchmark)]
    try:
        if capture:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
            first_line = result.stdout.strip().split("\n")[0] if result.stdout.strip() else ""
            if first_line in ("sat", "unsat"):
                print(first_line)
                return True
            return False
        subprocess.run(cmd, timeout=timeout)
        return None
    except subprocess.TimeoutExpired:
        dbg("Z3 timed out")
        return False if capture else None


def try_presolver(benchmark: Path, timeout: float) -> bool:
    """Run default Z3 for timeout seconds before feature extraction.

    Returns True (and prints the result) if sat/unsat is produced.
    Returns False silently on timeout or unknown.
    """
    if run_z3([], benchmark, timeout=timeout, capture=True):
        dbg("presolver solved")
        return True
    dbg("presolver timed out or unknown")
    return False


def solve(
    ranked: list[str],
    strategy_cli: dict,
    use_scheduler: bool,
    scheduler_timeout: float,
    benchmark: Path,
) -> None:
    """Run strategies in ranked order, exiting on sat/unsat.

    rank-1 is killed after scheduler_timeout if use_scheduler=True; otherwise
    all strategies run until they exit naturally. Falls back to default Z3 if
    all strategies exit without producing sat/unsat.
    """
    for i, label in enumerate(ranked):
        params  = strategy_cli.get(label, [])
        timeout = scheduler_timeout if (use_scheduler and i == 0) else None
        dbg(f"rank-{i+1}: {label}  timeout={timeout}")
        if run_z3(params, benchmark, timeout=timeout, capture=True):
            sys.exit(0)

    dbg("all strategies exhausted, falling back to default Z3")
    run_z3([], benchmark)
    sys.exit(0)


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
    registry = load_selector_registry(SELECTORS_DIR)
    entry = registry.get(logic) if logic else None

    if entry:
        selector_dir, meta = entry
        fixed_strategy = meta.get("fixed_strategy")
        if fixed_strategy is not None:
            dbg(f"logic={logic} fixed_strategy={fixed_strategy}")
            run_z3(fixed_strategy, benchmark)
            sys.exit(0)

        presolver_seconds = float(meta.get("presolver_seconds") or 0)
        if presolver_seconds > 0:
            if try_presolver(benchmark, presolver_seconds):
                sys.exit(0)

        try:
            from smt_select import PairwiseSelector, bench_feature_vector
            selector          = PairwiseSelector.load(selector_dir / "selector.joblib")
            validate_selector_meta(selector, meta)
            raw               = bench_feature_vector(benchmark)
            if raw is None:
                dbg("feature extraction failed, using fallback ordering")
                ranked = list(selector.strategies)
                ranked.insert(0, ranked.pop(selector.fallback_idx))
            else:
                ranked = selector.rank_all_vec(raw)
            strategy_cli      = meta.get("strategy_cli", {})
            use_scheduler     = meta.get("scheduler", False)
            scheduler_timeout = float(meta.get("scheduler_timeout", 960.0))
            dbg(f"logic={logic} ranked={ranked} scheduler={use_scheduler}")
            solve(ranked, strategy_cli, use_scheduler, scheduler_timeout, benchmark)
        except Exception as e:
            dbg(f"selector failed ({e}), using plain Z3")
            run_z3([], benchmark)
    else:
        dbg(f"no selector for logic {logic!r}, using plain Z3")
        run_z3([], benchmark)


if __name__ == "__main__":
    main()
