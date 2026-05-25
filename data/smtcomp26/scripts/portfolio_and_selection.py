#!/usr/bin/env python3
"""
Greedy portfolio analysis + PWC algorithm selection for strategy evaluation results.

Loads per-instance CSV results from a strategy_eval directory, runs greedy
portfolio selection with cumulative coverage stats, then prompts for a shortlist
size k and trains a PWC SVM selector on the top-k strategies.

Usage:
    python portfolio_and_selection.py <eval_dir>

Example:
    python portfolio_and_selection.py data/smtcomp26/strategy_eval/QF_Datatypes
"""

import argparse
import csv
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from z3alpha.ml_selector import bench_feature_vector, train_pwc_selector, FEATURE_NAMES
from z3alpha.strategy_portfolio import create_greedy_linear_strategy_portfolio, virtual_add_strategy
from z3alpha.utils import solved_num, par_n

logging.disable(logging.CRITICAL)  # suppress library INFO/WARNING logs

REPO_ROOT      = Path(__file__).resolve().parents[3]
BENCH_LIST_DIR = REPO_ROOT / "data/smtcomp26/smtcomp25_benchlist"
FEATURES_DIR   = REPO_ROOT / "data/smtcomp26/features"

def _load_env_config() -> dict:
    config_path = REPO_ROOT / "env_config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return json.load(f)

_ENV = _load_env_config()
SMTLIB_ROOT = Path(_ENV.get("smtlib_root", "/home/z52lu/smtlib25"))
N_WORKERS   = _ENV.get("workers", 8)

# ─── Feature cache ────────────────────────────────────────────────────────────

def _extract_one(rel_path: str) -> tuple[str, float, object]:
    """Worker: extract features for one benchmark. Returns (rel_path, elapsed, feat_or_None)."""
    t0 = time.perf_counter()
    feat = bench_feature_vector(SMTLIB_ROOT / rel_path)
    return rel_path, time.perf_counter() - t0, feat

def load_or_build_features(bench_paths: list[str]) -> dict[str, np.ndarray]:
    """Return a path -> feature vector dict for bench_paths.

    Features are cached per bench list file under FEATURES_DIR. On first call
    for a given logic, all benchmarks in the bench list are extracted from
    SMTLIB_ROOT and saved; subsequent calls load instantly from the .npz cache.
    """
    bench_set = set(bench_paths)

    # Find which bench list files cover these paths
    relevant: list[tuple[str, list[str]]] = []
    for list_file in sorted(BENCH_LIST_DIR.glob("*.txt")):
        paths_in_list = [p.strip() for p in list_file.read_text().splitlines() if p.strip()]
        if bench_set & set(paths_in_list):
            relevant.append((list_file.stem, paths_in_list))

    if not relevant:
        sys.exit("No bench list files match the eval instances. Check BENCH_LIST_DIR.")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    feature_dict: dict[str, np.ndarray] = {}

    for logic_name, all_paths in relevant:
        cache_file = FEATURES_DIR / f"{logic_name}.csv"

        if cache_file.exists():
            with open(cache_file, newline="") as f:
                reader = csv.reader(f)
                next(reader)  # header: path, extraction_time_s, f0, f1, ...
                cached_paths = []
                for row in reader:
                    p, feat = row[0], np.array(row[2:], dtype=float)
                    cached_paths.append(p)
                    if p in bench_set:
                        feature_dict[p] = feat
            n_hit = sum(1 for p in cached_paths if p in bench_set)
            print(f"  [{logic_name}] loaded from cache  ({n_hit}/{len(cached_paths)} in eval)")
        else:
            print(f"  [{logic_name}] extracting {len(all_paths)} benchmarks ({N_WORKERS} workers) ...", end="", flush=True)
            rows_out = []
            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                for p, elapsed, feat in pool.map(_extract_one, all_paths):
                    if feat is not None:
                        rows_out.append((p, elapsed, feat))
                        if p in bench_set:
                            feature_dict[p] = feat
            with open(cache_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "extraction_time_s"] + FEATURE_NAMES)
                for p, elapsed, feat in rows_out:
                    writer.writerow([p, f"{elapsed:.6f}"] + feat.tolist())
            print(f" {len(rows_out)}/{len(all_paths)} OK  (saved to {cache_file.name})")

    return feature_dict


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_results(eval_dir: Path):
    """Load per-instance results from all strategy subdirs.

    Returns (result_database, bench_paths, timeout) where:
      result_database: label -> list of (solved, time, status)
      bench_paths: ordered list of benchmark file paths
    """
    result_database: dict[str, list] = {}
    bench_paths: list[str] = []
    timeout = 30

    subdirs = sorted(p for p in eval_dir.iterdir() if p.is_dir())
    if not subdirs:
        sys.exit(f"No subdirectories found in {eval_dir}")

    for subdir in subdirs:
        csv_path = subdir / "instance_results.csv"
        summary_path = subdir / "summary.json"
        if not csv_path.exists():
            continue

        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            timeout = summary.get("run_config", {}).get("timeout", timeout)
            label = summary.get("label", subdir.name)
        else:
            label = subdir.name

        rows: dict[str, tuple] = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row["path"]
                solved = row["solved"].strip().lower() == "true"
                time = float(row["time"])
                status = row.get("result", "")
                rows[path] = (solved, time, status)

        if not bench_paths:
            bench_paths = list(rows.keys())

        result_database[label] = [rows[p] for p in bench_paths]

    if not result_database:
        sys.exit(f"No instance_results.csv files found under {eval_dir}")

    return result_database, bench_paths, timeout


def db2(result_database):
    """Strip status field → (solved, time) tuples for portfolio functions."""
    return {k: [(r[0], r[1]) for r in v] for k, v in result_database.items()}


def virtual_best_res(result_database):
    d = db2(result_database)
    keys = list(d.keys())
    best = d[keys[0]]
    for k in keys[1:]:
        best, _ = virtual_add_strategy(best, d[k])
    return best


# ─── Display helpers ──────────────────────────────────────────────────────────

SEP_WIDE  = "─" * 74
SEP_MID   = "─" * 60


def print_strategy_stats(result_database, timeout, n):
    rows = []
    for label, res in result_database.items():
        s = solved_num([(r[0], r[1]) for r in res])
        p2 = par_n([(r[0], r[1]) for r in res], 2, timeout) / n
        rows.append((label, s, p2))
    rows.sort(key=lambda x: -x[1])

    print(f"\n{'Strategy':<40} {'Solved':>7} {'PAR2':>8}")
    print(SEP_MID)
    for label, s, p2 in rows:
        print(f"  {label:<38} {s:>7} {p2:>8.3f}")


def print_portfolio_table(selected, result_database, timeout, n, vb_solved):
    d = db2(result_database)
    print(f"\n{'k':<4} {'Strategy':<40} {'Solved':>7} {'PAR2':>8} {'VB cover%':>10}")
    print(SEP_WIDE)
    cur_best = [(False, None)] * n
    for i, strat in enumerate(selected):
        cur_best, _ = virtual_add_strategy(cur_best, d[strat])
        s = solved_num(cur_best)
        p2 = par_n(cur_best, 2, timeout) / n
        vb_pct = 100.0 * s / vb_solved if vb_solved > 0 else 0.0
        print(f"  {i+1:<3} {strat:<40} {s:>7} {p2:>8.3f} {vb_pct:>9.1f}%")


def run_selection(selector, bench_paths, result_database, timeout, n, vb_solved, sbs_solved, sbs_par2, feature_dict=None):
    """Run per-instance selection and print results. Returns list of (path, strategy, solved, time)."""
    path_index = {p: i for i, p in enumerate(bench_paths)}
    predictions = []
    sel_results = []
    strategy_counts: dict[str, int] = {}
    for path in bench_paths:
        if feature_dict and path in feature_dict:
            strat = selector.select_vec(feature_dict[path])
        else:
            strat = selector.select(path)
        idx = path_index[path]
        r = result_database[strat][idx]
        sel_results.append((r[0], r[1]))
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
        predictions.append((path, strat, r[0], r[1]))

    sel_solved = solved_num(sel_results)
    sel_par2 = par_n(sel_results, 2, timeout) / n
    vb_cover = 100.0 * sel_solved / vb_solved if vb_solved > 0 else 0.0

    print(f"\n{'Approach':<35} {'Solved':>7} {'PAR2':>8} {'VB cover%':>10}")
    print(SEP_MID)
    print(f"  {'Virtual best (oracle)':<33} {vb_solved:>7}          {'100.0%':>10}")
    print(f"  {'SBS (best single)':<33} {sbs_solved:>7} {sbs_par2:>8.3f} {100.0*sbs_solved/vb_solved:>9.1f}%")
    print(f"  {'PWC selector (train eval)':<33} {sel_solved:>7} {sel_par2:>8.3f} {vb_cover:>9.1f}%")

    print(f"\n  Strategy selection distribution:")
    print(f"  {'Strategy':<40} {'Selected':>9} {'%':>6}")
    print("  " + "─" * 58)
    for strat, cnt in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / n
        mark = " *" if cnt == 0 else ""
        print(f"  {strat:<40} {cnt:>9} {pct:>5.1f}%{mark}")
    unused = sum(1 for c in strategy_counts.values() if c == 0)
    if unused:
        print(f"  (* {unused} strategies never selected)")

    return predictions


def save_outputs(out_dir: Path, selector, predictions, shortlist_labels, timeout, n):
    out_dir.mkdir(parents=True, exist_ok=True)

    selector.save(out_dir / "selector.joblib")

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "selected_strategy", "solved", "time"])
        for path, strat, solved, time in predictions:
            writer.writerow([path, strat, solved, f"{time:.4f}"])

    meta = {
        "shortlist": shortlist_labels,
        "n_instances": n,
        "timeout": timeout,
        "n_pairs": len(shortlist_labels) * (len(shortlist_labels) - 1) // 2,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to {out_dir}/")
    print(f"    selector.joblib   — trained PWC selector")
    print(f"    predictions.csv   — per-instance strategy assignments")
    print(f"    meta.json         — shortlist and run config")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path, help="Path to a strategy_eval logic directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory under which a timestamped run folder is created (default: <eval_dir>/selection_runs)")
    args = parser.parse_args()

    if not args.eval_dir.is_dir():
        sys.exit(f"Directory not found: {args.eval_dir}")

    result_database, bench_paths, timeout = load_results(args.eval_dir)
    n = len(bench_paths)
    n_strats = len(result_database)
    print(f"Loaded {n_strats} strategies, {n} instances, timeout={timeout}s")

    # ── Per-strategy stats ────────────────────────────────────────────────────
    print_strategy_stats(result_database, timeout, n)

    # ── Virtual best ──────────────────────────────────────────────────────────
    vb = virtual_best_res(result_database)
    vb_solved = solved_num(vb)
    vb_par2 = par_n(vb, 2, timeout) / n
    print(f"\n  {'Virtual best (oracle)':<38} {vb_solved:>7} {vb_par2:>8.3f}")

    # ── Full greedy portfolio ─────────────────────────────────────────────────
    print(f"\n{'═'*74}")
    print(f"  Greedy portfolio (all {n_strats} strategies)")
    print(f"{'═'*74}")

    all_selected, log_str = create_greedy_linear_strategy_portfolio(n_strats, db2(result_database), timeout)

    for line in log_str.strip().splitlines():
        print(" ", line)

    print_portfolio_table(all_selected, result_database, timeout, n, vb_solved)

    # ── Prompt for shortlist size ─────────────────────────────────────────────
    print(f"\n{'═'*74}")
    while True:
        try:
            raw = input(f"  Choose shortlist size k for PWC algorithm selection (1–{len(all_selected)}): ").strip()
            k = int(raw)
            if 1 <= k <= len(all_selected):
                break
            print(f"  Please enter a number between 1 and {len(all_selected)}.")
        except (ValueError, EOFError):
            print("  Invalid input.")

    shortlist_labels = all_selected[:k]

    # SBS within shortlist
    d = db2(result_database)
    sbs_label = min(shortlist_labels, key=lambda s: par_n(d[s], 2, timeout))
    sbs_solved = solved_num(d[sbs_label])
    sbs_par2 = par_n(d[sbs_label], 2, timeout) / n

    # ── Train PWC selector ────────────────────────────────────────────────────
    print(f"\n{'═'*74}")
    print(f"  PWC Algorithm Selection  (shortlist: {k} strategies)")
    print(f"{'═'*74}")

    print(f"  Features:")
    feature_dict = load_or_build_features(bench_paths)
    n_ok = sum(1 for p in bench_paths if p in feature_dict)
    print(f"  {n_ok}/{n} benchmarks have features")

    print(f"  Training {k*(k-1)//2} pairwise SVM models ...", end="", flush=True)
    shortlist = [(s, result_database[s]) for s in shortlist_labels]
    selector = train_pwc_selector(shortlist, bench_paths, timeout, precomputed_features=feature_dict)
    print(" done")

    predictions = run_selection(selector, bench_paths, result_database, timeout, n, vb_solved, sbs_solved, sbs_par2, feature_dict)

    # ── Save outputs ──────────────────────────────────────────────────────────
    base_dir = args.output_dir if args.output_dir else args.eval_dir / "selection_runs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{timestamp}_k{k}"
    save_outputs(run_dir, selector, predictions, shortlist_labels, timeout, n)


if __name__ == "__main__":
    main()
