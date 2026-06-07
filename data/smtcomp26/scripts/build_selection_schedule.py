#!/usr/bin/env python3
"""
Greedy portfolio analysis + PWC algorithm selection for strategy evaluation results.

Loads per-instance CSV results from a strategy_eval directory, ranks all strategies
with a greedy portfolio, then trains a PWC SVM selector on the full set.

Usage:
    python build_selection_schedule.py <eval_dir>
    python build_selection_schedule.py <eval_dir> --no-schedule

Example:
    python build_selection_schedule.py data/smtcomp26/strategy_eval/QF_NRA
    python build_selection_schedule.py data/smtcomp26/strategy_eval/QF_NRA --no-schedule
"""

import argparse
import csv
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from z3alpha.smt_select import bench_feature_vector, train_pwc_selector, FEATURE_NAMES
from z3alpha.strategy_portfolio import create_greedy_linear_strategy_portfolio, virtual_add_strategy
from z3alpha.utils import solved_num, par_n

import logging
logging.disable(logging.CRITICAL)

REPO_ROOT      = Path(__file__).resolve().parents[3]
BENCH_LIST_DIR = REPO_ROOT / "data/smtcomp26/smtcomp25_benchlist"
FEATURES_DIR   = REPO_ROOT / "data/smtcomp26/features"

def _load_env_config() -> dict:
    config_path = REPO_ROOT / "env_config.json"
    return json.loads(config_path.read_text()) if config_path.exists() else {}

_ENV = _load_env_config()
SMTLIB_ROOT = Path(_ENV.get("smtlib_root", "/home/z52lu/smtlib25"))
N_WORKERS   = _ENV.get("workers", 8)


# ─── Feature cache ────────────────────────────────────────────────────────────

def _extract_one(rel_path: str) -> tuple[str, float, object]:
    t0 = time.perf_counter()
    feat = bench_feature_vector(SMTLIB_ROOT / rel_path)
    return rel_path, time.perf_counter() - t0, feat


def load_or_build_features(bench_paths: list[str]) -> dict[str, np.ndarray]:
    """Return path → feature vector for bench_paths, building and caching as needed."""
    bench_set = set(bench_paths)

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
                cached = []
                for row in reader:
                    p, feat = row[0], np.array(row[2:], dtype=float)
                    cached.append(p)
                    if p in bench_set:
                        feature_dict[p] = feat
            n_hit = sum(1 for p in cached if p in bench_set)
            print(f"  [{logic_name}] loaded from cache  ({n_hit}/{len(cached)} in eval)")
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

    Returns (result_database, bench_paths, timeout, strategy_cli) where:
      result_database: label -> list of (solved, time, status) aligned to bench_paths
      bench_paths: ordered list of relative benchmark paths
      timeout: per-instance timeout in seconds
      strategy_cli: label -> list of Z3 CLI args
    """
    result_database: dict[str, list] = {}
    strategy_cli: dict[str, list[str]] = {}
    bench_paths: list[str] = []
    timeout = 30

    subdirs = sorted(p for p in eval_dir.iterdir() if p.is_dir())
    if not subdirs:
        sys.exit(f"No subdirectories found in {eval_dir}")

    for subdir in subdirs:
        csv_path = subdir / "instance_results.csv"
        if not csv_path.exists():
            continue

        summary_path = subdir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            run_cfg = summary.get("run_config", {})
            timeout = run_cfg.get("timeout", timeout)
            label = summary.get("label", subdir.name)
            strategy = run_cfg.get("strategy")
            z3_params = run_cfg.get("z3_extra_params", [])[:]
            if strategy:
                z3_params = [f"tactic.default_tactic={strategy}"] + z3_params
            strategy_cli[label] = z3_params
        else:
            label = subdir.name
            strategy_cli[label] = []

        rows: dict[str, tuple] = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                rows[row["path"]] = (
                    row["solved"].strip().lower() == "true",
                    float(row["time"]),
                    row.get("result", ""),
                )

        if not bench_paths:
            bench_paths = list(rows.keys())

        result_database[label] = [rows[p] for p in bench_paths]

    if not result_database:
        sys.exit(f"No instance_results.csv files found under {eval_dir}")

    return result_database, bench_paths, timeout, strategy_cli


def _strip_status(result_database):
    """Drop status field: label -> [(solved, time), ...]."""
    return {k: [(r[0], r[1]) for r in v] for k, v in result_database.items()}


def _virtual_best(result_database):
    d = _strip_status(result_database)
    best = list(d.values())[0]
    for res in list(d.values())[1:]:
        best, _ = virtual_add_strategy(best, res)
    return best


# ─── Threshold cross-validation ──────────────────────────────────────────────

CV_FOLDS         = 4
CV_WEIGHT_TYPES  = ["par2", "par10"]
CV_THRESHOLDS    = [0, 5, 10]
CV_N_ESTIMATORS  = [50, 100]
CV_MAX_DEPTHS    = [3, 6]
SCHEDULE_SPLIT   = 0.80


def _ref_stats(idx_list, result_database, shortlist_labels, timeout):
    """Compute VBS and SBS solved/PAR-2 for a set of instance indices."""
    vbs = [(False, None)] * len(idx_list)
    for label in shortlist_labels:
        fold_res = [(result_database[label][i][0], result_database[label][i][1]) for i in idx_list]
        vbs, _ = virtual_add_strategy(vbs, fold_res)

    sbs_label = min(shortlist_labels,
                    key=lambda s: par_n([(result_database[s][i][0], result_database[s][i][1]) for i in idx_list], 2, timeout))
    sbs = [(result_database[sbs_label][i][0], result_database[sbs_label][i][1]) for i in idx_list]

    n = len(idx_list)
    return (solved_num(vbs), par_n(vbs, 2, timeout) / n,
            solved_num(sbs), par_n(sbs, 2, timeout) / n)


def run_threshold_cv(shortlist_labels, result_database, bench_paths, feature_dict,
                     timeout, n_folds=CV_FOLDS, weight_types=CV_WEIGHT_TYPES,
                     thresholds=CV_THRESHOLDS, n_estimators_list=CV_N_ESTIMATORS,
                     max_depths=CV_MAX_DEPTHS, random_seed=42):
    """Phase 1: K-fold CV over weight_type × n_estimators × max_depth × threshold (selector only).

    Returns (best_params, cv_rows, ref_rows) where:
      best_params: dict with keys weight_type, n_estimators, max_depth, threshold
      cv_rows: list of (weight_type, n_estimators, max_depth, threshold,
                        mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2)
      ref_rows: list of (label, mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2) for VBS and SBS
    Best combination chosen by selector test solved (then test PAR-2).
    """
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(bench_paths)).tolist()
    folds = [indices[i::n_folds] for i in range(n_folds)]

    def _eval_selector(selector, idx_list):
        results = []
        for i in idx_list:
            path = bench_paths[i]
            strat = (selector.select_vec(feature_dict[path])
                     if path in feature_dict else selector.select(SMTLIB_ROOT / path))
            r = result_database[strat][i]
            results.append((r[0], r[1]))
        return solved_num(results), par_n(results, 2, timeout) / len(idx_list)

    def _eval_scheduler(selector, idx_list, split=0.75):
        budget_a, budget_b = split * timeout, (1 - split) * timeout
        results = []
        for i in idx_list:
            path = bench_paths[i]
            if path in feature_dict:
                rank1, rank2 = selector.select_top2_vec(feature_dict[path])
            else:
                rank1, rank2 = selector.select_top2_vec(bench_feature_vector(SMTLIB_ROOT / path))
            r1 = result_database[rank1][i]
            if r1[0] and r1[1] <= budget_a:
                results.append((True, r1[1]))
            else:
                r2 = result_database[rank2][i]
                if r2[0] and r2[1] <= budget_b:
                    results.append((True, budget_a + r2[1]))
                else:
                    results.append((False, timeout))
        return solved_num(results), par_n(results, 2, timeout) / len(idx_list)

    # ── Reference stats (VBS/SBS, threshold-independent) ─────────────────
    fold_ref = {"vbs_tr": [], "vbs_te": [], "sbs_tr": [], "sbs_te": []}
    for fold_idx in range(n_folds):
        test_idx  = folds[fold_idx]
        train_idx = [i for f in range(n_folds) if f != fold_idx for i in folds[f]]
        vbs_tr_s, vbs_tr_p2, sbs_tr_s, sbs_tr_p2 = _ref_stats(train_idx, result_database, shortlist_labels, timeout)
        vbs_te_s, vbs_te_p2, sbs_te_s, sbs_te_p2 = _ref_stats(test_idx,  result_database, shortlist_labels, timeout)
        fold_ref["vbs_tr"].append((vbs_tr_s, vbs_tr_p2))
        fold_ref["vbs_te"].append((vbs_te_s, vbs_te_p2))
        fold_ref["sbs_tr"].append((sbs_tr_s, sbs_tr_p2))
        fold_ref["sbs_te"].append((sbs_te_s, sbs_te_p2))

    def _agg(pairs):
        ss = [p[0] for p in pairs]; p2s = [p[1] for p in pairs]
        return float(np.mean(ss)), float(np.std(ss)), float(np.mean(p2s))

    ref_rows = [
        ("VBS", *_agg(fold_ref["vbs_tr"]), *_agg(fold_ref["vbs_te"])),
        ("SBS", *_agg(fold_ref["sbs_tr"]), *_agg(fold_ref["sbs_te"])),
    ]

    # ── Full grid CV (selector only) ──────────────────────────────────────
    cv_rows = []
    for wtype in weight_types:
        for n_est in n_estimators_list:
            for depth in max_depths:
                for thresh in thresholds:
                    fold_train_solved, fold_train_par2 = [], []
                    fold_test_solved,  fold_test_par2  = [], []
                    for fold_idx in range(n_folds):
                        test_idx  = folds[fold_idx]
                        train_idx = [i for f in range(n_folds) if f != fold_idx for i in folds[f]]
                        train_shortlist = [
                            (s, [result_database[s][i] for i in train_idx])
                            for s in shortlist_labels
                        ]
                        selector = train_pwc_selector(
                            train_shortlist,
                            [bench_paths[i] for i in train_idx],
                            timeout,
                            precomputed_features=feature_dict,
                            perf_diff_threshold=thresh,
                            model_type="xgboost",
                            weight_type=wtype,
                            n_estimators=n_est,
                            max_depth=depth,
                        )
                        s, p = _eval_selector(selector, train_idx); fold_train_solved.append(s); fold_train_par2.append(p)
                        s, p = _eval_selector(selector, test_idx);  fold_test_solved.append(s);  fold_test_par2.append(p)

                    cv_rows.append((
                        wtype, n_est, depth, thresh,
                        float(np.mean(fold_train_solved)), float(np.std(fold_train_solved)), float(np.mean(fold_train_par2)),
                        float(np.mean(fold_test_solved)),  float(np.std(fold_test_solved)),  float(np.mean(fold_test_par2)),
                    ))

    best = min(cv_rows, key=lambda r: (-r[7], r[9]))
    best_params = {"weight_type": best[0], "n_estimators": best[1], "max_depth": best[2], "threshold": best[3]}
    return best_params, cv_rows, ref_rows


def run_scheduler_cv(best_params, shortlist_labels, result_database, bench_paths, feature_dict,
                     timeout, n_folds=CV_FOLDS, split=SCHEDULE_SPLIT, random_seed=42):
    """Phase 2: Compare selector vs scheduler using the best hyperparameters from phase 1.

    Returns (sch_rows) where sch_rows is a list of
    (label, mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2) for selector and scheduler.
    """
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(bench_paths)).tolist()
    folds = [indices[i::n_folds] for i in range(n_folds)]

    def _eval_selector(selector, idx_list):
        results = []
        for i in idx_list:
            path = bench_paths[i]
            strat = (selector.select_vec(feature_dict[path])
                     if path in feature_dict else selector.select(SMTLIB_ROOT / path))
            r = result_database[strat][i]
            results.append((r[0], r[1]))
        return solved_num(results), par_n(results, 2, timeout) / len(idx_list)

    def _eval_scheduler(selector, idx_list):
        budget_a, budget_b = split * timeout, (1 - split) * timeout
        results = []
        for i in idx_list:
            path = bench_paths[i]
            if path in feature_dict:
                rank1, rank2 = selector.select_top2_vec(feature_dict[path])
            else:
                rank1, rank2 = selector.select_top2_vec(bench_feature_vector(SMTLIB_ROOT / path))
            r1 = result_database[rank1][i]
            if r1[0] and r1[1] <= budget_a:
                results.append((True, r1[1]))
            else:
                r2 = result_database[rank2][i]
                if r2[0] and r2[1] <= budget_b:
                    results.append((True, budget_a + r2[1]))
                else:
                    results.append((False, timeout))
        return solved_num(results), par_n(results, 2, timeout) / len(idx_list)

    sel_tr_s, sel_tr_p2_l, sel_te_s, sel_te_p2_l = [], [], [], []
    sch_tr_s, sch_tr_p2_l, sch_te_s, sch_te_p2_l = [], [], [], []

    for fold_idx in range(n_folds):
        test_idx  = folds[fold_idx]
        train_idx = [i for f in range(n_folds) if f != fold_idx for i in folds[f]]
        train_shortlist = [
            (s, [result_database[s][i] for i in train_idx])
            for s in shortlist_labels
        ]
        selector = train_pwc_selector(
            train_shortlist,
            [bench_paths[i] for i in train_idx],
            timeout,
            precomputed_features=feature_dict,
            perf_diff_threshold=best_params["threshold"],
            model_type="xgboost",
            weight_type=best_params["weight_type"],
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
        )
        s, p = _eval_selector(selector, train_idx); sel_tr_s.append(s); sel_tr_p2_l.append(p)
        s, p = _eval_selector(selector, test_idx);  sel_te_s.append(s); sel_te_p2_l.append(p)
        s, p = _eval_scheduler(selector, train_idx); sch_tr_s.append(s); sch_tr_p2_l.append(p)
        s, p = _eval_scheduler(selector, test_idx);  sch_te_s.append(s); sch_te_p2_l.append(p)

    def _agg(ss, p2s):
        return float(np.mean(ss)), float(np.std(ss)), float(np.mean(p2s))

    return [
        ("Selector",                *_agg(sel_tr_s, sel_tr_p2_l), *_agg(sel_te_s, sel_te_p2_l)),
        (f"Scheduler ({split:.0%}+{1-split:.0%})", *_agg(sch_tr_s, sch_tr_p2_l), *_agg(sch_te_s, sch_te_p2_l)),
    ]


def print_cv_table(cv_rows, best_params, ref_rows):
    header1 = f"  {'':>10} {'':>5} {'':>5} {'':>6}   {'--- Train ---':^26}   {'--- Test ---':^26}"
    header2 = f"  {'Weight':>10} {'n_est':>5} {'depth':>5} {'thresh':>6}   {'solved (mean±std)':>18} {'PAR-2':>6}   {'solved (mean±std)':>18} {'PAR-2':>6}"
    sep = "  " + "─" * 82

    def _stat_cols(mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2):
        return f"{mtr_s:>7.1f} ± {str_s:<6.1f} {mtr_p2:>6.1f}   {mte_s:>7.1f} ± {ste_s:<6.1f} {mte_p2:>6.1f}"

    print(f"\n{header1}\n{header2}\n{sep}")
    for label, mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2 in ref_rows:
        print(f"  {'':>10} {'':>5} {'':>5} {'':>6}   {_stat_cols(mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2)}  ({label})")
    print(sep)
    cur_group = None
    for wtype, n_est, depth, thresh, mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2 in cv_rows:
        group = (wtype, n_est)
        if cur_group and group != cur_group:
            print()
        cur_group = group
        is_best = (wtype == best_params["weight_type"] and n_est == best_params["n_estimators"]
                   and depth == best_params["max_depth"] and thresh == best_params["threshold"])
        mark = "  ← best" if is_best else ""
        print(f"  {wtype:>10} {n_est:>5} {depth:>5} {thresh:>6}   {_stat_cols(mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2)}{mark}")


def print_scheduler_comparison(sch_rows, ref_rows):
    header1 = f"  {'':>26}   {'--- Train ---':^26}   {'--- Test ---':^26}"
    header2 = f"  {'Approach':>26}   {'solved (mean±std)':>18} {'PAR-2':>6}   {'solved (mean±std)':>18} {'PAR-2':>6}"
    sep = "  " + "─" * 82

    def _stat_cols(mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2):
        return f"{mtr_s:>7.1f} ± {str_s:<6.1f} {mtr_p2:>6.1f}   {mte_s:>7.1f} ± {ste_s:<6.1f} {mte_p2:>6.1f}"

    print(f"\n{header1}\n{header2}\n{sep}")
    for label, mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2 in ref_rows:
        print(f"  {label:>26}   {_stat_cols(mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2)}")
    print(sep)
    for label, mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2 in sch_rows:
        print(f"  {label:>26}   {_stat_cols(mtr_s, str_s, mtr_p2, mte_s, ste_s, mte_p2)}")


# ─── Display helpers ──────────────────────────────────────────────────────────

SEP_WIDE = "─" * 74
SEP_MID  = "─" * 60


def print_strategy_stats(result_database, timeout, n):
    rows = []
    for label, res in result_database.items():
        d = [(r[0], r[1]) for r in res]
        rows.append((label, solved_num(d), par_n(d, 2, timeout) / n))
    rows.sort(key=lambda x: -x[1])

    print(f"\n{'Strategy':<40} {'Solved':>7} {'PAR2':>8}")
    print(SEP_MID)
    for label, s, p2 in rows:
        print(f"  {label:<38} {s:>7} {p2:>8.3f}")



def run_selection(selector, bench_paths, result_database, shortlist_labels, timeout, n, sbs_solved, sbs_par2, feature_dict=None):
    """Run per-instance selection and print summary."""
    sel_results = []
    strategy_counts = {s: 0 for s in shortlist_labels}

    for i, path in enumerate(bench_paths):
        strat = selector.select_vec(feature_dict[path]) if (feature_dict and path in feature_dict) else selector.select(path)
        r = result_database[strat][i]
        sel_results.append((r[0], r[1]))
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

    sel_solved = solved_num(sel_results)
    sel_par2 = par_n(sel_results, 2, timeout) / n

    print(f"\n{'Approach':<35} {'Solved':>7} {'PAR2':>8}")
    print(SEP_MID)
    print(f"  {'SBS (best single)':<33} {sbs_solved:>7} {sbs_par2:>8.3f}")
    print(f"  {'PWC selector (train eval)':<33} {sel_solved:>7} {sel_par2:>8.3f}")

    print(f"\n  Strategy selection distribution:")
    print(f"  {'Strategy':<40} {'Selected':>9} {'%':>6}")
    print("  " + "─" * 58)
    for strat, cnt in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / n
        mark = " (never selected)" if cnt == 0 else ""
        print(f"  {strat:<40} {cnt:>9} {pct:>5.1f}%{mark}")


def save_outputs(out_dir: Path, selector, shortlist_labels, strategy_cli, timeout,
                 use_scheduler: bool, logic: str):
    selector.save(out_dir / "selector.joblib")

    meta = {
        "logic": logic,
        "shortlist": shortlist_labels,
        "strategy_cli": {s: strategy_cli.get(s, []) for s in shortlist_labels},
        "scheduler": use_scheduler,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved to {out_dir}/")
    print(f"    selector.joblib  — trained PWC selector")
    print(f"    meta.json        — inference config (logic, shortlist, strategy_cli, scheduler)")
    print(f"    experiment.log   — full run log")


# ─── Main ─────────────────────────────────────────────────────────────────────

class _Tee:
    """Write to two streams simultaneously (stdout + log file)."""
    def __init__(self, primary, secondary):
        self._p, self._s = primary, secondary
    def write(self, s):
        self._p.write(s); self._s.write(s)
    def flush(self):
        self._p.flush(); self._s.flush()
    def fileno(self):
        return self._p.fileno()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dir", type=Path, help="Path to a strategy_eval logic directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory under which a timestamped run folder is created "
                             "(default: data/smtcomp26/selection_runs/<logic>)")
    parser.add_argument("--no-schedule", action="store_true",
                        help="Skip Phase 2 scheduler CV; write scheduler=false to meta.json")
    args = parser.parse_args()

    if not args.eval_dir.is_dir():
        sys.exit(f"Directory not found: {args.eval_dir}")

    base_dir = args.output_dir or REPO_ROOT / "data/smtcomp26/selection_runs" / args.eval_dir.resolve().name
    run_dir = base_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True)

    log_file = open(run_dir / "experiment.log", "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    result_database, bench_paths, timeout, strategy_cli = load_results(args.eval_dir)
    n = len(bench_paths)
    n_strats = len(result_database)
    print(f"Loaded {n_strats} strategies, {n} instances, timeout={timeout}s")

    print_strategy_stats(result_database, timeout, n)

    vb = _virtual_best(result_database)
    vb_par2 = par_n(vb, 2, timeout) / n
    print(f"\n  {'Virtual best (oracle)':<38} {solved_num(vb):>7} {vb_par2:>8.3f}")

    print(f"\n{'═'*74}")
    print(f"  Greedy portfolio")
    print(f"{'═'*74}")

    shortlist_labels, log_str = create_greedy_linear_strategy_portfolio(n_strats, _strip_status(result_database), timeout)
    for line in log_str.strip().splitlines():
        print(" ", line)

    k = len(shortlist_labels)
    d = _strip_status(result_database)
    sbs_label = min(shortlist_labels, key=lambda s: par_n(d[s], 2, timeout))
    sbs_solved = solved_num(d[sbs_label])
    sbs_par2 = par_n(d[sbs_label], 2, timeout) / n

    print(f"\n{'═'*74}")
    print(f"  PWC Algorithm Selection  ({k} strategies)")
    print(f"{'═'*74}")

    feature_dict = load_or_build_features(bench_paths)
    n_ok = sum(1 for p in bench_paths if p in feature_dict)
    print(f"  Features: {n_ok}/{n} benchmarks")

    print(f"\n{'═'*74}")
    print(f"  Phase 1: Hyperparameter CV (selector only)")
    print(f"{'═'*74}")
    n_cv_models = CV_FOLDS * len(CV_WEIGHT_TYPES) * len(CV_N_ESTIMATORS) * len(CV_MAX_DEPTHS) * len(CV_THRESHOLDS) * k*(k-1)//2
    print(f"  Training {n_cv_models} models "
          f"({CV_FOLDS} folds × {len(CV_WEIGHT_TYPES)} weights × {len(CV_N_ESTIMATORS)} n_est × "
          f"{len(CV_MAX_DEPTHS)} depths × {len(CV_THRESHOLDS)} thresholds × {k*(k-1)//2} pairs) ...", end="", flush=True)
    best_params, cv_rows, ref_rows = run_threshold_cv(shortlist_labels, result_database, bench_paths, feature_dict, timeout)
    print(" done")
    print_cv_table(cv_rows, best_params, ref_rows)
    print(f"\n  Best params: {best_params}")

    if args.no_schedule:
        use_scheduler = False
        print(f"\n  Phase 2 skipped (--no-schedule); scheduler=false")
    else:
        print(f"\n{'═'*74}")
        print(f"  Phase 2: Selector vs Scheduler ({SCHEDULE_SPLIT:.0%}+{1-SCHEDULE_SPLIT:.0%} split)")
        print(f"{'═'*74}")
        n_sch_models = CV_FOLDS * k*(k-1)//2
        print(f"  Training {n_sch_models} models ({CV_FOLDS} folds × {k*(k-1)//2} pairs) ...", end="", flush=True)
        sch_rows = run_scheduler_cv(best_params, shortlist_labels, result_database, bench_paths, feature_dict, timeout)
        print(" done")
        print_scheduler_comparison(sch_rows, ref_rows)
        use_scheduler = sch_rows[1][6] < sch_rows[0][6]

    print(f"\n{'═'*74}")
    print(f"  Final selector (all instances)")
    print(f"{'═'*74}")
    print(f"  Training ...", end="", flush=True)
    shortlist = [(s, result_database[s]) for s in shortlist_labels]
    selector = train_pwc_selector(shortlist, bench_paths, timeout, precomputed_features=feature_dict,
                                  perf_diff_threshold=best_params["threshold"], model_type="xgboost",
                                  weight_type=best_params["weight_type"], n_estimators=best_params["n_estimators"],
                                  max_depth=best_params["max_depth"])
    print(" done")

    run_selection(selector, bench_paths, result_database, shortlist_labels, timeout, n, sbs_solved, sbs_par2, feature_dict)

    save_outputs(run_dir, selector, shortlist_labels, strategy_cli, timeout, use_scheduler,
                 args.eval_dir.resolve().name)
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
