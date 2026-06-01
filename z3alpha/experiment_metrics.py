"""Stage-1 run metrics (union coverage, k_union, coverage curve)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from z3alpha.utils import par_n, solved_num

RUN_METRICS_COLUMNS: tuple[str, ...] = (
    "run_name",
    "sims",
    "num_strategies",
    "union",
    "best_single",
    "union_gap",
    "best_single_par2",
    "best_single_par10",
    "k_union",
    "wall_time_s",
    "llm_calls",
)

COVERAGE_CURVE_COLUMNS: tuple[str, ...] = (
    "sim",
    "num_strategies",
    "union",
    "best_single",
    "union_gap",
    "k_union",
)


def _benchmark_count(result_database: dict[str, list]) -> int:
    if not result_database:
        return 0
    first = next(iter(result_database.values()))
    return len(first)


def _union_indices(result_database: dict[str, list]) -> set[int]:
    n = _benchmark_count(result_database)
    covered: set[int] = set()
    for res_list in result_database.values():
        for i in range(n):
            if res_list[i][0]:
                covered.add(i)
    return covered


def _solved_indices(res_list: list) -> set[int]:
    return {i for i, row in enumerate(res_list) if row[0]}


def k_union(result_database: dict[str, list]) -> int:
    """Greedy min strategies needed to cover 100% of final union."""
    final_union = _union_indices(result_database)
    if not final_union:
        return 0
    target = len(final_union)

    covered: set[int] = set()
    remaining = set(result_database.keys())
    k = 0
    while len(covered) < target and remaining:
        best_strat = None
        best_gain = -1
        for strat in remaining:
            gain = len(_solved_indices(result_database[strat]) - covered)
            if gain > best_gain:
                best_gain = gain
                best_strat = strat
        if best_strat is None or best_gain <= 0:
            break
        covered |= _solved_indices(result_database[best_strat])
        remaining.remove(best_strat)
        k += 1
    return k


def filter_res_database_by_max_sim(
    result_database: dict[str, list],
    strat_first_sim: dict[str, int],
    max_sim: int,
) -> dict[str, list]:
    """Keep strategies first discovered at or before ``max_sim`` (0-based sim index)."""
    return {
        strat: result_database[strat]
        for strat in result_database
        if strat_first_sim.get(strat, max_sim + 1) <= max_sim
    }


def compute_coverage_metrics(result_database: dict[str, list]) -> dict[str, int]:
    """Return search coverage metrics without PAR (for coverage curve checkpoints)."""
    if not result_database:
        return {
            "num_strategies": 0,
            "union": 0,
            "best_single": 0,
            "union_gap": 0,
            "k_union": 0,
        }
    union = len(_union_indices(result_database))
    best_single = max(solved_num(res) for res in result_database.values())
    return {
        "num_strategies": len(result_database),
        "union": union,
        "best_single": best_single,
        "union_gap": union - best_single,
        "k_union": k_union(result_database),
    }


def compute_run_metrics(
    result_database: dict[str, list],
    timeout: float,
) -> dict[str, int | float]:
    """Return end-of-run coverage and best-single PAR metrics from an MCTS res_database."""
    coverage = compute_coverage_metrics(result_database)
    if not result_database:
        return {
            **coverage,
            "best_single_par2": 0.0,
            "best_single_par10": 0.0,
        }
    best_single_par2 = min(par_n(res, 2, timeout) for res in result_database.values())
    best_single_par10 = min(par_n(res, 10, timeout) for res in result_database.values())
    return {
        **coverage,
        "best_single_par2": best_single_par2,
        "best_single_par10": best_single_par10,
    }


def append_run_metrics_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one row to the shared run metrics ledger; write header if new file."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_METRICS_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: row[col] for col in RUN_METRICS_COLUMNS})


def init_coverage_curve_csv(csv_path: Path) -> None:
    """Create ``coverage_curve.csv`` with header in a run output folder."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COVERAGE_CURVE_COLUMNS)
        writer.writeheader()


def append_coverage_curve_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Append one checkpoint row to ``coverage_curve.csv``."""
    csv_path = Path(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COVERAGE_CURVE_COLUMNS)
        writer.writerow({col: row[col] for col in COVERAGE_CURVE_COLUMNS})


_LLM_API_QA_LOG_KINDS = frozenset({"llm_prior_qa", "llm_prior_request_error"})


def llm_calls_from_qa_log(qa_log_path: Path) -> int:
    """Count live LLM API attempts recorded in ``llm_prior_qa.log``."""
    qa_log_path = Path(qa_log_path)
    if not qa_log_path.is_file():
        return 0
    count = 0
    with open(qa_log_path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("kind: "):
                continue
            kind = line[len("kind: ") :].strip()
            if kind in _LLM_API_QA_LOG_KINDS:
                count += 1
    return count


def res_database_from_per_benchmark_csv(csv_path: Path) -> dict[str, list[tuple]]:
    """Rebuild res_database from linear_strategy_per_benchmark.csv."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    by_strat: dict[str, dict[str, tuple]] = {}
    bench_order: list[str] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strat = row["strat"]
            bench = row["benchmark"]
            solved = row["solved"].strip().lower() in ("true", "1", "yes")
            time_s = float(row["time_s"])
            status = row["status"]
            if bench not in bench_order:
                bench_order.append(bench)
            by_strat.setdefault(strat, {})[bench] = (solved, time_s, status)

    bench_order.sort()
    result_database: dict[str, list[tuple]] = {}
    for strat, bench_map in by_strat.items():
        result_database[strat] = [bench_map[b] for b in bench_order]
    return result_database
