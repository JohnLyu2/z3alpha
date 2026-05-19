"""Stage-1 run metrics (union coverage, k-for-90%-union) and CSV ledger append."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from z3alpha.utils import solved_num

RUN_METRICS_COLUMNS: tuple[str, ...] = (
    "run_name",
    "sims",
    "num_strategies",
    "union",
    "best_single",
    "k_for_90_union",
    "wall_time_s",
    "llm_calls",
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


def k_for_union_fraction(
    result_database: dict[str, list],
    union_fraction: float = 0.9,
) -> int:
    """Greedy count of strategies needed to cover ``union_fraction`` of final union."""
    final_union = _union_indices(result_database)
    if not final_union:
        return 0
    target = math.ceil(union_fraction * len(final_union))
    if target <= 0:
        return 0

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


def compute_run_metrics(
    result_database: dict[str, list],
    *,
    union_fraction: float = 0.9,
) -> dict[str, int]:
    """Return num_strategies, union, best_single, k_for_90_union from an MCTS res_database."""
    if not result_database:
        return {
            "num_strategies": 0,
            "union": 0,
            "best_single": 0,
            "k_for_90_union": 0,
        }
    union = len(_union_indices(result_database))
    best_single = max(solved_num(res) for res in result_database.values())
    return {
        "num_strategies": len(result_database),
        "union": union,
        "best_single": best_single,
        "k_for_90_union": k_for_union_fraction(result_database, union_fraction),
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
