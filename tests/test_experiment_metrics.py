"""Unit tests for z3alpha.experiment_metrics."""

import csv
import tempfile
import unittest
from pathlib import Path

from z3alpha.experiment_metrics import (
    RUN_METRICS_COLUMNS,
    append_run_metrics_row,
    compute_run_metrics,
    k_for_union_fraction,
    res_database_from_per_benchmark_csv,
)


def _res(solved_flags: list[bool]) -> list[tuple]:
    return [(s, 1.0 if s else 10.0, "sat" if s else "timeout") for s in solved_flags]


class TestComputeRunMetrics(unittest.TestCase):
    def test_empty_database(self) -> None:
        m = compute_run_metrics({})
        self.assertEqual(m["num_strategies"], 0)
        self.assertEqual(m["union"], 0)
        self.assertEqual(m["best_single"], 0)
        self.assertEqual(m["k_for_90_union"], 0)

    def test_union_and_best_single(self) -> None:
        db = {
            "a": _res([True, True, False, False]),
            "b": _res([False, False, True, True]),
        }
        m = compute_run_metrics(db)
        self.assertEqual(m["num_strategies"], 2)
        self.assertEqual(m["union"], 4)
        self.assertEqual(m["best_single"], 2)

    def test_k_for_90_union_disjoint_pair(self) -> None:
        # 10 instances: strat1 covers 6, strat2 covers 4 disjoint -> union 10, 90% -> 9, k=2
        db = {
            "s1": _res([True] * 6 + [False] * 4),
            "s2": _res([False] * 6 + [True] * 4),
        }
        self.assertEqual(k_for_union_fraction(db, 0.9), 2)
        m = compute_run_metrics(db)
        self.assertEqual(m["k_for_90_union"], 2)

    def test_k_zero_when_no_union(self) -> None:
        db = {"s": _res([False, False])}
        self.assertEqual(k_for_union_fraction(db, 0.9), 0)


class TestAppendRunMetricsRow(unittest.TestCase):
    def test_header_once_then_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_metrics.csv"
            row = {
                "run_name": "out-test",
                "sims": 10,
                "num_strategies": 3,
                "union": 5,
                "best_single": 4,
                "k_for_90_union": 2,
                "wall_time_s": 100,
                "llm_calls": 0,
            }
            append_run_metrics_row(path, row)
            append_run_metrics_row(path, {**row, "run_name": "out-test-2"})

            with open(path, newline="", encoding="utf-8") as f:
                lines = list(csv.reader(f))
            self.assertEqual(lines[0], list(RUN_METRICS_COLUMNS))
            self.assertEqual(len(lines), 3)


class TestResDatabaseFromCsv(unittest.TestCase):
    def test_roundtrip_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["strat", "benchmark", "status", "time_s", "solved"])
                w.writerow(["s1", "b0.smt2", "sat", "0.1", "True"])
                w.writerow(["s1", "b1.smt2", "timeout", "10", "False"])
                w.writerow(["s2", "b0.smt2", "sat", "0.2", "True"])
                w.writerow(["s2", "b1.smt2", "sat", "0.3", "True"])

            db = res_database_from_per_benchmark_csv(path)
            self.assertEqual(len(db), 2)
            self.assertEqual(len(db["s1"]), 2)
            m = compute_run_metrics(db)
            self.assertEqual(m["union"], 2)
            self.assertEqual(m["best_single"], 2)


if __name__ == "__main__":
    unittest.main()
