"""Unit tests for z3alpha.experiment_metrics."""

import csv
import tempfile
import unittest
from pathlib import Path

from z3alpha.experiment_metrics import (
    COVERAGE_CURVE_COLUMNS,
    RUN_METRICS_COLUMNS,
    append_coverage_curve_row,
    append_run_metrics_row,
    compute_coverage_metrics,
    compute_run_metrics,
    filter_res_database_by_max_sim,
    init_coverage_curve_csv,
    k_union,
    llm_calls_from_qa_log,
    res_database_from_per_benchmark_csv,
)


def _res(solved_flags: list[bool]) -> list[tuple]:
    return [(s, 1.0 if s else 10.0, "sat" if s else "timeout") for s in solved_flags]


class TestComputeRunMetrics(unittest.TestCase):
    def test_empty_database(self) -> None:
        m = compute_run_metrics({}, timeout=10)
        self.assertEqual(m["num_strategies"], 0)
        self.assertEqual(m["union"], 0)
        self.assertEqual(m["best_single"], 0)
        self.assertEqual(m["union_gap"], 0)
        self.assertEqual(m["best_single_par2"], 0.0)
        self.assertEqual(m["best_single_par10"], 0.0)
        self.assertEqual(m["k_union"], 0)

    def test_union_and_best_single(self) -> None:
        db = {
            "a": _res([True, True, False, False]),
            "b": _res([False, False, True, True]),
        }
        m = compute_run_metrics(db, timeout=10)
        self.assertEqual(m["num_strategies"], 2)
        self.assertEqual(m["union"], 4)
        self.assertEqual(m["best_single"], 2)
        self.assertEqual(m["union_gap"], 2)
        self.assertEqual(m["best_single_par2"], 42.0)
        self.assertEqual(m["best_single_par10"], 202.0)

    def test_k_union_disjoint_pair(self) -> None:
        db = {
            "s1": _res([True] * 6 + [False] * 4),
            "s2": _res([False] * 6 + [True] * 4),
        }
        self.assertEqual(k_union(db), 2)
        m = compute_run_metrics(db, timeout=10)
        self.assertEqual(m["k_union"], 2)

    def test_k_union_dominant_single_with_small_gap(self) -> None:
        db = {
            "s1": _res([True] * 9 + [False]),
            "s2": _res([False] * 9 + [True]),
        }
        m = compute_coverage_metrics(db)
        self.assertEqual(m["union"], 10)
        self.assertEqual(m["best_single"], 9)
        self.assertEqual(m["union_gap"], 1)
        self.assertEqual(m["k_union"], 2)

    def test_k_zero_when_no_union(self) -> None:
        db = {"s": _res([False, False])}
        self.assertEqual(k_union(db), 0)


class TestFilterResDatabaseByMaxSim(unittest.TestCase):
    def test_partial_database(self) -> None:
        db = {
            "early": _res([True, False]),
            "late": _res([False, True]),
        }
        first_sim = {"early": 2, "late": 8}
        subset = filter_res_database_by_max_sim(db, first_sim, max_sim=4)
        self.assertEqual(set(subset.keys()), {"early"})
        m = compute_coverage_metrics(subset)
        self.assertEqual(m["num_strategies"], 1)
        self.assertEqual(m["union"], 1)


class TestCoverageCurveCsv(unittest.TestCase):
    def test_init_and_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "coverage_curve.csv"
            init_coverage_curve_csv(path)
            append_coverage_curve_row(
                path,
                {
                    "sim": 5,
                    "num_strategies": 3,
                    "union": 10,
                    "best_single": 8,
                    "union_gap": 2,
                    "k_union": 2,
                },
            )
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(list(rows[0].keys()), list(COVERAGE_CURVE_COLUMNS))
            self.assertEqual(rows[0]["sim"], "5")
            self.assertEqual(rows[0]["k_union"], "2")


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
                "union_gap": 1,
                "best_single_par2": 12.0,
                "best_single_par10": 52.0,
                "k_union": 2,
                "wall_time_s": 100,
                "llm_calls": 0,
                "notes": "smoke test",
            }
            append_run_metrics_row(path, row)
            append_run_metrics_row(path, {**row, "run_name": "out-test-2", "notes": ""})

            with open(path, newline="", encoding="utf-8") as f:
                lines = list(csv.reader(f))
            self.assertEqual(lines[0], list(RUN_METRICS_COLUMNS))
            self.assertEqual(len(lines), 3)


class TestLlmCallsFromQaLog(unittest.TestCase):
    def test_counts_api_attempts_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llm_prior_qa.log"
            path.write_text(
                "\n".join(
                    [
                        "kind: llm_prior_qa",
                        "kind: llm_prior_request_error",
                        "kind: llm_prior_cache_hit",
                        "kind: llm_prior_uniform_fallback",
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(llm_calls_from_qa_log(path), 2)

    def test_missing_log_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(llm_calls_from_qa_log(Path(tmp) / "missing.log"), 0)


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
            m = compute_run_metrics(db, timeout=10)
            self.assertEqual(m["union"], 2)
            self.assertEqual(m["best_single"], 2)
            self.assertEqual(m["union_gap"], 0)
            self.assertEqual(m["best_single_par2"], 0.5)
            self.assertEqual(m["best_single_par10"], 0.5)


if __name__ == "__main__":
    unittest.main()
