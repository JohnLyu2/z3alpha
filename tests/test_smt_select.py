"""Unit tests for z3alpha.smt_select (feature extraction, inference, training)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from z3alpha.smt_select.features import extract_features
from z3alpha.smt_select.infer import FEATURE_NAMES, PairwiseSelector, bench_feature_vector
from z3alpha.smt_select.train import train_pwc_selector


def _write_smt2(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".smt2", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestExtractFeatures(unittest.TestCase):
    def test_structural_counts_and_metadata(self) -> None:
        path = _write_smt2(
            "(set-logic QF_LIA)\n"
            "(declare-fun x () Int)\n"
            "(declare-fun y () Int)\n"
            "(assert (< x y))\n"
            "(assert (> y 0))\n"
            "(check-sat)\n"
        )
        entries = extract_features(path)
        # one query entry, then one benchmark-level metadata entry
        self.assertEqual(len(entries), 2)
        query = entries[0]
        self.assertEqual(query["assertsCount"], 2)
        self.assertEqual(query["declareConstCount"], 2)  # 0-arity declare-fun -> const
        self.assertEqual(query["declareFunCount"], 0)
        bm = entries[1]
        self.assertEqual(bm["logic"], "QF_LIA")
        self.assertEqual(bm["queryCount"], 1)
        self.assertFalse(bm["isIncremental"])

    def test_incremental_benchmark_has_multiple_queries(self) -> None:
        path = _write_smt2(
            "(declare-fun x () Int)\n"
            "(assert (> x 0))\n"
            "(check-sat)\n"
            "(assert (< x 0))\n"
            "(check-sat)\n"
        )
        entries = extract_features(path)
        bm = entries[-1]
        self.assertEqual(bm["queryCount"], 2)
        self.assertTrue(bm["isIncremental"])
        # asserts accumulate within the same scope across check-sat calls
        self.assertEqual(entries[0]["assertsCount"], 1)
        self.assertEqual(entries[1]["assertsCount"], 2)

    def test_no_check_sat_yields_no_query_entries(self) -> None:
        path = _write_smt2("(declare-fun x () Int)\n(assert (> x 0))\n")
        entries = extract_features(path)
        self.assertEqual(len(entries), 1)  # only the benchmark-level metadata


class TestBenchFeatureVector(unittest.TestCase):
    def test_vector_length_matches_feature_names(self) -> None:
        path = _write_smt2("(declare-fun x () Int)\n(assert (> x 0))\n(check-sat)\n")
        vec = bench_feature_vector(path)
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), len(FEATURE_NAMES))

    def test_normalized_size_feature_is_positive(self) -> None:
        path = _write_smt2("(declare-fun x () Int)\n(assert (> x 0))\n(check-sat)\n")
        vec = bench_feature_vector(path)
        idx = FEATURE_NAMES.index("normalizedSize")
        self.assertGreater(vec[idx], 0)

    def test_returns_none_when_no_queries(self) -> None:
        path = _write_smt2("(declare-fun x () Int)\n(assert (> x 0))\n")
        self.assertIsNone(bench_feature_vector(path))

    def test_returns_none_on_missing_file(self) -> None:
        self.assertIsNone(bench_feature_vector("/nonexistent/path/does_not_exist.smt2"))


class TestPairwiseSelectorTraining(unittest.TestCase):
    def test_single_strategy_shortlist_always_selected(self) -> None:
        shortlist = [("strat_a", [(True, 1.0, "sat")])]
        bench_paths = ["bench1.smt2"]
        precomputed = {"bench1.smt2": np.array([1.0, 2.0])}
        selector = train_pwc_selector(
            shortlist, bench_paths, timeout=10, precomputed_features=precomputed
        )
        self.assertEqual(selector.select_vec(np.array([5.0, 5.0])), "strat_a")

    def test_two_strategies_split_by_feature_value(self) -> None:
        # strat_a solves low-feature benchmarks fast; strat_b solves high-feature ones fast.
        bench_paths = [f"b{i}.smt2" for i in range(20)]
        precomputed = {p: np.array([float(i)]) for i, p in enumerate(bench_paths)}
        results_a = [
            (i < 10, 1.0, "sat") if i < 10 else (False, 0.0, "timeout")
            for i in range(20)
        ]
        results_b = [
            (i >= 10, 1.0, "sat") if i >= 10 else (False, 0.0, "timeout")
            for i in range(20)
        ]
        shortlist = [("strat_a", results_a), ("strat_b", results_b)]
        selector = train_pwc_selector(
            shortlist, bench_paths, timeout=10, precomputed_features=precomputed
        )
        self.assertEqual(selector.select_vec(np.array([1.0])), "strat_a")
        self.assertEqual(selector.select_vec(np.array([18.0])), "strat_b")

    def test_save_and_load_round_trip(self) -> None:
        shortlist = [("strat_a", [(True, 1.0, "sat")])]
        bench_paths = ["bench1.smt2"]
        precomputed = {"bench1.smt2": np.array([1.0, 2.0])}
        selector = train_pwc_selector(
            shortlist, bench_paths, timeout=10, precomputed_features=precomputed
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "selector.joblib"
            selector.save(out_path)
            loaded = PairwiseSelector.load(out_path)
        self.assertEqual(loaded.select_vec(np.array([5.0, 5.0])), "strat_a")


if __name__ == "__main__":
    unittest.main()
