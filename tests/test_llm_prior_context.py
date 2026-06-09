"""Tests for same-run LLM prior context formatting (V1)."""

from __future__ import annotations

import unittest

import z3alpha.config  # noqa: F401 — ensure package init order before mcts submodules

from z3alpha.mcts.llm_prior_context import (
    RunContextVersion,
    build_strategy_context_rows,
    compute_run_context_version,
    format_run_context,
    select_strategies_for_context,
    StrategyContextRow,
)


def _res(solved_flags: list[bool], timeout: float = 10.0) -> list[tuple]:
    return [
        (s, 1.0 if s else timeout, "sat" if s else "timeout") for s in solved_flags
    ]


class TestComputeRunContextVersion(unittest.TestCase):
    def test_empty_database(self) -> None:
        v = compute_run_context_version({})
        self.assertEqual(v, RunContextVersion(num_strategies=0, best_n_solved=0))

    def test_nonempty(self) -> None:
        db = {
            "a": _res([True, True, False]),
            "b": _res([True, False, False]),
        }
        v = compute_run_context_version(db)
        self.assertEqual(v.num_strategies, 2)
        self.assertEqual(v.best_n_solved, 2)


class TestSelectStrategiesForContext(unittest.TestCase):
    def _rows(self, n: int) -> list[StrategyContextRow]:
        return [
            StrategyContextRow(strategy=f"s{i}", n_solved=i, par10_avg=float(i))
            for i in range(n - 1, -1, -1)
        ]

    def test_all_when_at_most_20(self) -> None:
        rows = self._rows(12)
        selected = select_strategies_for_context(rows)
        self.assertEqual(len(selected), 12)
        self.assertEqual([r.strategy for r in selected], [f"s{i}" for i in range(11, -1, -1)])

    def test_top_15_bottom_5_when_over_20(self) -> None:
        rows = self._rows(25)
        selected = select_strategies_for_context(rows)
        self.assertEqual(len(selected), 20)
        strategies = [r.strategy for r in selected]
        self.assertEqual(strategies[:15], [f"s{i}" for i in range(24, 9, -1)])
        self.assertEqual(strategies[15:], [f"s{i}" for i in range(4, -1, -1)])


class TestBuildStrategyContextRows(unittest.TestCase):
    def test_sort_order(self) -> None:
        db = {
            "low": _res([False, False]),
            "high": _res([True, True]),
            "mid": _res([True, False]),
        }
        rows = build_strategy_context_rows(db, timeout=10.0)
        self.assertEqual([r.strategy for r in rows], ["high", "mid", "low"])

    def test_par10_avg(self) -> None:
        db = {"s": _res([True, False], timeout=10.0)}
        rows = build_strategy_context_rows(db, timeout=10.0)
        self.assertEqual(len(rows), 1)
        # par10: solved 1.0 + unsolved 100 = 101 / 2
        self.assertAlmostEqual(rows[0].par10_avg, 50.5)


class TestFormatRunContext(unittest.TestCase):
    def test_empty_database(self) -> None:
        text, version = format_run_context({}, timeout=10.0, sim_id=0)
        self.assertEqual(version, RunContextVersion(0, 0))
        self.assertIn("(no strategies evaluated yet)", text)
        self.assertIn("sim 0", text)
        self.assertIn("best-performing strategies", text)

    def test_includes_all_rows_up_to_20(self) -> None:
        db = {f"s{i}": _res([True] * i + [False] * (10 - i)) for i in range(8)}
        text, version = format_run_context(db, timeout=10.0, sim_id=5)
        self.assertEqual(version.num_strategies, 8)
        for i in range(8):
            self.assertIn(f"s{i}", text)

    def test_caps_at_20_rows(self) -> None:
        n_bench = 30
        db = {
            f"s{i}": _res([True] * i + [False] * (n_bench - i))
            for i in range(25)
        }
        text, version = format_run_context(db, timeout=10.0, sim_id=30)
        self.assertEqual(version.num_strategies, 25)
        line_count = sum(
            1 for line in text.splitlines() if line.strip().startswith("n_solved=")
        )
        self.assertEqual(line_count, 20)
        self.assertIn("s24", text)
        self.assertIn("s0", text)
        self.assertNotIn(" s7", text)
        self.assertNotIn(" s8", text)


if __name__ == "__main__":
    unittest.main()
