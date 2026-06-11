"""Tests for the SMT-COMP z3alpha2 submission entry point."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
import unittest.mock
from pathlib import Path

from tests.z3alpha2_submission_cases import (
    REQUIRED_SELECTOR_LOGICS,
    benchmark_path,
    check_case,
    load_cases,
    resolve_benchmark,
    run_case,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SUBMISSION = _PROJECT_ROOT / "data" / "smtcomp26" / "submission"
_Z3ALPHA2 = _SUBMISSION / "z3alpha2.py"
_Z3 = _SUBMISSION / "bin" / "z3"


def _load_z3alpha2():
    spec = importlib.util.spec_from_file_location("z3alpha2_submission", _Z3ALPHA2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@unittest.skipUnless(_Z3ALPHA2.is_file(), "submission z3alpha2.py not found")
class TestZ3alpha2Registry(unittest.TestCase):
    """Sanity-check the selector registry loaded from the submission directory."""

    def test_all_submission_logics_registered(self) -> None:
        mod = _load_z3alpha2()
        registry = mod.load_selector_registry(_SUBMISSION / "selectors")
        missing = [logic for logic in REQUIRED_SELECTOR_LOGICS if logic not in registry]
        self.assertEqual(missing, [])

    def test_lia_is_fixed_strategy_not_selector(self) -> None:
        mod = _load_z3alpha2()
        registry = mod.load_selector_registry(_SUBMISSION / "selectors")
        selector_dir, meta = registry["LIA"]
        self.assertEqual(meta["logic"], "LIA")
        self.assertIn("fixed_strategy", meta)
        self.assertFalse((selector_dir / "selector.joblib").exists())


# ---------------------------------------------------------------------------
# Behavior cases (data/smoke/benchmarks/cases.json)
# ---------------------------------------------------------------------------

def _skip_if_no_z3(cls):
    return unittest.skipUnless(_Z3.is_file(), "submission bin/z3 not found")(cls)


def _cases_for(category: str) -> list[dict]:
    return [c for c in load_cases() if c.get("category") == category]


def _available(cases: list[dict]) -> list[dict]:
    return [c for c in cases if benchmark_path(c) is not None]


def _run_cases(test: unittest.TestCase, cases: list[dict]) -> None:
    available = _available(cases)
    if not available:
        test.skipTest("no benchmarks available (configure smtlib_root or use bundled fixtures)")
    for case in available:
        with test.subTest(case=case["name"], logic=case["logic"]):
            rc, stdout, stderr = run_case(_Z3ALPHA2, _SUBMISSION, case)
            errors = check_case(case, stderr, stdout, rc)
            test.assertEqual(errors, [], msg="\n".join(errors) + f"\nstderr:\n{stderr}")


@_skip_if_no_z3
class TestZ3alpha2FixedStrategy(unittest.TestCase):
    """LIA uses a hardcoded tactic; presolver and selector must not run."""

    def test_fixed_strategy(self) -> None:
        _run_cases(self, _cases_for("fixed_strategy"))


@_skip_if_no_z3
class TestZ3alpha2PresolverSolved(unittest.TestCase):
    """Plain Z3 presolver finds sat/unsat within the presolver budget; selector never runs."""

    def test_presolver_solved(self) -> None:
        _run_cases(self, _cases_for("presolver_solved"))


@_skip_if_no_z3
class TestZ3alpha2PresolverOrSelector(unittest.TestCase):
    """Either path (presolver or selector) is valid depending on benchmark difficulty."""

    def test_presolver_or_selector(self) -> None:
        _run_cases(self, _cases_for("presolver_or_selector"))


@_skip_if_no_z3
class TestZ3alpha2SelectorWithScheduler(unittest.TestCase):
    """ML selector picks rank-1 strategy; scheduler gives it a timeout before rank-2."""

    def test_selector_scheduler(self) -> None:
        _run_cases(self, _cases_for("selector_scheduler"))


@_skip_if_no_z3
class TestZ3alpha2SelectorNoScheduler(unittest.TestCase):
    """ML selector runs without a scheduler (some logics disable it)."""

    def test_selector_no_scheduler(self) -> None:
        _run_cases(self, _cases_for("selector_no_scheduler"))


@_skip_if_no_z3
class TestZ3alpha2SelectorMLReordering(unittest.TestCase):
    """ML ranks a strategy above the shortlist default, verifying pairwise reordering."""

    def test_selector_ml_reordering(self) -> None:
        _run_cases(self, _cases_for("selector_ml_reordering"))


@_skip_if_no_z3
class TestZ3alpha2SchedulerFires(unittest.TestCase):
    """Rank-1 returns unknown quickly; scheduler fires and invokes rank-2.

    The process is killed after the per-case timeout — only debug lines emitted
    before the kill are checked; a final sat/unsat verdict is not required.
    """

    def test_scheduler_fires_rank2(self) -> None:
        _run_cases(self, _cases_for("scheduler_fires"))


@_skip_if_no_z3
class TestZ3alpha2UnmappedLogic(unittest.TestCase):
    """Logics not in the selector registry fall back to plain Z3."""

    def test_unmapped_logic_default_z3(self) -> None:
        _run_cases(self, _cases_for("unmapped_logic"))


# ---------------------------------------------------------------------------
# Solve dispatch unit tests (mocked run_z3, no real benchmark needed)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_Z3ALPHA2.is_file(), "submission z3alpha2.py not found")
class TestZ3alpha2SolveDispatch(unittest.TestCase):
    """Unit tests for solve() scheduler and strategy-dispatch logic.

    run_z3 is patched so no real solver or benchmark file is needed.
    """

    def _mod(self):
        return _load_z3alpha2()

    def test_rank1_solves_exits_immediately(self):
        """solve() calls sys.exit(0) after rank-1 succeeds; rank-2 is never invoked."""
        mod = self._mod()
        calls = []

        def mock_run_z3(params, benchmark, timeout=None, *, capture=False):
            calls.append({"params": params, "timeout": timeout, "capture": capture})
            return True if capture else None

        with unittest.mock.patch.object(mod, "run_z3", mock_run_z3):
            with self.assertRaises(SystemExit) as cm:
                mod.solve(["A", "B"], {"A": ["--opt-a"], "B": ["--opt-b"]},
                          False, 30.0, Path("/fake.smt2"))

        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(len(calls), 1, "rank-2 must not be called when rank-1 solves")
        self.assertEqual(calls[0]["params"], ["--opt-a"])

    def test_scheduler_timeout_applied_to_rank1_only(self):
        """With scheduler=True, rank-1 receives scheduler_timeout; rank-2 gets None."""
        mod = self._mod()
        calls = []

        def mock_run_z3(params, benchmark, timeout=None, *, capture=False):
            calls.append({"timeout": timeout, "capture": capture})
            if capture:
                return len(calls) >= 2  # rank-1 fails, rank-2 succeeds
            return None

        with unittest.mock.patch.object(mod, "run_z3", mock_run_z3):
            with self.assertRaises(SystemExit):
                mod.solve(["A", "B"], {"A": [], "B": []}, True, 42.0, Path("/fake.smt2"))

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["timeout"], 42.0, "rank-1 must get scheduler_timeout")
        self.assertIsNone(calls[1]["timeout"], "rank-2 must have no timeout")

    def test_no_scheduler_all_ranks_have_no_timeout(self):
        """With scheduler=False, every rank runs without a timeout."""
        mod = self._mod()
        calls = []

        def mock_run_z3(params, benchmark, timeout=None, *, capture=False):
            calls.append({"timeout": timeout, "capture": capture})
            if capture:
                return len(calls) >= 2
            return None

        with unittest.mock.patch.object(mod, "run_z3", mock_run_z3):
            with self.assertRaises(SystemExit):
                mod.solve(["A", "B"], {"A": [], "B": []}, False, 42.0, Path("/fake.smt2"))

        self.assertIsNone(calls[0]["timeout"])
        self.assertIsNone(calls[1]["timeout"])

    def test_rank2_tried_when_rank1_returns_unknown(self):
        """rank-2 is invoked when rank-1 returns unknown (run_z3 returns False)."""
        mod = self._mod()
        calls = []

        def mock_run_z3(params, benchmark, timeout=None, *, capture=False):
            calls.append(params)
            if capture:
                return len(calls) >= 2  # rank-1 → False, rank-2 → True
            return None

        with unittest.mock.patch.object(mod, "run_z3", mock_run_z3):
            with self.assertRaises(SystemExit) as cm:
                mod.solve(["A", "B"], {"A": ["--opt-a"], "B": ["--opt-b"]},
                          True, 42.0, Path("/fake.smt2"))

        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(len(calls), 2, "rank-2 must be tried after rank-1 fails")
        self.assertEqual(calls[0], ["--opt-a"])
        self.assertEqual(calls[1], ["--opt-b"])

    def test_fallback_z3_when_all_strategies_exhausted(self):
        """Plain Z3 (capture=False) is called after every ranked strategy fails."""
        mod = self._mod()
        calls = []

        def mock_run_z3(params, benchmark, timeout=None, *, capture=False):
            calls.append({"params": params, "capture": capture})
            return False if capture else None

        with unittest.mock.patch.object(mod, "run_z3", mock_run_z3):
            with self.assertRaises(SystemExit):
                mod.solve(["A", "B"], {"A": [], "B": []}, False, 30.0, Path("/fake.smt2"))

        # 2 strategy calls (capture=True) + 1 fallback call (capture=False)
        self.assertEqual(len(calls), 3)
        last = calls[-1]
        self.assertFalse(last["capture"], "fallback must be a non-capturing call")
        self.assertEqual(last["params"], [], "fallback must use no extra params")


# ---------------------------------------------------------------------------
# Opt-in integration: scheduler actually fires on a real benchmark
# ---------------------------------------------------------------------------

@unittest.skipUnless(_Z3.is_file(), "submission bin/z3 not found")
@unittest.skipUnless(
    os.environ.get("RUN_SCHEDULER_FIRES_TESTS") == "1",
    "set RUN_SCHEDULER_FIRES_TESTS=1 to enable (takes ~15 s per case)",
)
class TestZ3alpha2SchedulerFiresIntegration(unittest.TestCase):
    """Integration tests: the scheduler actually fires and rank-2 is invoked.

    These run z3alpha2 against real SMT-LIB benchmarks where rank-1 returns
    'unknown' quickly (not via timeout), triggering rank-2.  z3alpha2 is killed
    after a short budget; only the debug lines printed before the kill matter.

    Enable:
        RUN_SCHEDULER_FIRES_TESTS=1 python -m pytest \\
            tests/test_z3alpha2_submission.py::TestZ3alpha2SchedulerFiresIntegration -v

    Benchmark: QF_NIA/leipzig/term-unsat-02.smt2
      - presolver (8 s budget) times out on this hard instance
      - rank-1  H16_nla2bv64_qfbv  returns 'unknown' in <1 s
      - rank-2  H11_onlypoly_noreorder  is then invoked by the scheduler
    """

    _KILL_AFTER = 15
    _BENCH_REL = "non-incremental/QF_NIA/leipzig/term-unsat-02.smt2"

    def _run_partial(self, bench: Path) -> str:
        try:
            proc = subprocess.run(
                [sys.executable, str(_Z3ALPHA2), "--debug", str(bench)],
                cwd=_SUBMISSION,
                capture_output=True,
                timeout=self._KILL_AFTER,
            )
            stderr_bytes = proc.stderr
        except subprocess.TimeoutExpired as exc:
            stderr_bytes = exc.stderr or b""
        return stderr_bytes.decode("utf-8", errors="replace")

    def test_qf_nia_rank2_invoked_when_rank1_returns_unknown(self):
        bench = resolve_benchmark(self._BENCH_REL)
        if bench is None:
            self.skipTest(f"benchmark not available: {self._BENCH_REL}")

        stderr = self._run_partial(bench)

        self.assertIn(
            "rank-1: H16_nla2bv64_qfbv", stderr,
            "rank-1 strategy log line missing — selector or debug output changed",
        )
        self.assertIn(
            "rank-2: H11_onlypoly_noreorder", stderr,
            "rank-2 was never invoked: scheduler did not fire after rank-1 returned unknown",
        )
        self.assertNotIn("Z3 timed out\n[z3alpha2] rank-2", stderr,
                         "rank-1 appeared to time out rather than return unknown")


if __name__ == "__main__":
    unittest.main()
