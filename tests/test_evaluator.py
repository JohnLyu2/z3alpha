"""Unit tests for z3alpha.evaluator (SolverRunner execute() and output parsing)."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from z3alpha.evaluator import SolverRunner

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SMT = str(_PROJECT_ROOT / "data" / "sample" / "benchmarks" / "0.smt2")


def _execute_with_mock(runner: SolverRunner, communicate_return: tuple[bytes, bytes]) -> tuple:
    """Execute runner with mocked Popen so no real solver is invoked."""
    mock_process = MagicMock()
    mock_process.communicate = MagicMock(return_value=communicate_return)

    with patch("z3alpha.evaluator.subprocess.Popen", return_value=mock_process):
        return runner.execute()


class TestSolverRunnerExecute(unittest.TestCase):
    """Test SolverRunner.execute() with various solver outputs."""

    def test_empty_stdout_returns_error(self) -> None:
        runner = SolverRunner("/usr/bin/z3", SAMPLE_SMT, timeout=10, run_id=0)
        run_id, res, runtime, path = _execute_with_mock(runner, (b"", b""))
        self.assertEqual(run_id, 0)
        self.assertEqual(res, "error")
        self.assertIsInstance(runtime, (int, float))
        self.assertEqual(path, SAMPLE_SMT)

    def test_none_stdout_returns_error(self) -> None:
        runner = SolverRunner("/usr/bin/z3", SAMPLE_SMT, timeout=10, run_id=1)
        run_id, res, runtime, path = _execute_with_mock(runner, (None, b""))
        self.assertEqual(run_id, 1)
        self.assertEqual(res, "error")
        self.assertEqual(path, SAMPLE_SMT)

    def test_no_lines_returns_error(self) -> None:
        runner = SolverRunner("/usr/bin/z3", SAMPLE_SMT, timeout=10, run_id=2)
        run_id, res, runtime, path = _execute_with_mock(runner, (b"\n\n", b""))
        self.assertEqual(run_id, 2)
        self.assertEqual(res, "error")
        self.assertEqual(path, SAMPLE_SMT)

    def test_normal_sat_returns_sat(self) -> None:
        runner = SolverRunner("/usr/bin/z3", SAMPLE_SMT, timeout=10, run_id=3)
        run_id, res, runtime, path = _execute_with_mock(runner, (b"sat\n", b""))
        self.assertEqual(run_id, 3)
        self.assertEqual(res, "sat")
        self.assertIsInstance(runtime, (int, float))
        self.assertEqual(path, SAMPLE_SMT)

    def test_normal_unsat_returns_unsat(self) -> None:
        runner = SolverRunner("/usr/bin/z3", SAMPLE_SMT, timeout=10, run_id=4)
        run_id, res, runtime, path = _execute_with_mock(runner, (b"unsat\n", b""))
        self.assertEqual(run_id, 4)
        self.assertEqual(res, "unsat")
        self.assertEqual(path, SAMPLE_SMT)

    def test_solver_error_line_returns_error(self) -> None:
        runner = SolverRunner("/usr/bin/z3", SAMPLE_SMT, timeout=10, run_id=5)
        run_id, res, runtime, path = _execute_with_mock(
            runner, (b"(error \"something went wrong\")\n", b"")
        )
        self.assertEqual(run_id, 5)
        self.assertEqual(res, "error")
        self.assertEqual(path, SAMPLE_SMT)

    def test_build_cmd_without_strategy(self) -> None:
        runner = SolverRunner("/usr/bin/z3", SAMPLE_SMT, timeout=10, run_id=0)
        cmd = runner._build_cmd()
        self.assertEqual(cmd, ["/usr/bin/z3", SAMPLE_SMT])

    def test_build_cmd_with_strategy(self) -> None:
        runner = SolverRunner(
            "/usr/bin/z3", SAMPLE_SMT, timeout=10, run_id=0,
            z3_strategy="(then simplify smt)",
        )
        cmd = runner._build_cmd()
        self.assertEqual(
            cmd,
            ["/usr/bin/z3", "tactic.default_tactic=(then simplify smt)", SAMPLE_SMT],
        )
