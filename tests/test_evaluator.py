"""Unit tests for z3alpha.evaluator (SolverRunner collect() error paths)."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from z3alpha.evaluator import SolverRunner, _z3_timeout_arg

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# strategy=None so runner uses path as-is; we mock Popen so solver is not executed.
SAMPLE_SMT = str(_PROJECT_ROOT / "data" / "sample" / "benchmarks" / "0.smt2")


def _run_then_collect(runner: SolverRunner, communicate_return: tuple[bytes, bytes]) -> tuple:
    """Start runner with mocked Popen, let run() finish, then collect()."""
    mock_process = MagicMock()
    mock_process.wait = MagicMock(return_value=None)
    mock_process.communicate = MagicMock(return_value=communicate_return)

    with patch("z3alpha.evaluator.subprocess.Popen", return_value=mock_process):
        runner.start()
        runner.join()
    return runner.collect()


class TestSolverRunnerCollect(unittest.TestCase):
    """Test SolverRunner.collect() with various solver outputs."""

    def test_collect_empty_stdout_returns_error(self) -> None:
        """Empty stdout should return result with 'error' and not raise."""
        runner = SolverRunner(
            "/usr/bin/z3",
            SAMPLE_SMT,
            timeout=10,
            run_id=0,
            strategy=None,
        )
        run_id, res, runtime, path = _run_then_collect(runner, (b"", b""))
        self.assertEqual(run_id, 0)
        self.assertEqual(res, "error")
        self.assertIsInstance(runtime, (int, float))
        self.assertEqual(path, SAMPLE_SMT)

    def test_collect_none_stdout_returns_error(self) -> None:
        """None stdout (e.g. no PIPE) should return 'error' and not raise."""
        runner = SolverRunner(
            "/usr/bin/z3",
            SAMPLE_SMT,
            timeout=10,
            run_id=1,
            strategy=None,
        )
        run_id, res, runtime, path = _run_then_collect(runner, (None, b""))
        self.assertEqual(run_id, 1)
        self.assertEqual(res, "error")
        self.assertEqual(path, SAMPLE_SMT)

    def test_collect_no_lines_returns_error(self) -> None:
        """Output with only newlines (no first line) should return 'error'."""
        runner = SolverRunner(
            "/usr/bin/z3",
            SAMPLE_SMT,
            timeout=10,
            run_id=2,
            strategy=None,
        )
        run_id, res, runtime, path = _run_then_collect(runner, (b"\n\n", b""))
        self.assertEqual(run_id, 2)
        self.assertEqual(res, "error")
        self.assertEqual(path, SAMPLE_SMT)

    def test_collect_normal_sat_returns_sat(self) -> None:
        """Normal 'sat' output should return result 'sat'."""
        runner = SolverRunner(
            "/usr/bin/z3",
            SAMPLE_SMT,
            timeout=10,
            run_id=3,
            strategy=None,
        )
        run_id, res, runtime, path = _run_then_collect(runner, (b"sat\n", b""))
        self.assertEqual(run_id, 3)
        self.assertEqual(res, "sat")
        self.assertIsInstance(runtime, (int, float))
        self.assertEqual(path, SAMPLE_SMT)

    def test_collect_normal_unsat_returns_unsat(self) -> None:
        """Normal 'unsat' output should return result 'unsat'."""
        runner = SolverRunner(
            "/usr/bin/z3",
            SAMPLE_SMT,
            timeout=10,
            run_id=4,
            strategy=None,
        )
        run_id, res, runtime, path = _run_then_collect(runner, (b"unsat\n", b""))
        self.assertEqual(run_id, 4)
        self.assertEqual(res, "unsat")
        self.assertEqual(path, SAMPLE_SMT)

    def test_collect_solver_error_line_returns_error(self) -> None:
        """First line starting with '(error' should return 'error'."""
        runner = SolverRunner(
            "/usr/bin/z3",
            SAMPLE_SMT,
            timeout=10,
            run_id=5,
            strategy=None,
        )
        run_id, res, runtime, path = _run_then_collect(
            runner, (b"(error \"something went wrong\")\n", b"")
        )
        self.assertEqual(run_id, 5)
        self.assertEqual(res, "error")
        self.assertEqual(path, SAMPLE_SMT)


class TestZ3TimeoutArg(unittest.TestCase):
    """Test _z3_timeout_arg helper."""

    def test_returns_T_flag_seconds(self) -> None:
        self.assertEqual(_z3_timeout_arg(10), ["-T:10"])
        self.assertEqual(_z3_timeout_arg(300), ["-T:300"])
        self.assertEqual(_z3_timeout_arg(1.7), ["-T:1"])
