"""Tests for the SMT-COMP z3alpha2 submission entry point."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SUBMISSION = _PROJECT_ROOT / "data" / "smtcomp26" / "submission"
_Z3ALPHA2 = _SUBMISSION / "z3alpha2.py"
_Z3 = _SUBMISSION / "bin" / "z3"

# SMT-COMP 2025 LIA track; same family as data/smtcomp26/smtcomp25_benchlist/LIA.txt
LIA_BENCH_REL = (
    "non-incremental/LIA/20190429-UltimateAutomizerSvcomp2019/"
    "jain_7_true-unreach-call_true-no-overflow_false-termination.i_10.smt2"
)
LIA_BENCH_FALLBACK = (
    _PROJECT_ROOT / "data" / "smoke" / "benchmarks" / "lia" /
    "jain_7_true-unreach-call_true-no-overflow_false-termination.i_10.smt2"
)

# SMT-COMP 2025 LRA track; same family as data/smtcomp26/smtcomp25_benchlist/LRA.txt
LRA_BENCH_REL = "non-incremental/LRA/2010-Monniaux-QE/mjollnir1/formula_014.smt2"
LRA_BENCH_FALLBACK = _PROJECT_ROOT / "data" / "smoke" / "benchmarks" / "lra" / "formula_014.smt2"


def _smtlib_root() -> Path | None:
    config_path = _PROJECT_ROOT / "env_config.json"
    if config_path.is_file():
        root = json.loads(config_path.read_text()).get("smtlib_root")
        if root:
            return Path(root)
    env = __import__("os").environ.get("SMTLIB_ROOT")
    return Path(env) if env else None


def _resolve_benchmark(rel: str, fallback: Path) -> Path | None:
    root = _smtlib_root()
    if root is not None:
        candidate = root / rel
        if candidate.is_file():
            return candidate
    if fallback.is_file():
        return fallback
    return None


def _lia_benchmark() -> Path | None:
    return _resolve_benchmark(LIA_BENCH_REL, LIA_BENCH_FALLBACK)


def _lra_benchmark() -> Path | None:
    return _resolve_benchmark(LRA_BENCH_REL, LRA_BENCH_FALLBACK)


def _load_z3alpha2():
    spec = importlib.util.spec_from_file_location("z3alpha2_submission", _Z3ALPHA2)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(_Z3ALPHA2.is_file(), "submission z3alpha2.py not found")
class TestZ3alpha2Registry(unittest.TestCase):
    def test_lia_registered_as_fixed_strategy(self) -> None:
        mod = _load_z3alpha2()
        registry = mod.load_selector_registry(_SUBMISSION / "selectors")
        self.assertIn("LIA", registry)
        _dir, meta = registry["LIA"]
        self.assertEqual(meta["logic"], "LIA")
        self.assertEqual(
            meta["fixed_strategy"],
            ["tactic.default_tactic=(then qe_rec smt)"],
        )
        self.assertFalse((_dir / "selector.joblib").exists())

    def test_lra_registered_with_selector(self) -> None:
        mod = _load_z3alpha2()
        registry = mod.load_selector_registry(_SUBMISSION / "selectors")
        self.assertIn("LRA", registry)
        selector_dir, meta = registry["LRA"]
        self.assertEqual(meta["logic"], "LRA")
        self.assertFalse(meta.get("scheduler"))
        self.assertEqual(meta.get("presolver_seconds"), 0.0)
        self.assertIn("shortlist", meta)
        self.assertTrue((selector_dir / "selector.joblib").is_file())
        self.assertNotIn("fixed_strategy", meta)


@unittest.skipUnless(_Z3.is_file(), "submission bin/z3 not found")
@unittest.skipUnless(_lia_benchmark() is not None, "LIA benchmark not available")
class TestZ3alpha2LIA(unittest.TestCase):
    def test_lia_runs_fixed_strategy_on_smtlib_benchmark(self) -> None:
        benchmark = _lia_benchmark()
        assert benchmark is not None

        proc = subprocess.run(
            [sys.executable, str(_Z3ALPHA2), "--debug", str(benchmark)],
            cwd=_SUBMISSION,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        verdict = proc.stdout.strip().split("\n")[0]
        self.assertIn(verdict, ("sat", "unsat"))
        self.assertIn("logic=LIA", proc.stderr)
        self.assertIn("fixed_strategy=", proc.stderr)
        self.assertNotIn("presolver solved", proc.stderr)
        self.assertNotIn("ranked=", proc.stderr)

    def test_lia_benchmark_is_lia_logic(self) -> None:
        benchmark = _lia_benchmark()
        assert benchmark is not None
        mod = _load_z3alpha2()
        self.assertEqual(mod.detect_logic(benchmark), "LIA")


@unittest.skipUnless(_Z3.is_file(), "submission bin/z3 not found")
@unittest.skipUnless(_lra_benchmark() is not None, "LRA benchmark not available")
class TestZ3alpha2LRA(unittest.TestCase):
    def test_lra_runs_selector_on_smtlib_benchmark(self) -> None:
        benchmark = _lra_benchmark()
        assert benchmark is not None

        proc = subprocess.run(
            [sys.executable, str(_Z3ALPHA2), "--debug", str(benchmark)],
            cwd=_SUBMISSION,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        verdict = proc.stdout.strip().split("\n")[0]
        self.assertIn(verdict, ("sat", "unsat"))
        self.assertIn("logic=LRA", proc.stderr)
        self.assertIn("ranked=", proc.stderr)
        self.assertIn("scheduler=False", proc.stderr)
        self.assertNotIn("presolver solved", proc.stderr)
        self.assertNotIn("fixed_strategy=", proc.stderr)

    def test_lra_benchmark_is_lra_logic(self) -> None:
        benchmark = _lra_benchmark()
        assert benchmark is not None
        mod = _load_z3alpha2()
        self.assertEqual(mod.detect_logic(benchmark), "LRA")


if __name__ == "__main__":
    unittest.main()
