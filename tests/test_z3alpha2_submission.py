"""Tests for the SMT-COMP z3alpha2 submission entry point."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from tests.z3alpha2_submission_cases import (
    REQUIRED_SELECTOR_LOGICS,
    benchmark_path,
    check_case,
    load_cases,
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


@unittest.skipUnless(_Z3ALPHA2.is_file(), "submission z3alpha2.py not found")
class TestZ3alpha2Registry(unittest.TestCase):
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


@unittest.skipUnless(_Z3.is_file(), "submission bin/z3 not found")
@unittest.skipUnless(
    any(benchmark_path(c) is not None for c in load_cases()),
    "no benchmarks available (configure smtlib_root or use bundled fixtures)",
)
class TestZ3alpha2Behaviors(unittest.TestCase):
    def test_smtlib_logic_benchmark_cases(self) -> None:
        available = [c for c in load_cases() if benchmark_path(c) is not None]
        self.assertGreater(len(available), 0, "no SMT-LIB benchmarks found")

        for case in available:
            with self.subTest(case=case["name"], logic=case["logic"]):
                rc, stdout, stderr = run_case(_Z3ALPHA2, _SUBMISSION, case)
                errors = check_case(case, stderr, stdout, rc)
                self.assertEqual(errors, [], msg="\n".join(errors) + f"\nstderr:\n{stderr}")


if __name__ == "__main__":
    unittest.main()
