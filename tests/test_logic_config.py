"""Tests for z3alpha.tactics.logic_config: JSON parsing and a full
logic-config -> strategy-tree -> z3 execution pipeline.
"""
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from z3alpha.evaluator import SolverRunner
from z3alpha.strategy_tree import LinearStrategyTree
from z3alpha.tactics.catalog import NAME_TO_ID
from z3alpha.tactics.logic_config import load_logic_config

_Z3_PATH = shutil.which("z3")
# trivially unsat over integers (x*x is never both >= 0 and < 0); kept tiny
# so the solver returns almost instantly regardless of which tactic is used.
_TINY_UNSAT_NIA = (
    "(set-logic QF_NIA)\n"
    "(declare-fun x () Int)\n"
    "(assert (>= (* x x) 0))\n"
    "(assert (< (* x x) 0))\n"
    "(check-sat)\n"
)


def _write_smt2(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".smt2", delete=False)
    f.write(content)
    f.close()
    return f.name


def _write_config(tmp_dir: str, name: str, body: dict) -> None:
    (Path(tmp_dir) / f"{name}.json").write_text(json.dumps(body))


class TestLoadLogicConfig:
    def test_qf_nia_solver_and_preprocess_ids(self) -> None:
        cfg = load_logic_config("QF_NIA")
        assert cfg["solver_tactics"] == [
            NAME_TO_ID["smt"],
            NAME_TO_ID["qfnra-nlsat"],
            NAME_TO_ID["qfnia"],
        ]
        assert NAME_TO_ID["simplify"] in cfg["preprocess_tactics"]
        assert NAME_TO_ID["nla2bv"] in cfg["preprocess_tactics"]

    def test_qf_nia_params_keyed_by_id(self) -> None:
        cfg = load_logic_config("QF_NIA")
        smt_id = NAME_TO_ID["smt"]
        assert cfg["params"][smt_id] == {
            "random_seed": {"default": 0, "values": [0, 100, 200, 300, 400, 500]}
        }
        # tactics with no params entry (e.g. "qfnia") are absent from the dict
        assert NAME_TO_ID["qfnia"] not in cfg["params"]

    def test_unknown_logic_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_logic_config("NOT_A_REAL_LOGIC")

    def test_missing_solver_field_defaults_to_preprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_config(tmp, "FOO", {"simplify": {}})
            cfg = load_logic_config("FOO", config_dir=tmp)
        assert cfg["solver_tactics"] == []
        assert cfg["preprocess_tactics"] == [NAME_TO_ID["simplify"]]
        assert cfg["params"] == {}

    def test_missing_params_field_omitted_from_params_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_config(tmp, "FOO", {"smt": {"solver": True}})
            cfg = load_logic_config("FOO", config_dir=tmp)
        assert cfg["solver_tactics"] == [NAME_TO_ID["smt"]]
        assert cfg["params"] == {}

    def test_explicit_solver_and_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_config(
                tmp,
                "FOO",
                {
                    "smt": {
                        "solver": True,
                        "params": {"random_seed": {"default": 0, "values": [1, 2]}},
                    }
                },
            )
            cfg = load_logic_config("FOO", config_dir=tmp)
        smt_id = NAME_TO_ID["smt"]
        assert cfg["solver_tactics"] == [smt_id]
        assert cfg["params"][smt_id] == {"random_seed": {"default": 0, "values": [1, 2]}}

    def test_override_dir_takes_precedence_over_builtin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_config(tmp, "QF_NIA", {"smt": {"solver": True}})
            cfg = load_logic_config("QF_NIA", config_dir=tmp)
        assert cfg["solver_tactics"] == [NAME_TO_ID["smt"]]
        assert cfg["preprocess_tactics"] == []

    def test_falls_back_to_builtin_when_override_file_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_logic_config("QF_NIA", config_dir=tmp)
        assert cfg == load_logic_config("QF_NIA")


class TestLogicConfigStrategyTreeEndToEnd:
    """Build a strategy from a real logic config and run it through z3."""

    def _params_for(self, cfg: dict, action: int) -> dict | None:
        grid = cfg["params"].get(action)
        if not grid:
            return None
        return {name: spec["default"] for name, spec in grid.items()}

    def test_strategy_string_includes_params_from_config(self) -> None:
        cfg = load_logic_config("QF_NIA")
        tree = LinearStrategyTree("QF_NIA", timeout=10, logic_config=cfg)

        simplify_id = NAME_TO_ID["simplify"]
        smt_id = NAME_TO_ID["smt"]
        tree.apply_rule(simplify_id, self._params_for(cfg, simplify_id))
        tree.apply_rule(smt_id, self._params_for(cfg, smt_id))

        assert tree.is_terminal()
        strat_str = str(tree)
        assert "simplify" in strat_str
        assert "smt" in strat_str
        assert ":random_seed 0" in strat_str  # the config's declared z3 default

    @pytest.mark.skipif(_Z3_PATH is None, reason="z3 binary not found on PATH")
    def test_synthesized_strategy_solves_real_benchmark(self) -> None:
        cfg = load_logic_config("QF_NIA")
        tree = LinearStrategyTree("QF_NIA", timeout=10, logic_config=cfg)

        simplify_id = NAME_TO_ID["simplify"]
        qfnia_id = NAME_TO_ID["qfnia"]
        tree.apply_rule(simplify_id, self._params_for(cfg, simplify_id))
        tree.apply_rule(qfnia_id, self._params_for(cfg, qfnia_id))
        assert tree.is_terminal()

        strat_str = str(tree)
        bench_path = _write_smt2(_TINY_UNSAT_NIA)
        runner = SolverRunner(
            _Z3_PATH, bench_path, timeout=10, run_id=0, z3_strategy=strat_str
        )
        run_id, res, runtime, path = runner.execute()
        assert run_id == 0
        assert path == bench_path
        assert res == "unsat"
