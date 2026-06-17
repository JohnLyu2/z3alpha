"""Tests for z3alpha.mcts.param_selection: MabParamSelector and config wiring."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from z3alpha.config import MctsConfig, parse_experiment_config, resolve_mcts_config
from z3alpha.mcts.linear import LinearStrategySearchRun
from z3alpha.mcts.param_selection import (
    DEFAULT_PARAM_C_UCB,
    MabParamSelector,
    ParamSelectionConfig,
    ParamSelector,
)

# ---------------------------------------------------------------------------
# MabParamSelector unit tests
# ---------------------------------------------------------------------------

_GRID = {"random_seed": {"default": 0, "values": [0, 100, 200]}}


def _make_selector() -> MabParamSelector:
    return MabParamSelector(c_ucb=DEFAULT_PARAM_C_UCB, is_mean=False)


def test_select_returns_value_from_grid():
    sel = _make_selector()
    chosen = sel.select("QF_NIA", "(then simplify)", "smt", _GRID)
    assert "random_seed" in chosen
    assert chosen["random_seed"] in [0, 100, 200]


def test_arms_seeded_from_values_not_unioned_with_default():
    """Arms must come from 'values' only, not extended by 'default'."""
    grid = {"p": {"default": 99, "values": [1, 2, 3]}}
    sel = _make_selector()
    seen: set = set()
    for _ in range(30):
        chosen = sel.select("L", "t", "tac", grid)
        sel.backup_episode(1.0)
        seen.add(chosen["p"])
    # 99 (the default, not in 'values') must never appear
    assert 99 not in seen
    assert seen <= {1, 2, 3}


def test_unvisited_arms_tried_before_exploitation():
    """UCB1: unvisited arms get +inf, so all arms are tried before exploitation begins."""
    grid = {"p": {"default": "a", "values": ["a", "b", "c"]}}
    sel = MabParamSelector(c_ucb=1.0, is_mean=False)
    seen: list = []
    for _ in range(3):
        chosen = sel.select("L", "s", "t", grid)
        sel.backup_episode(0.5)
        seen.append(chosen["p"])
    assert set(seen) == {"a", "b", "c"}


def test_ucb1_converges_toward_higher_reward_arm():
    """After many iterations with consistent rewards, the higher-reward arm dominates."""
    grid = {"p": {"default": 0, "values": [0, 1]}}
    sel = MabParamSelector(c_ucb=0.1, is_mean=True)  # low c -> fast exploitation
    for _ in range(50):
        chosen = sel.select("L", "s", "t", grid)
        reward = 1.0 if chosen["p"] == 1 else 0.0
        sel.backup_episode(reward)
    counts = {0: 0, 1: 0}
    for _ in range(20):
        chosen = sel.select("L", "s", "t", grid)
        sel.backup_episode(1.0 if chosen["p"] == 1 else 0.0)
        counts[chosen["p"]] += 1
    assert counts[1] > counts[0], f"Expected arm 1 dominant, got {counts}"


def test_independent_keys_do_not_share_state():
    sel = _make_selector()
    grid = {"p": {"default": "x", "values": ["x", "y"]}}
    for _ in range(20):
        c = sel.select("L", "strategy_A", "tac", grid)
        sel.backup_episode(1.0 if c["p"] == "x" else 0.0)
    visits_b_before = sel._visits.get(("strategy_B", "tac"), 0)
    assert visits_b_before == 0
    sel.select("L", "strategy_B", "tac", grid)
    sel.backup_episode(0.5)
    assert sel._visits[("strategy_A", "tac")] == 20
    assert sel._visits[("strategy_B", "tac")] == 1


def test_pending_accumulates_across_multiple_selects_before_backup():
    """Selections from tree phase AND rollout phase both get backed up."""
    sel = _make_selector()
    grid_a = {"p": {"default": 0, "values": [0, 1]}}
    grid_b = {"q": {"default": "x", "values": ["x", "y"]}}
    sel.select("L", "s0", "tacA", grid_a)
    sel.select("L", "s1", "tacB", grid_b)
    assert len(sel._pending) == 2
    sel.backup_episode(0.8)
    assert len(sel._pending) == 0
    assert sel._visits[("s0", "tacA")] == 1
    assert sel._visits[("s1", "tacB")] == 1


def test_backup_episode_updates_visit_count_and_q():
    sel = MabParamSelector(c_ucb=0.5, is_mean=True)
    grid = {"p": {"default": 0, "values": [0, 1]}}
    chosen = sel.select("L", "s", "t", grid)
    val = chosen["p"]
    sel.backup_episode(0.6)
    n, q = sel._mabs[("s", "t")]["p"][val]
    assert n == 1
    assert q == pytest.approx(0.6)


def test_is_mean_false_uses_max():
    sel = MabParamSelector(c_ucb=0.0, is_mean=False)
    grid = {"p": {"default": 0, "values": [0]}}
    sel.select("L", "s", "t", grid)
    sel.backup_episode(0.3)
    sel.select("L", "s", "t", grid)
    sel.backup_episode(0.9)
    n, q = sel._mabs[("s", "t")]["p"][0]
    assert n == 2
    assert q == pytest.approx(0.9)  # max, not mean


def test_is_mean_true_uses_running_mean():
    sel = MabParamSelector(c_ucb=0.0, is_mean=True)
    grid = {"p": {"default": 0, "values": [0]}}
    sel.select("L", "s", "t", grid)
    sel.backup_episode(0.4)
    sel.select("L", "s", "t", grid)
    sel.backup_episode(0.8)
    n, q = sel._mabs[("s", "t")]["p"][0]
    assert n == 2
    assert q == pytest.approx(0.6)  # (0.4 + 0.8) / 2


def test_param_selector_protocol_check():
    sel = MabParamSelector(c_ucb=0.2, is_mean=False)
    assert isinstance(sel, ParamSelector)


# ---------------------------------------------------------------------------
# Config wiring tests
# ---------------------------------------------------------------------------

_MINIMAL_EXP = {
    "logic": "QF_NIA",
    "train_dir": "data/smoke/benchmarks",
    "timeout": 1,
    "mcts_sims": 3,
    "branched_sims": 1,
    "ln_strat_num": 1,
}


def test_resolve_mcts_config_param_search_defaults():
    class Args:
        c_uct = None
        random_seed = None

    experiment = parse_experiment_config(_MINIMAL_EXP)
    cfg = resolve_mcts_config(Args(), experiment)
    assert cfg.param_selector is not None
    assert cfg.param_selector.enabled is True
    assert cfg.param_selector.c_ucb == pytest.approx(DEFAULT_PARAM_C_UCB)


# ---------------------------------------------------------------------------
# params_for default-omission behaviour
# ---------------------------------------------------------------------------

def _make_run_with_selector(tmp_path, logic_config, mcts=None):
    """Return a LinearStrategySearchRun with param_selector enabled, env initialised."""
    if mcts is None:
        mcts = MctsConfig(
            sim_num=1, timeout=1, c_uct=0.5, random_seed=0,
            param_selector=ParamSelectionConfig(enabled=True, c_ucb=DEFAULT_PARAM_C_UCB),
        )
    log_dir = tmp_path / "run"
    log_dir.mkdir(exist_ok=True)
    with patch("z3alpha.environment.SolverEvaluator"):
        run = LinearStrategySearchRun(
            mcts, ["dummy.smt2"], "QF_NIA", "z3", "par10", log_dir,
            batch_size=1, logic_config=logic_config,
        )
        run.env = run._create_env()
        run.num_sim = 0
    return run


def test_params_for_omits_default_value(tmp_path):
    """When the MAB selects the default value, that param is absent from the result."""
    logic_config = {
        "solver_tactics": [10],
        "preprocess_tactics": [],
        "params": {10: {"random_seed": {"default": 0, "values": [0, 100]}}},
    }
    run = _make_run_with_selector(tmp_path, logic_config)
    with patch.object(run._param_selector, "select", return_value={"random_seed": 0}):
        result = run.params_for(10)
    assert result is None


def test_params_for_returns_none_when_all_selected_are_defaults(tmp_path):
    """When every selected param matches its default, params_for returns None (no using-params)."""
    logic_config = {
        "solver_tactics": [10],
        "preprocess_tactics": [],
        "params": {10: {
            "random_seed": {"default": 0, "values": [0, 100]},
            "factor": {"default": "true", "values": ["true", "false"]},
        }},
    }
    run = _make_run_with_selector(tmp_path, logic_config)
    with patch.object(run._param_selector, "select", return_value={"random_seed": 0, "factor": "true"}):
        result = run.params_for(10)
    assert result is None


def test_params_for_keeps_non_default_value(tmp_path):
    """A param value that differs from its default is included in the result."""
    logic_config = {
        "solver_tactics": [10],
        "preprocess_tactics": [],
        "params": {10: {"random_seed": {"default": 0, "values": [0, 100]}}},
    }
    run = _make_run_with_selector(tmp_path, logic_config)
    with patch.object(run._param_selector, "select", return_value={"random_seed": 100}):
        result = run.params_for(10)
    assert result == {"random_seed": 100}


def test_params_for_partial_default_omission(tmp_path):
    """Only non-default params survive; default ones are stripped."""
    logic_config = {
        "solver_tactics": [10],
        "preprocess_tactics": [],
        "params": {10: {
            "random_seed": {"default": 0, "values": [0, 100]},
            "factor": {"default": "true", "values": ["true", "false"]},
        }},
    }
    run = _make_run_with_selector(tmp_path, logic_config)
    with patch.object(run._param_selector, "select", return_value={"random_seed": 100, "factor": "true"}):
        result = run.params_for(10)
    assert result == {"random_seed": 100}


# ---------------------------------------------------------------------------
# Regression: params_for returns None when param_selector is disabled
# ---------------------------------------------------------------------------

def test_params_for_none_when_selector_disabled(tmp_path):
    mcts = MctsConfig(sim_num=1, timeout=1, c_uct=0.5, random_seed=0)
    logic_config = {
        "solver_tactics": [10],
        "preprocess_tactics": [],
        "params": {10: {"random_seed": {"default": 0, "values": [0, 100]}}},
    }
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    with patch("z3alpha.environment.SolverEvaluator"):
        run = LinearStrategySearchRun(
            mcts,
            ["dummy.smt2"],
            "QF_NIA",
            "z3",
            "par10",
            log_dir,
            batch_size=1,
            logic_config=logic_config,
        )
        run.env = run._create_env()
        assert run.params_for(10) is None
        assert run.params_for(99) is None


# ---------------------------------------------------------------------------
# End-to-end: param_selector enabled -> using-params in top strategy string
# ---------------------------------------------------------------------------

class FixedParamLinearRun(LinearStrategySearchRun):
    """Always steers PUCT toward tactic id 10 (the only solver tactic)."""

    def _priors_for(self, actions):
        return {a: (1.0 if a == 10 else 0.0) for a in actions}


def test_param_selector_produces_using_params_in_strategy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    benches = sorted((root / "data/smoke/benchmarks").rglob("*.smt2"))
    assert benches, "smoke benchmarks missing"
    bench_lst = [str(benches[0])]

    logic_config = {
        "solver_tactics": [10],
        "preprocess_tactics": [],
        "params": {10: {"random_seed": {"default": 0, "values": [0, 100, 200]}}},
    }
    mcts = MctsConfig(
        sim_num=5,
        timeout=1,
        c_uct=0.5,
        random_seed=0,
        param_selector=ParamSelectionConfig(enabled=True, c_ucb=DEFAULT_PARAM_C_UCB),
    )

    log_dir = tmp_path / "run"
    log_dir.mkdir()
    with patch("z3alpha.environment.SolverEvaluator") as mock_ev:
        mock_ev.return_value.evaluate.return_value = [(True, 0.01, "sat")]
        run = FixedParamLinearRun(
            mcts,
            bench_lst,
            "QF_NIA",
            "z3",
            "par10",
            log_dir,
            batch_size=1,
            logic_config=logic_config,
        )
        run.start()

    assert run.get_best_strat() is not None, "No strategy produced"
    # Each arm is tried at least once; non-default seeds (100, 200) must appear in the database
    strats_with_params = [s for s in run.res_database if ":random_seed" in s]
    assert strats_with_params, "Expected at least one strategy with non-default random_seed"
    for s in strats_with_params:
        m = re.search(r":random_seed\s+(\d+)", s)
        assert m is not None
        assert int(m.group(1)) in [100, 200], f"Unexpected seed value in: {s}"
