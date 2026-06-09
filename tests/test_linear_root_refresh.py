"""Tests for root child prior refresh when run context version changes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import z3alpha.config  # noqa: F401

from z3alpha.config import MctsConfig
from z3alpha.mcts.linear import LinearStrategySearchRun
from z3alpha.mcts.llm_prior import LLMPriorConfig, LLMPriorScorer
from z3alpha.mcts.llm_prior_context import RunContextVersion, root_partial_strategy
from z3alpha.mcts.node import MCTSNode


def _res(solved: bool) -> list[tuple]:
    return [(solved, 0.1 if solved else 10.0, "sat" if solved else "timeout")]


def _make_run(tmp_path: Path, scorer: LLMPriorScorer) -> LinearStrategySearchRun:
    logic_config = {"solver_tactics": [10, 11], "preprocess_tactics": [32]}
    mcts = MctsConfig(
        sim_num=1,
        timeout=10,
        c_uct=0.5,
        random_seed=0,
        llm_prior=LLMPriorConfig(enabled=True),
    )
    run = LinearStrategySearchRun(
        mcts,
        ["bench.smt2"],
        "QF_NIA",
        "z3",
        "par10",
        tmp_path / "run",
        batch_size=1,
        logic_config=logic_config,
    )
    run._scorer = scorer
    return run


def test_root_partial_strategy_matches_linear_strategy_str() -> None:
    assert root_partial_strategy("QF_NIA") == "<LinearStrategy>(QF_NIA)"


def test_root_refresh_updates_child_priors(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scorer = LLMPriorScorer(LLMPriorConfig(enabled=True))
    calls: list[RunContextVersion] = []

    def fake_score(
        self,
        logic,
        partial_strategy,
        candidate_actions,
        sim_id=None,
        *,
        run_context=None,
        run_context_version=None,
    ):
        calls.append(run_context_version)
        self.last_prior_source = "api_call"
        return {name: 0.75 if name == "smt" else 0.25 for name in candidate_actions}

    monkeypatch.setattr(LLMPriorScorer, "score", fake_score)
    run = _make_run(tmp_path, scorer)
    run.root.children[10] = MCTSNode([10])
    run.root.children[10].prior = 0.1
    run.root.children[11] = MCTSNode([11])
    run.root.children[11].prior = 0.9
    run.num_sim = 1

    run.res_database = {"s1": _res(True)}
    run._maybe_refresh_root_priors()
    assert run.root.children[10].prior == pytest.approx(0.75)
    assert run.root.children[11].prior == pytest.approx(0.25)
    assert calls == [RunContextVersion(1, 1)]

    run.res_database = {"s1": _res(True), "s2": _res(False)}
    run._maybe_refresh_root_priors()
    assert calls == [
        RunContextVersion(1, 1),
        RunContextVersion(2, 1),
    ]


def test_root_refresh_skips_same_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scorer = LLMPriorScorer(LLMPriorConfig(enabled=True))
    call_count = {"n": 0}

    def fake_score(self, *args, **kwargs):
        call_count["n"] += 1
        self.last_prior_source = "api_call"
        names = kwargs.get("candidate_actions") or args[2]
        return {name: 0.5 for name in names}

    monkeypatch.setattr(LLMPriorScorer, "score", fake_score)
    run = _make_run(tmp_path, scorer)
    run.root.children[10] = MCTSNode([10])
    run.num_sim = 1
    run.res_database = {"s1": _res(True)}
    run._maybe_refresh_root_priors()
    run._maybe_refresh_root_priors()
    assert call_count["n"] == 1


def test_root_refresh_skips_unexpanded_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scorer = LLMPriorScorer(LLMPriorConfig(enabled=True))
    call_count = {"n": 0}

    def fake_score(self, *args, **kwargs):
        call_count["n"] += 1
        return {}

    monkeypatch.setattr(LLMPriorScorer, "score", fake_score)
    run = _make_run(tmp_path, scorer)
    run.res_database = {"s1": _res(True)}
    run._maybe_refresh_root_priors()
    assert call_count["n"] == 0


def test_root_refresh_before_select(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    scorer = LLMPriorScorer(LLMPriorConfig(enabled=True))
    order: list[str] = []

    def fake_refresh(self) -> None:
        order.append("refresh")

    def fake_select(self):
        order.append("select")
        return self.root, [self.root]

    monkeypatch.setattr(LinearStrategySearchRun, "_maybe_refresh_root_priors", fake_refresh)
    monkeypatch.setattr(LinearStrategySearchRun, "_select", fake_select)

    root = Path(__file__).resolve().parents[1]
    benches = sorted((root / "data/smoke/benchmarks").rglob("*.smt2"))
    assert benches, "sample benchmarks missing"

    logic_config = {"solver_tactics": [10, 11], "preprocess_tactics": []}
    mcts = MctsConfig(
        sim_num=1,
        timeout=1,
        c_uct=0.5,
        random_seed=0,
        llm_prior=LLMPriorConfig(enabled=True),
    )
    log_folder = tmp_path / "run"
    log_folder.mkdir()

    with patch("z3alpha.environment.SolverEvaluator") as mock_ev:
        mock_ev.return_value.evaluate.return_value = [(True, 0.01, "sat")]
        run = LinearStrategySearchRun(
            mcts,
            [str(benches[0])],
            "QF_NIA",
            "z3",
            "par10",
            log_folder,
            batch_size=1,
            logic_config=logic_config,
        )
        run._scorer = scorer
        run.num_sim = 1
        run.env = run._create_env()
        run._one_simulation()

    assert order == ["refresh", "select"]
