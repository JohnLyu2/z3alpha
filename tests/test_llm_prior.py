"""Tests for LLM-derived PUCT priors and MCTS wiring (no live OpenAI calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from z3alpha.config import MctsConfig, parse_experiment_config, resolve_mcts_config
from z3alpha.mcts.linear import LinearStrategySearchRun
from z3alpha.mcts.llm_prior import (
    LLMPriorConfig,
    LLMPriorScorer,
    TacticPriorScores,
    TacticScoreItem,
)


def test_resolve_mcts_config_llm_prior_flags():
    class Args:
        c_uct = None
        random_seed = None
        llm_prior = True
        llm_model = "gpt-4o"
        llm_base_url = "https://example.invalid/v1"
        llm_timeout = 42.0
        llm_temperature = 0.1

    experiment = parse_experiment_config(
        {
            "logic": "QF_NIA",
            "batch_size": 1,
            "train_dir": "data/smoke/benchmarks",
            "timeout": 1,
            "mcts_sims": 3,
            "branched_sims": 1,
            "ln_strat_num": 1,
        }
    )
    cfg = resolve_mcts_config(Args(), experiment)
    assert cfg.llm_prior is not None
    assert cfg.llm_prior.enabled is True
    assert cfg.llm_prior.model == "gpt-4o"
    assert cfg.llm_prior.base_url == "https://example.invalid/v1"
    assert cfg.llm_prior.llm_timeout == 42.0
    assert cfg.llm_prior.temperature == pytest.approx(0.1)


def test_llm_prior_scorer_requires_api_key_when_enabled(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = LLMPriorConfig(enabled=True)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        LLMPriorScorer(cfg)


def test_llm_prior_scorer_normalize_and_cache(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True, model="m")
    scorer = LLMPriorScorer(cfg)
    calls = {"n": 0}

    def fake_parse(self, user_content: str):
        calls["n"] += 1
        return TacticPriorScores(
            scores=[
                TacticScoreItem(tactic_name="smt", value=1),
                TacticScoreItem(tactic_name="qfnra-nlsat", value=4),
            ]
        )

    monkeypatch.setattr(LLMPriorScorer, "_responses_parse_scores", fake_parse)
    out1 = scorer.score("QF_NIA", "(then simplify)", ["smt", "qfnra-nlsat"])
    assert out1 == {"smt": pytest.approx(0.2), "qfnra-nlsat": pytest.approx(0.8)}
    assert calls["n"] == 1
    out2 = scorer.score("QF_NIA", "(then simplify)", ["smt", "qfnra-nlsat"])
    assert out2 == out1
    assert calls["n"] == 1


def test_llm_prior_scorer_all_zero_uniform(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)

    def fake_parse(self, user_content: str):
        return TacticPriorScores(
            scores=[
                TacticScoreItem(tactic_name="a", value=0),
                TacticScoreItem(tactic_name="b", value=0),
            ]
        )

    monkeypatch.setattr(LLMPriorScorer, "_responses_parse_scores", fake_parse)
    scorer = LLMPriorScorer(cfg)
    out = scorer.score("L", "x", ["a", "b"])
    assert out == {"a": 1.0, "b": 1.0}


def test_llm_prior_scorer_bad_json_uniform(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)

    def fake_parse(self, user_content: str) -> None:
        return None

    monkeypatch.setattr(LLMPriorScorer, "_responses_parse_scores", fake_parse)
    scorer = LLMPriorScorer(cfg)
    out = scorer.score("L", "x", ["x", "y"])
    assert out == {"x": 1.0, "y": 1.0}


class FixedPriorLinearRun(LinearStrategySearchRun):
    """Deterministic priors for two tactic ids (solver ids 10 and 11)."""

    def _priors_for(self, actions):
        if set(actions) == {10, 11}:
            return {10: 0.75, 11: 0.25}
        return {a: 1.0 for a in actions}


def test_expand_sets_child_priors_e2e(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    benches = sorted((root / "data/smoke/benchmarks").rglob("*.smt2"))
    assert benches, "sample benchmarks missing"
    bench_lst = [str(benches[0])]
    log_folder = tmp_path / "run"
    log_folder.mkdir()

    logic_config = {"solver_tactics": [10, 11], "preprocess_tactics": []}
    mcts = MctsConfig(
        sim_num=1,
        timeout=1,
        c_uct=0.5,
        random_seed=0,
        llm_prior=None,
    )

    with patch("z3alpha.environment.SolverEvaluator") as mock_ev:
        mock_ev.return_value.evaluate.return_value = [(True, 0.01, "sat")]
        run = FixedPriorLinearRun(
            mcts,
            bench_lst,
            "QF_NIA",
            "z3",
            "par10",
            log_folder,
            batch_size=1,
            logic_config=logic_config,
        )
        run.start()

    assert 10 in run.root.children
    assert 11 in run.root.children
    assert run.root.children[10].prior == pytest.approx(0.75)
    assert run.root.children[11].prior == pytest.approx(0.25)
