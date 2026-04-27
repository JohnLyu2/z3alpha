"""Tests for LLM-derived PUCT priors and MCTS wiring (no live OpenAI calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from z3alpha.config import MctsConfig, parse_experiment_config, resolve_mcts_config
from z3alpha.mcts.linear import LinearStrategySearchRun
from z3alpha.mcts.llm_prior import LLMPriorConfig, LLMPriorScorer


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
    assert cfg.llm_prior.timeout_s == 42.0
    assert cfg.llm_prior.temperature == pytest.approx(0.1)


def test_llm_prior_scorer_normalize_and_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cache = tmp_path / "llm_priors.json"
    cfg = LLMPriorConfig(
        enabled=True,
        model="m",
        cache_path=cache,
    )
    scorer = LLMPriorScorer(cfg)
    calls = {"n": 0}

    def fake_chat(self, user_content: str) -> str | None:
        calls["n"] += 1
        return '{"10": 1, "11": 4}'

    monkeypatch.setattr(LLMPriorScorer, "_call_chat_completions", fake_chat)
    out1 = scorer.score("QF_NIA", "(then simplify)", ["10", "11"])
    assert out1 == {"10": pytest.approx(0.2), "11": pytest.approx(0.8)}
    assert calls["n"] == 1
    out2 = scorer.score("QF_NIA", "(then simplify)", ["10", "11"])
    assert out2 == out1
    assert calls["n"] == 1
    assert cache.exists()


def test_llm_prior_scorer_all_zero_uniform(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True, cache_path=tmp_path / "c.json")

    def fake_chat(self, user_content: str) -> str | None:
        return '{"a": 0, "b": 0}'

    monkeypatch.setattr(LLMPriorScorer, "_call_chat_completions", fake_chat)
    scorer = LLMPriorScorer(cfg)
    out = scorer.score("L", "x", ["a", "b"])
    assert out == {"a": 1.0, "b": 1.0}


def test_llm_prior_scorer_bad_json_uniform(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True, cache_path=tmp_path / "c.json")

    def fake_chat(self, user_content: str) -> str | None:
        return "not json"

    monkeypatch.setattr(LLMPriorScorer, "_call_chat_completions", fake_chat)
    scorer = LLMPriorScorer(cfg)
    out = scorer.score("L", "x", ["x", "y"])
    assert out == {"x": 1.0, "y": 1.0}


def test_llm_prior_scorer_disk_reload(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cache = tmp_path / "llm_priors.json"
    cfg = LLMPriorConfig(enabled=True, model="m1", cache_path=cache)
    calls = {"n": 0}

    def fake_chat(self, user_content: str) -> str | None:
        calls["n"] += 1
        return '{"p": 3, "q": 1}'

    monkeypatch.setattr(LLMPriorScorer, "_call_chat_completions", fake_chat)
    s1 = LLMPriorScorer(cfg)
    assert s1.score("L", "s", ["p", "q"]) == {
        "p": pytest.approx(0.75),
        "q": pytest.approx(0.25),
    }
    assert calls["n"] == 1

    def boom(self, user_content: str) -> str | None:
        raise AssertionError("no HTTP")

    monkeypatch.setattr(LLMPriorScorer, "_call_chat_completions", boom)
    s2 = LLMPriorScorer(cfg)
    assert s2.score("L", "s", ["p", "q"]) == {
        "p": pytest.approx(0.75),
        "q": pytest.approx(0.25),
    }


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
