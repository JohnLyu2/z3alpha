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
from z3alpha.mcts.llm_prior_context import RunContextVersion


def _install_mock_chat_client(monkeypatch, *, parsed: TacticPriorScores | None, error: Exception | None = None):
    """Mock OpenAI client with chat.completions.parse."""

    class _Message:
        def __init__(self):
            self.parsed = parsed
            self.content = (
                parsed.model_dump_json()
                if parsed is not None
                else None
            )
            self.refusal = None

    class _Choice:
        def __init__(self):
            self.message = _Message()
            self.finish_reason = "stop"

    class _Completion:
        def __init__(self):
            self.choices = [_Choice()]

    class _Completions:
        @staticmethod
        def parse(**kwargs):
            if error is not None:
                raise error
            return _Completion()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kwargs):
            self.chat = _Chat()

    monkeypatch.setattr("z3alpha.mcts.llm_prior.OpenAI", _Client)


def test_resolve_mcts_config_llm_prior_flags():
    class Args:
        c_uct = None
        random_seed = None
        llm_prior = True

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

    with patch.dict(
        "os.environ",
        {
            "Z3ALPHA_LLM_MODEL": "gpt-4o",
            "Z3ALPHA_LLM_BASE_URL": "https://example.invalid/v1",
            "Z3ALPHA_LLM_TIMEOUT": "42.0",
            "Z3ALPHA_LLM_TEMPERATURE": "0.1",
        },
        clear=False,
    ):
        cfg = resolve_mcts_config(Args(), experiment)
    assert cfg.llm_prior is not None
    assert cfg.llm_prior.enabled is True
    assert cfg.llm_prior.model == "gpt-4o"
    assert cfg.llm_prior.base_url == "https://example.invalid/v1"
    assert cfg.llm_prior.api_key_env == "OPENAI_API_KEY"
    assert cfg.llm_prior.llm_timeout == 42.0
    assert cfg.llm_prior.temperature == pytest.approx(0.1)
    assert cfg.llm_prior.prior_epsilon == pytest.approx(0.15)


def test_resolve_mcts_config_llm_prior_defaults_to_openrouter():
    class Args:
        c_uct = None
        random_seed = None
        llm_prior = True

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
    with patch.dict("os.environ", {}, clear=True):
        cfg = resolve_mcts_config(Args(), experiment)
    assert cfg.llm_prior is not None
    assert cfg.llm_prior.enabled is True
    assert cfg.llm_prior.model == "openrouter/free"
    assert cfg.llm_prior.base_url == "https://openrouter.ai/api/v1"
    assert cfg.llm_prior.api_key_env == "OPENROUTER_API_KEY"
    assert cfg.llm_prior.llm_timeout == 30.0
    assert cfg.llm_prior.temperature == pytest.approx(0.0)


def test_llm_prior_scorer_requires_api_key_when_enabled(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = LLMPriorConfig(enabled=True)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        LLMPriorScorer(cfg)


def test_llm_prior_scorer_uses_openrouter_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = LLMPriorConfig(enabled=True, base_url="https://openrouter.ai/api/v1")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        LLMPriorScorer(cfg)


def test_llm_prior_scorer_thresholded_softmax_and_cache(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(
        enabled=True, model="m", softmax_temperature=2.0, prior_epsilon=0.0
    )
    scorer = LLMPriorScorer(cfg)
    calls = {"n": 0}

    def fake_parse(self, user_content: str, sim_id=None):
        calls["n"] += 1
        return TacticPriorScores(
            scores=[
                TacticScoreItem(tactic_name="smt", value=2),
                TacticScoreItem(tactic_name="qfnra-nlsat", value=8),
            ]
        )

    monkeypatch.setattr(LLMPriorScorer, "_chat_completions_parse_scores", fake_parse)
    out1 = scorer.score("QF_NIA", "(then simplify)", ["smt", "qfnra-nlsat"], sim_id=0)
    assert scorer.last_prior_source == "api_call"
    assert out1 == {
        "smt": pytest.approx(0.04742587317756679),
        "qfnra-nlsat": pytest.approx(0.9525741268224334),
    }
    assert calls["n"] == 1
    out2 = scorer.score("QF_NIA", "(then simplify)", ["smt", "qfnra-nlsat"], sim_id=1)
    assert scorer.last_prior_source == "cache_hit"
    assert out2 == out1
    assert calls["n"] == 1


def test_llm_prior_scorer_epsilon_mix(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True, prior_epsilon=0.15)
    scorer = LLMPriorScorer(cfg)

    def fake_parse(self, user_content: str, sim_id=None):
        return TacticPriorScores(
            scores=[
                TacticScoreItem(tactic_name="winner", value=8),
                TacticScoreItem(tactic_name="mid", value=5),
                TacticScoreItem(tactic_name="rejected", value=1),
            ]
        )

    monkeypatch.setattr(LLMPriorScorer, "_chat_completions_parse_scores", fake_parse)
    out = scorer.score(
        "QF_NIA", "(then simplify)", ["winner", "mid", "rejected"], sim_id=0
    )
    assert out["rejected"] == pytest.approx(0.15 / 3)
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["winner"] > out["rejected"]
    assert scorer.last_mapping_mode == "thresholded_softmax_eps0.15"


def test_llm_prior_scorer_counts_api_calls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)
    scorer = LLMPriorScorer(cfg)

    parsed = TacticPriorScores(
        scores=[
            TacticScoreItem(tactic_name="a", value=8),
            TacticScoreItem(tactic_name="b", value=2),
        ]
    )
    _install_mock_chat_client(monkeypatch, parsed=parsed)
    scorer.score("L", "partial-a", ["a", "b"])
    assert scorer.api_call_count == 1
    scorer.score("L", "partial-a", ["a", "b"])
    assert scorer.api_call_count == 1
    scorer.score("L", "partial-b", ["a", "b"])
    assert scorer.api_call_count == 2


def test_llm_prior_scorer_counts_failed_api_calls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)
    scorer = LLMPriorScorer(cfg)

    _install_mock_chat_client(
        monkeypatch, parsed=None, error=RuntimeError("401 Unauthorized")
    )
    out = scorer.score("L", "partial-a", ["a", "b"])
    assert out == {"a": 1.0, "b": 1.0}
    assert scorer.api_call_count == 1


def test_llm_prior_scorer_all_zero_uniform(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)

    def fake_parse(self, user_content: str, sim_id=None):
        return TacticPriorScores(
            scores=[
                TacticScoreItem(tactic_name="a", value=0),
                TacticScoreItem(tactic_name="b", value=0),
            ]
        )

    monkeypatch.setattr(LLMPriorScorer, "_chat_completions_parse_scores", fake_parse)
    scorer = LLMPriorScorer(cfg)
    out = scorer.score("L", "x", ["a", "b"])
    assert out == {"a": 1.0, "b": 1.0}


def test_llm_prior_scorer_uncertain_uniform(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)

    def fake_parse(self, user_content: str, sim_id=None):
        return TacticPriorScores(
            scores=[
                TacticScoreItem(tactic_name="a", value=4),
                TacticScoreItem(tactic_name="b", value=5),
            ]
        )

    monkeypatch.setattr(LLMPriorScorer, "_chat_completions_parse_scores", fake_parse)
    scorer = LLMPriorScorer(cfg)
    out = scorer.score("L", "x", ["a", "b"])
    assert out == {"a": 1.0, "b": 1.0}


def test_llm_prior_scorer_bad_json_uniform(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)

    def fake_parse(self, user_content: str, sim_id=None) -> None:
        return None

    monkeypatch.setattr(LLMPriorScorer, "_chat_completions_parse_scores", fake_parse)
    scorer = LLMPriorScorer(cfg)
    out = scorer.score("L", "x", ["x", "y"])
    assert out == {"x": 1.0, "y": 1.0}


def test_llm_prior_scorer_empty_scores_uniform(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)
    _install_mock_chat_client(monkeypatch, parsed=TacticPriorScores(scores=[]))
    scorer = LLMPriorScorer(cfg)
    out = scorer.score("L", "x", ["a", "b"])
    assert out == {"a": 1.0, "b": 1.0}
    assert not any(key[0] == "x" for key in scorer._memory)


def test_llm_prior_scorer_rejects_none_choices(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)
    scorer = LLMPriorScorer(cfg)

    class _Completion:
        choices = None

    class _Completions:
        @staticmethod
        def parse(**kwargs):
            return _Completion()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kwargs):
            self.chat = _Chat()

    monkeypatch.setattr("z3alpha.mcts.llm_prior.OpenAI", _Client)
    out = scorer.score("L", "partial-none-choices", ["a", "b"], sim_id=3)
    assert out == {"a": 1.0, "b": 1.0}
    assert scorer.api_call_count == 1
    assert scorer.last_prior_source == "api_call_failed"


def test_llm_prior_scorer_api_fallback_not_cached(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)
    _install_mock_chat_client(monkeypatch, parsed=None)
    scorer = LLMPriorScorer(cfg)
    scorer.score("L", "partial-z", ["a", "b"])
    scorer.score("L", "partial-z", ["a", "b"])
    assert scorer.api_call_count == 2


def test_versioned_cache_hits_same_version(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)
    scorer = LLMPriorScorer(cfg)
    calls = {"n": 0}

    def fake_parse(self, user_content: str, sim_id=None):
        calls["n"] += 1
        return TacticPriorScores(
            scores=[
                TacticScoreItem(tactic_name="a", value=8),
                TacticScoreItem(tactic_name="b", value=2),
            ]
        )

    monkeypatch.setattr(LLMPriorScorer, "_chat_completions_parse_scores", fake_parse)
    version = RunContextVersion(num_strategies=3, best_n_solved=120)
    ctx = "Results from this run so far (factual; sim 5, 3 strategies evaluated, best 120/150):"
    scorer.score(
        "QF_NIA",
        "partial",
        ["a", "b"],
        sim_id=5,
        run_context=ctx,
        run_context_version=version,
    )
    scorer.score(
        "QF_NIA",
        "partial",
        ["a", "b"],
        sim_id=6,
        run_context=ctx,
        run_context_version=version,
    )
    assert calls["n"] == 1
    assert scorer.last_prior_source == "cache_hit"


def test_versioned_cache_recalls_on_new_best(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)
    scorer = LLMPriorScorer(cfg)
    calls = {"n": 0}

    def fake_parse(self, user_content: str, sim_id=None):
        calls["n"] += 1
        return TacticPriorScores(
            scores=[
                TacticScoreItem(tactic_name="a", value=8),
                TacticScoreItem(tactic_name="b", value=2),
            ]
        )

    monkeypatch.setattr(LLMPriorScorer, "_chat_completions_parse_scores", fake_parse)
    scorer.score(
        "QF_NIA",
        "partial",
        ["a", "b"],
        run_context="ctx-a",
        run_context_version=RunContextVersion(2, 120),
    )
    scorer.score(
        "QF_NIA",
        "partial",
        ["a", "b"],
        run_context="ctx-b",
        run_context_version=RunContextVersion(3, 140),
    )
    assert calls["n"] == 2


def test_user_prompt_includes_run_context(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = LLMPriorConfig(enabled=True)
    scorer = LLMPriorScorer(cfg)
    captured: list[str] = []

    def fake_parse(self, user_content: str, sim_id=None):
        captured.append(user_content)
        return TacticPriorScores(
            scores=[
                TacticScoreItem(tactic_name="smt", value=5),
                TacticScoreItem(tactic_name="qfnia", value=5),
            ]
        )

    monkeypatch.setattr(LLMPriorScorer, "_chat_completions_parse_scores", fake_parse)
    run_context = (
        "Results from this run so far (factual; sim 2, 1 strategies evaluated, best 1/1):\n"
        "  n_solved=1  par10_avg=1.00  smt\n"
        "\n"
        "Use these results when scoring: prefer extending the best-performing strategies; "
        "avoid next steps that continue patterns matching the worst outcomes."
    )
    scorer.score(
        "QF_NIA",
        "<LinearStrategy>(QF_NIA)",
        ["smt", "qfnia"],
        run_context=run_context,
        run_context_version=RunContextVersion(1, 1),
    )
    assert len(captured) == 1
    prompt = captured[0]
    assert "Results from this run so far (factual" in prompt
    assert "best-performing strategies" in prompt
    assert "<LinearStrategy>(QF_NIA)" in prompt
    assert '"smt"' in prompt


def test_llm_prior_scorer_writes_qa_log(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    qa_log = tmp_path / "llm_prior_qa.log"
    cfg = LLMPriorConfig(
        enabled=True, qa_log_path=str(qa_log), prior_epsilon=0.0
    )

    parsed = TacticPriorScores(
        scores=[
            TacticScoreItem(tactic_name="smt", value=2),
            TacticScoreItem(tactic_name="qfnra-nlsat", value=8),
        ]
    )

    _install_mock_chat_client(monkeypatch, parsed=parsed)
    scorer = LLMPriorScorer(cfg)
    out = scorer.score("QF_NIA", "(then simplify)", ["smt", "qfnra-nlsat"], sim_id=7)
    assert out == {
        "smt": pytest.approx(0.04742587317756679),
        "qfnra-nlsat": pytest.approx(0.9525741268224334),
    }
    assert scorer.api_call_count == 1
    txt = qa_log.read_text(encoding="utf-8")
    assert "kind: llm_prior_qa" in txt
    assert "kind: llm_prior_scores" in txt
    assert "sim_id: 7" in txt
    assert "mapping_mode: thresholded_softmax" in txt
    assert "ranked_priors" in txt


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
