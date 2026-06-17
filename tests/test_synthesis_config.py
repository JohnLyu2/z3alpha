import pytest

from z3alpha.config import (
    DEFAULT_C_UCT,
    DEFAULT_IS_MEAN,
    DEFAULT_RANDOM_SEED,
    MctsConfig,
    parse_experiment_config,
    resolve_mcts_config,
)


def _minimal_experiment():
    return {
        "logic": "QF_NIA",
        "batch_size": 2,
        "train_dir": "data/smoke/benchmarks",
        "timeout": 1,
        "mcts_sims": 15,
        "branched_sims": 50,
        "ln_strat_num": 5,
    }


def test_parse_experiment_config_minimal():
    e = parse_experiment_config(_minimal_experiment())
    assert e.logic == "QF_NIA"
    assert e.mcts_sims == 15
    assert e.value_type == "par10"
    assert e.z3path is None


def test_parse_experiment_config_value_type_in_json():
    raw = _minimal_experiment()
    raw["value_type"] = "foo"
    e = parse_experiment_config(raw)
    assert e.value_type == "foo"


def test_parse_experiment_config_unknown_key_raises():
    raw = _minimal_experiment()
    raw["mcts_config"] = {"c_uct": 1.0}
    with pytest.raises(ValueError, match="Unknown key"):
        parse_experiment_config(raw)


def test_resolve_mcts_config_defaults():
    from z3alpha.mcts.param_selection import DEFAULT_PARAM_C_UCB, ParamSelectionConfig

    class A:
        c_uct = None
        random_seed = None

    experiment = parse_experiment_config(_minimal_experiment())
    cfg = resolve_mcts_config(A(), experiment)
    assert cfg == MctsConfig(
        sim_num=experiment.mcts_sims,
        timeout=experiment.timeout,
        c_uct=DEFAULT_C_UCT,
        random_seed=DEFAULT_RANDOM_SEED,
        is_mean=DEFAULT_IS_MEAN,
        llm_prior=None,
        param_selector=ParamSelectionConfig(enabled=True, c_ucb=DEFAULT_PARAM_C_UCB),
    )
    assert cfg.is_mean is False


def test_resolve_mcts_config_cli_overrides():
    class A:
        c_uct = 0.9
        random_seed = 7

    experiment = parse_experiment_config(_minimal_experiment())
    cfg = resolve_mcts_config(A(), experiment)
    assert cfg.c_uct == 0.9
    assert cfg.random_seed == 7
    assert cfg.sim_num == experiment.mcts_sims
    assert cfg.timeout == experiment.timeout
    assert cfg.is_mean is DEFAULT_IS_MEAN
