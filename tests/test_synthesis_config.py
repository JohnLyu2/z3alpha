import pytest

from z3alpha.synthesis_config import (
    DEFAULT_C_UCT,
    DEFAULT_C_UCB,
    DEFAULT_RANDOM_SEED,
    MCTSParams,
    parse_experiment_config,
    resolve_mcts_params,
)


def _minimal_experiment():
    return {
        "logic": "QF_NIA",
        "batch_size": 2,
        "train_dir": "data/sample/benchmarks",
        "timeout": 1,
        "mcts_sims": 15,
        "s2_sims": 50,
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


def test_resolve_mcts_params_defaults():
    class A:
        c_uct = None
        c_ucb = None
        random_seed = None

    m = resolve_mcts_params(A())
    assert m == MCTSParams(
        c_uct=DEFAULT_C_UCT,
        c_ucb=DEFAULT_C_UCB,
        random_seed=DEFAULT_RANDOM_SEED,
    )


def test_resolve_mcts_params_cli():
    class A:
        c_uct = 0.9
        c_ucb = None
        random_seed = 7

    m = resolve_mcts_params(A())
    assert m.c_uct == 0.9
    assert m.c_ucb == DEFAULT_C_UCB
    assert m.random_seed == 7
