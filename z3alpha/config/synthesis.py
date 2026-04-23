"""Typed experiment configuration, MCTS parameters, and optional synthesis run bundle."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any, Protocol, runtime_checkable

DEFAULT_C_UCT = 0.5
DEFAULT_C_UCB = 0.2
DEFAULT_RANDOM_SEED = 0


@dataclass(frozen=True)
class ExperimentConfig:
    """Fields allowed in the experiment JSON file (see :func:`parse_experiment_config`)."""

    logic: str
    batch_size: int
    train_dir: str
    timeout: int
    mcts_sims: int
    branched_sims: int
    ln_strat_num: int
    value_type: str = "par10"
    z3path: str | None = None
    parent_log_dir: str | None = None
    log_level: str | None = None
    logic_config_dir: str | None = None


@dataclass(frozen=True)
class MCTSParams:
    c_uct: float
    c_ucb: float
    random_seed: int


@dataclass(frozen=True)
class SynthesisRun:
    """One synthesis invocation: experiment file + resolved MCTS/seed (CLI over code defaults)."""

    experiment: ExperimentConfig
    m: MCTSParams


def _experiment_field_names() -> frozenset[str]:
    return frozenset(f.name for f in fields(ExperimentConfig))


def _required_experiment_keys() -> frozenset[str]:
    return frozenset(f.name for f in fields(ExperimentConfig) if f.default is MISSING)


def parse_experiment_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Build :class:`ExperimentConfig` from experiment JSON. Unknown keys raise ``ValueError``."""
    allowed = _experiment_field_names()
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"Unknown key(s) in experiment JSON: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )
    required = _required_experiment_keys() - set(raw)
    if required:
        raise ValueError(
            f"Missing required key(s) in experiment JSON: {sorted(required)}"
        )

    return ExperimentConfig(
        logic=raw["logic"],
        batch_size=raw["batch_size"],
        train_dir=raw["train_dir"],
        timeout=raw["timeout"],
        mcts_sims=raw["mcts_sims"],
        branched_sims=raw["branched_sims"],
        ln_strat_num=raw["ln_strat_num"],
        value_type=raw.get("value_type", "par10"),
        z3path=raw.get("z3path"),
        parent_log_dir=raw.get("parent_log_dir"),
        log_level=raw.get("log_level"),
        logic_config_dir=raw.get("logic_config_dir"),
    )


@runtime_checkable
class MctsCliArgs(Protocol):
    c_uct: float | None
    c_ucb: float | None
    random_seed: int | None


def resolve_mcts_params(args: MctsCliArgs) -> MCTSParams:
    """UCT/UCB/seed from module defaults, overridden by CLI flags when set."""
    c_uct = DEFAULT_C_UCT if args.c_uct is None else args.c_uct
    c_ucb = DEFAULT_C_UCB if args.c_ucb is None else args.c_ucb
    random_seed = DEFAULT_RANDOM_SEED if args.random_seed is None else args.random_seed
    return MCTSParams(c_uct=c_uct, c_ucb=c_ucb, random_seed=random_seed)
