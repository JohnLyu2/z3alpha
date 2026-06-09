"""Experiment JSON (``ExperimentConfig``), MCTS/seed defaults, CLI overrides."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
import os
from typing import Any, Protocol, runtime_checkable

from z3alpha.mcts.llm_prior import LLMPriorConfig
from z3alpha.mcts.run import DEFAULT_IS_MEAN, MctsConfig

# CLI / code defaults (experiment JSON does not set these; use --c-uct, --random-seed)
DEFAULT_C_UCT = 0.5
DEFAULT_RANDOM_SEED = 0
DEFAULT_LLM_MODEL = "openrouter/free"
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_TIMEOUT = 30.0
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_PRIOR_EPSILON = 0.15


@dataclass(frozen=True)
class ExperimentConfig:
    """One experiment file; see :func:`parse_experiment_config` for the JSON schema."""

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
class SynthesisRun:
    experiment: ExperimentConfig
    mcts: MctsConfig


def _allowed_keys() -> frozenset[str]:
    return frozenset(f.name for f in fields(ExperimentConfig))


def _required_keys() -> frozenset[str]:
    return frozenset(f.name for f in fields(ExperimentConfig) if f.default is MISSING)


def parse_experiment_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Reject unknown keys; require every field with no default."""
    allowed = _allowed_keys()
    if extra := (set(raw) - allowed):
        raise ValueError(
            f"Unknown key(s) in experiment JSON: {sorted(extra)}. Allowed: {sorted(allowed)}"
        )
    if missing := (_required_keys() - set(raw)):
        raise ValueError(
            f"Missing required key(s) in experiment JSON: {sorted(missing)}"
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
    random_seed: int | None


def resolve_mcts_config(args: MctsCliArgs, experiment: ExperimentConfig) -> MctsConfig:
    """Build the linear-stage :class:`MctsConfig` from CLI overrides + experiment.

    Branched runs derive their own config by copying this one with a different
    ``sim_num`` (see :func:`z3alpha.stage2.search_runtime.run_branched_synthesis`).
    """
    llm_prior: LLMPriorConfig | None = None
    if getattr(args, "llm_prior", False):
        base_url = os.environ.get("Z3ALPHA_LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
        api_key_env = (
            "OPENROUTER_API_KEY"
            if "openrouter.ai" in base_url.lower()
            else "OPENAI_API_KEY"
        )
        llm_prior = LLMPriorConfig(
            enabled=True,
            model=os.environ.get("Z3ALPHA_LLM_MODEL", DEFAULT_LLM_MODEL),
            base_url=base_url,
            api_key_env=api_key_env,
            llm_timeout=float(os.environ.get("Z3ALPHA_LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT)),
            temperature=float(
                os.environ.get("Z3ALPHA_LLM_TEMPERATURE", DEFAULT_LLM_TEMPERATURE)
            ),
            prior_epsilon=float(
                os.environ.get(
                    "Z3ALPHA_LLM_PRIOR_EPSILON", DEFAULT_LLM_PRIOR_EPSILON
                )
            ),
        )
    return MctsConfig(
        sim_num=experiment.mcts_sims,
        timeout=experiment.timeout,
        c_uct=DEFAULT_C_UCT if args.c_uct is None else args.c_uct,
        random_seed=DEFAULT_RANDOM_SEED if args.random_seed is None else args.random_seed,
        is_mean=DEFAULT_IS_MEAN,
        llm_prior=llm_prior,
    )
