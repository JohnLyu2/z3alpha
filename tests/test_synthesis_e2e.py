"""End-to-end tests for the z3alpha synthesis workflow.

Calls synthesize_linear_strategies / ml_synthesize / branched_synthesize with a
minimal config and verifies output files are produced and the resulting strategy
solves benchmarks.  All tests require a real z3 binary.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from z3alpha.config import SynthesisRun, parse_experiment_config, resolve_mcts_config
from z3alpha.config.env import EnvConfig
from z3alpha.evaluator import SolverEvaluator
from z3alpha.synthesize import branched_synthesize, ml_synthesize, synthesize_linear_strategies

_Z3_PATH = shutil.which("z3")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_BENCH = str(_REPO_ROOT / "data" / "smoke" / "benchmarks" / "QF_NIA")

pytestmark = pytest.mark.skipif(_Z3_PATH is None, reason="z3 binary not on PATH")


def _make_run(mcts_sims: int = 3, branched_sims: int = 3, max_ln_strategies: int = 1) -> SynthesisRun:
    experiment = parse_experiment_config({
        "logic": "QF_NIA",
        "train_dir": _SMOKE_BENCH,
        "timeout": 2,
        "mcts_sims": mcts_sims,
        "branched_sims": branched_sims,
        "max_ln_strategies": max_ln_strategies,
    })

    class _Args:
        c_uct = None
        random_seed = 0

    return SynthesisRun(experiment=experiment, mcts=resolve_mcts_config(_Args(), experiment))


def _env() -> EnvConfig:
    return EnvConfig(workers=2, z3_path=_Z3_PATH)


# ---------------------------------------------------------------------------
# Stage-1: synthesize_linear_strategies
# ---------------------------------------------------------------------------

def test_linear_synthesis_output_files(tmp_path):
    bench_lst, shortlist = synthesize_linear_strategies(_make_run(), tmp_path, env=_env())

    assert bench_lst, "benchmark list is empty"
    assert shortlist, "shortlist is empty — no strategy solved any benchmark"

    assert (tmp_path / "linear_strategy_mcts.log").exists()

    summary = tmp_path / "linear_strategy_summary.csv"
    assert summary.exists()
    rows = list(csv.DictReader(summary.read_text().splitlines()))
    assert rows, "summary CSV has no data rows"
    assert {"id", "strategy", "n_solved", "par2_avg", "par10_avg"} == set(rows[0].keys())

    per_bench = tmp_path / "linear_strategy_per_benchmark.csv"
    assert per_bench.exists()
    assert list(csv.DictReader(per_bench.read_text().splitlines())), "per-benchmark CSV has no data rows"

    selected = tmp_path / "linear_selected_strategies.csv"
    assert selected.exists()
    strats = [r["strat"] for r in csv.DictReader(selected.read_text().splitlines())]
    assert strats, "no strategies written to linear_selected_strategies.csv"


def test_linear_synthesis_shortlist_matches_selected_file(tmp_path):
    _, shortlist = synthesize_linear_strategies(_make_run(max_ln_strategies=2), tmp_path, env=_env())

    strats_from_file = [
        r["strat"]
        for r in csv.DictReader((tmp_path / "linear_selected_strategies.csv").read_text().splitlines())
    ]
    assert [s for s, _ in shortlist] == strats_from_file


def test_linear_synthesis_shortlist_results_match_bench_count(tmp_path):
    bench_lst, shortlist = synthesize_linear_strategies(_make_run(), tmp_path, env=_env())

    for strat, results in shortlist:
        assert len(results) == len(bench_lst), (
            f"result count {len(results)} != bench count {len(bench_lst)} for {strat!r}"
        )


# ---------------------------------------------------------------------------
# ml_synthesize: stage 1 + PWC selector
# ---------------------------------------------------------------------------

def test_ml_synthesize_saves_selector(tmp_path):
    ml_synthesize(_make_run(max_ln_strategies=2), tmp_path, env=_env())
    assert (tmp_path / "selector.pkl").exists()


# ---------------------------------------------------------------------------
# branched_synthesize: stage 1 + stage-2 MCTS
# ---------------------------------------------------------------------------

def test_branched_synthesize_produces_strategy_file(tmp_path):
    branched_synthesize(_make_run(branched_sims=5), tmp_path, env=_env())
    strat_file = tmp_path / "synthesized_strategy.txt"
    assert strat_file.exists()
    assert strat_file.read_text().strip(), "synthesized_strategy.txt is empty"


# ---------------------------------------------------------------------------
# Synthesized strategy actually solves benchmarks
# ---------------------------------------------------------------------------

def test_best_linear_strategy_solves_benchmark(tmp_path):
    """The best stage-1 strategy must solve at least one smoke benchmark."""
    bench_lst, shortlist = synthesize_linear_strategies(_make_run(mcts_sims=5), tmp_path, env=_env())

    assert shortlist, "shortlist is empty"
    best_strat = shortlist[0][0]
    evaluator = SolverEvaluator(_Z3_PATH, bench_lst, timeout=2, batch_size=2)
    results = evaluator.evaluate(best_strat)

    assert any(r[0] for r in results), f"Strategy {best_strat!r} solved 0/{len(bench_lst)} benchmarks"


def test_branched_strategy_solves_benchmark(tmp_path):
    """The branched-synthesis output strategy must solve at least one smoke benchmark."""
    run = _make_run(mcts_sims=5, branched_sims=5)
    bench_lst, _ = synthesize_linear_strategies(run, tmp_path, env=_env())
    branched_synthesize(run, tmp_path, env=_env())

    strat = (tmp_path / "synthesized_strategy.txt").read_text().strip()
    evaluator = SolverEvaluator(_Z3_PATH, bench_lst, timeout=2, batch_size=2)
    results = evaluator.evaluate(strat)

    assert any(r[0] for r in results), f"Strategy {strat!r} solved 0/{len(bench_lst)} benchmarks"
