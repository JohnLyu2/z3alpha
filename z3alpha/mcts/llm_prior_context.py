"""Format same-run MCTS eval results for LLM prior prompts (V1 factual context)."""

from __future__ import annotations

from dataclasses import dataclass

from z3alpha.utils import par_n, solved_num

_CONTEXT_ROW_CAP = 20
_CONTEXT_TOP_WHEN_CAPPED = 15
_CONTEXT_BOTTOM_WHEN_CAPPED = 5

_LIGHT_NUDGE = (
    "Use these results when scoring: prefer extending the best-performing strategies; "
    "avoid next steps that continue patterns matching the worst outcomes."
)


@dataclass(frozen=True)
class RunContextVersion:
    num_strategies: int
    best_n_solved: int


def root_partial_strategy(logic: str) -> str:
    """Root partial string; must match ``LinearStrategy.__str__``."""
    return f"<LinearStrategy>({logic})"


@dataclass(frozen=True)
class StrategyContextRow:
    strategy: str
    n_solved: int
    par10_avg: float


def _benchmark_count(result_database: dict[str, list]) -> int:
    if not result_database:
        return 0
    return len(next(iter(result_database.values())))


def compute_run_context_version(result_database: dict[str, list]) -> RunContextVersion:
    if not result_database:
        return RunContextVersion(num_strategies=0, best_n_solved=0)
    best = max(solved_num(res) for res in result_database.values())
    return RunContextVersion(num_strategies=len(result_database), best_n_solved=best)


def build_strategy_context_rows(
    result_database: dict[str, list],
    timeout: float,
) -> list[StrategyContextRow]:
    rows: list[StrategyContextRow] = []
    for strategy, res in result_database.items():
        nb = len(res)
        if nb == 0:
            continue
        rows.append(
            StrategyContextRow(
                strategy=strategy,
                n_solved=solved_num(res),
                par10_avg=par_n(res, 10, timeout) / nb,
            )
        )
    rows.sort(key=lambda r: (-r.n_solved, r.par10_avg, r.strategy))
    return rows


def select_strategies_for_context(rows: list[StrategyContextRow]) -> list[StrategyContextRow]:
    """Return up to 20 rows: all if <=20, else top 15 + bottom 5 (deduped, order preserved)."""
    if len(rows) <= _CONTEXT_ROW_CAP:
        return list(rows)

    top = rows[:_CONTEXT_TOP_WHEN_CAPPED]
    bottom = rows[-_CONTEXT_BOTTOM_WHEN_CAPPED:]
    seen: set[str] = set()
    selected: list[StrategyContextRow] = []
    for row in top + bottom:
        if row.strategy in seen:
            continue
        seen.add(row.strategy)
        selected.append(row)
    return selected


def format_run_context(
    result_database: dict[str, list],
    timeout: float,
    sim_id: int,
) -> tuple[str, RunContextVersion]:
    version = compute_run_context_version(result_database)
    n_bench = _benchmark_count(result_database)

    if not result_database:
        body = "  (no strategies evaluated yet)"
        header = (
            f"Results from this run so far (factual; sim {sim_id}, "
            f"0 strategies evaluated, best 0/{n_bench}):"
        )
    else:
        rows = select_strategies_for_context(
            build_strategy_context_rows(result_database, timeout)
        )
        header = (
            f"Results from this run so far (factual; sim {sim_id}, "
            f"{version.num_strategies} strategies evaluated, "
            f"best {version.best_n_solved}/{n_bench}):"
        )
        lines = [
            f"  n_solved={row.n_solved}  par10_avg={row.par10_avg:.2f}  {row.strategy}"
            for row in rows
        ]
        body = "\n".join(lines)

    text = "\n".join([header, body, "", _LIGHT_NUDGE])
    return text, version
