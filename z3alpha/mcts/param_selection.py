"""Per-tactic parameter selection via multi-armed bandits (UCB1).

Provides a pluggable ``ParamSelector`` Protocol so the MAB implementation can
later be swapped for an LLM-guided selector without restructuring callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

DEFAULT_PARAM_C_UCB = 0.2


@dataclass(frozen=True)
class ParamSelectionConfig:
    enabled: bool = True
    c_ucb: float = DEFAULT_PARAM_C_UCB


@runtime_checkable
class ParamSelector(Protocol):
    """Interface for tactic-parameter selectors."""

    def select(
        self,
        logic: str,
        partial_strategy: str,
        tactic_name: str,
        param_grid: dict[str, dict[str, Any]],
        sim_id: int | None = None,
    ) -> dict[str, Any]:
        """Return {param_name: chosen_value} for one tactic application.

        ``partial_strategy`` is the rendered strategy string *before* this
        tactic is applied (same convention as ``LLMPriorScorer.score``).
        """
        ...

    def backup_episode(self, reward: float) -> None:
        """Update arm statistics for all selections made in the current simulation."""
        ...


class MabParamSelector:
    """UCB1 multi-armed bandit for tactic parameter selection.

    One instance lives for the full ``sim_num`` of a run, accumulating
    statistics across all simulations. Arm state is keyed by
    ``(partial_strategy, tactic_name)`` so identical partial strategies at
    different tree positions share statistics. ``_pending`` accumulates
    selections from both the tree-traversal and rollout phases within one
    simulation before a single ``backup_episode`` call clears it.
    """

    def __init__(self, c_ucb: float, is_mean: bool) -> None:
        self.c_ucb = c_ucb
        self.is_mean = is_mean
        # key -> total selections (used as N in UCB1 ln(N)/n)
        self._visits: dict[tuple[str, str], int] = {}
        # key -> {param_name -> {value -> [n, q]}}
        self._mabs: dict[tuple[str, str], dict[str, dict[Any, list[int | float]]]] = {}
        # selections not yet backed up (cleared by backup_episode)
        self._pending: list[tuple[tuple[str, str], dict[str, Any]]] = []

    def select(
        self,
        logic: str,
        partial_strategy: str,
        tactic_name: str,
        param_grid: dict[str, dict[str, Any]],
        sim_id: int | None = None,
    ) -> dict[str, Any]:
        key: tuple[str, str] = (partial_strategy, tactic_name)
        mabs = self._mabs.setdefault(key, {})
        chosen: dict[str, Any] = {}
        for pname, spec in param_grid.items():
            arms = mabs.setdefault(pname, {v: [0, 0.0] for v in spec["values"]})
            # UCB1: N = total arm pulls for this param's bandit (standard convention).
            # Unvisited arms (n=0) get +inf so all arms are tried before exploitation
            # begins -- this is the standard UCB1 guarantee.
            total_n = sum(n for n, _ in arms.values())
            best_val, best_score = None, float("-inf")
            for val, (n, q) in arms.items():
                if n == 0:
                    score = float("inf")
                else:
                    score = q + self.c_ucb * math.sqrt(math.log(total_n) / n)
                if score > best_score:
                    best_score, best_val = score, val
            chosen[pname] = best_val
        self._pending.append((key, chosen))
        return chosen

    def backup_episode(self, reward: float) -> None:
        for key, chosen in self._pending:
            self._visits[key] = self._visits.get(key, 0) + 1
            mabs = self._mabs[key]
            for pname, val in chosen.items():
                n, q = mabs[pname][val]
                if n == 0:
                    q = reward
                elif self.is_mean:
                    q = (q * n + reward) / (n + 1)
                else:
                    q = max(q, reward)
                mabs[pname][val] = [n + 1, q]
        self._pending.clear()
