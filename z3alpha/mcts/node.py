"""Tree node for the MCTS engine."""

from __future__ import annotations

from typing import Any


class MCTSNode:
    """A node in the MCTS search tree.

    Holds the running visit count, value estimate, and child links keyed by
    action. Child histories are recorded for tracing and for AST-action
    lookup elsewhere.
    """

    def __init__(self, action_history: list[Any] | None = None) -> None:
        self.visit_count: int = 0
        self.action_history: list[Any] = (
            [] if action_history is None else list(action_history)
        )
        self.value_est: float = 0.0
        self.children: dict[Any, "MCTSNode"] = {}

    def __str__(self) -> str:
        return str(self.action_history)

    def is_expanded(self) -> bool:
        return bool(self.children)
