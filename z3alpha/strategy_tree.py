from __future__ import annotations

from z3alpha.ast_nodes import ASTNode, Root, TacticNode
from z3alpha.tactics.catalog import PREPROCESS_CATALOG, SOLVER_CATALOG


class LinearStrategy(ASTNode):
    def __init__(self, logic, logic_config):
        super().__init__()
        self.logic = logic
        self.logic_config = logic_config

    def __str__(self):
        return f"<LinearStrategy>({self.logic})"

    def is_terminal(self):
        return False

    def legal_actions(self, rollout=False):
        return list(self.logic_config["solver_tactics"]) + list(
            self.logic_config["preprocess_tactics"]
        )

    def apply_rule(self, action, params):
        assert self.is_leaf()
        assert action in self.legal_actions()
        if action in SOLVER_CATALOG:
            selected = TacticNode(SOLVER_CATALOG[action], params, action)
            self.parent.replace_child(selected, self.pos)
            return
        if action in PREPROCESS_CATALOG:
            selected = TacticNode(PREPROCESS_CATALOG[action], params, action)
            self.parent.replace_child(selected, self.pos)
            selected.add_children([LinearStrategy(self.logic, self.logic_config)])
            return
        raise Exception("unexpected action")


class SearchTreeBase:
    """Shared DFS navigation; root has one child (the strategy AST under search)."""

    root: Root
    timeout: int
    logic: str

    def __str__(self):
        return str(self.root)

    def find_first_nonterminal(self):
        nonterm_stack = [self.root]
        while nonterm_stack:
            node_to_search = nonterm_stack.pop()
            if not node_to_search.is_terminal():
                return node_to_search
            for child_node in reversed(node_to_search.children):
                nonterm_stack.append(child_node)
        return None

    def current_decision_node(self):
        return self.find_first_nonterminal()

    def is_terminal(self):
        return not bool(self.current_decision_node())

    def legal_actions(self, rollout: bool = False) -> list:
        node = self.current_decision_node()
        if node is None:
            return []
        return node.legal_actions(rollout)

    def apply_rule(self, action, params: dict | None) -> None:
        node = self.current_decision_node()
        assert node is not None
        node.apply_rule(action, params)

    def get_linear_strategies(self, probe_record):
        assert self.is_terminal()
        return self.root.get_ln_strats(self.timeout, probe_record)


class LinearStrategyTree(SearchTreeBase):
    """MCTS search tree for linear (non-conditional) strategies only."""

    def __init__(self, logic, timeout, *, logic_config=None):
        self.logic = logic
        self.timeout = timeout
        self.root = Root()
        self.root.add_children([LinearStrategy(logic, logic_config)])
