from z3alpha.ast_nodes import Root
from z3alpha.strat_tree_s1 import S1Strategy
from z3alpha.strat_tree_s2 import S2Strategy
from z3alpha.strat_tree_s2 import Action as S2Action

Action = int | S2Action


class StrategyAST:
    def __init__(
        self, stage, logic, timeout, s2config=None,
        logic_config=None,
    ):
        self.logic = logic
        self.timeout = timeout
        self.root = Root()
        if stage == 1:
            self.root.add_children(
                [S1Strategy(logic, logic_config)]
            )
        else:
            assert s2config
            s2dict = s2config["s2dict"]
            self.root.add_children([S2Strategy(timeout, s2dict, if_depth=0)])

    def __str__(self):
        return str(self.root)

    # Return the depth-first first nonterminal node in the tree.
    def find_fst_nonterm(self):
        nonterm_stack = [self.root]
        while nonterm_stack:
            node_to_search = nonterm_stack.pop()
            if not node_to_search.is_terminal():
                return node_to_search
            for child_node in reversed(node_to_search.children):
                nonterm_stack.append(child_node)
        return None

    def is_terminal(self):
        return not bool(self.find_fst_nonterm())

    def legal_actions(self, rollout: bool = False) -> list[Action]:
        if self.is_terminal():
            return []
        return self.find_fst_nonterm().legal_actions(rollout)

    def apply_rule(self, action: Action, params: dict | None) -> None:
        assert not self.is_terminal()
        node = self.find_fst_nonterm()
        node.apply_rule(action, params)

    def get_linear_strategies(self, probe_record):
        assert self.is_terminal()
        return self.root.get_ln_strats(self.timeout, probe_record)

    # Backward-compatible aliases.
    def findFstNonTerm(self):
        return self.find_fst_nonterm()

    def isTerminal(self):
        return self.is_terminal()

    def legalActions(self, rollout: bool = False) -> list[Action]:
        return self.legal_actions(rollout)

    def applyRule(self, action: Action, params: dict | None) -> None:
        return self.apply_rule(action, params)
