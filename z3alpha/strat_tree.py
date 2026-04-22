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

    def current_decision_node(self):
        return self.find_fst_nonterm()

    def is_terminal(self):
        return not bool(self.current_decision_node())

    def legal_actions(self, rollout: bool = False) -> list[Action]:
        node = self.current_decision_node()
        if node is None:
            return []
        return node.legal_actions(rollout)

    def apply_rule(self, action: Action, params: dict | None) -> None:
        node = self.current_decision_node()
        assert node is not None
        node.apply_rule(action, params)

    def get_linear_strategies(self, probe_record):
        assert self.is_terminal()
        return self.root.get_ln_strats(self.timeout, probe_record)

