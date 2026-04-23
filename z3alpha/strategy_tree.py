from z3alpha.ast_nodes import ASTNode, Root, TacticNode
from z3alpha.stage2.strategy_tree import Action as Stage2Action
from z3alpha.stage2.strategy_tree import Stage2Context, Stage2StrategyNode
from z3alpha.tactics.catalog import PREPROCESS_CATALOG, SOLVER_CATALOG

Action = int | Stage2Action


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


class StrategyTree:
    def __init__(
        self,
        stage,
        logic,
        timeout,
        *,
        logic_config=None,
        stage2_context: "Stage2Context | None" = None,
    ):
        self.logic = logic
        self.timeout = timeout
        self.root = Root()
        if stage == 1:
            self.root.add_children([LinearStrategy(logic, logic_config)])
        else:
            assert stage2_context is not None
            self.root.add_children(
                [Stage2StrategyNode(timeout, stage2_context, if_depth=0)]
            )

    def __str__(self):
        return str(self.root)

    # Return the depth-first first nonterminal node in the tree.
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
