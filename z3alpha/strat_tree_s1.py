from z3alpha.ast_nodes import ASTNode, TacticNode
from z3alpha.tactics.catalog import PREPROCESS_CATALOG, SOLVER_CATALOG


class S1Strategy(ASTNode):
    def __init__(self, logic, logic_config):
        super().__init__()
        self.logic = logic
        self.logic_config = logic_config

    def __str__(self):
        return f"<S1Strategy>({self.logic})"

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
            selected.add_children([S1Strategy(self.logic, self.logic_config)])
            return
        raise Exception("unexpected action")

