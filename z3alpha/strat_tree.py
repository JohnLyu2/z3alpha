from z3alpha.ast_nodes import Root
from z3alpha.strat_tree_s1 import S1Strategy
from z3alpha.strat_tree_s2 import S2Strategy


class StrategyAST:
    def __init__(
        self, stage, logic, timeout, s2config=None,
        logic_config=None,
    ):
        self.logic = logic
        self.timeout = timeout
        self.root = Root()
        if stage == 1:
            self.root.addChildren(
                [S1Strategy(logic, logic_config)]
            )
        else:
            assert s2config
            s2dict = s2config["s2dict"]
            self.root.addChildren([S2Strategy(timeout, s2dict, if_depth=0)])

    def __str__(self):
        return str(self.root)

    @staticmethod
    def _findFstNonTermRec(nonterm_stack):
        if not len(nonterm_stack):
            return None
        node2Search = nonterm_stack.pop()
        if not node2Search.isTerminal():
            return node2Search
        else:
            for childNode in reversed(node2Search.children):
                nonterm_stack.append(childNode)
        return StrategyAST._findFstNonTermRec(nonterm_stack)

    # return the depth-first first nontermial node in the tree; if nonexist, return None
    def findFstNonTerm(self):
        nonterm_stack = [self.root]
        return StrategyAST._findFstNonTermRec(nonterm_stack)

    def isTerminal(self):
        return not bool(self.findFstNonTerm())

    def legalActions(self, rollout=False):
        if self.isTerminal():
            return []
        return self.findFstNonTerm().legalActions(rollout)

    def applyRule(self, action, params):
        assert not self.isTerminal()
        node = self.findFstNonTerm()
        node.applyRule(action, params)

    def get_linear_strategies(self, probe_record):
        assert self.isTerminal()
        return self.root.getLnStrats(self.timeout, probe_record)
