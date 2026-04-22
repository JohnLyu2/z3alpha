class ASTNode:
    def __init__(self):
        self.children = []
        self.parent = None

    def is_leaf(self):
        return len(self.children) == 0

    def is_tactic(self):
        return False

    # only used for leaf nodes
    def add_children(self, children):
        assert self.is_leaf()
        assert len(children) > 0
        self.children = children
        for i in range(len(children)):
            children[i].parent = self
            children[i].pos = i

    # replace the pos'th child with the new children
    def replace_child(self, child, pos):
        assert not self.is_leaf()
        assert pos < len(self.children)
        self.children[pos] = child
        child.parent = self
        child.pos = pos

    # Backward-compatible aliases.
    def isLeaf(self):
        return self.is_leaf()

    def isTactic(self):
        return self.is_tactic()

    def addChildren(self, children):
        return self.add_children(children)

    def replaceChild(self, child, pos):
        return self.replace_child(child, pos)


class Root(ASTNode):
    def __init__(self):
        super().__init__()

    def __str__(self):
        assert len(self.children) == 1
        return str(self.children[0])

    def is_terminal(self):
        return True

    def get_ln_strats(self, timeout, probe_record):
        return self.children[0].get_ln_strats([([], timeout)], probe_record)

    # Backward-compatible aliases.
    def isTerminal(self):
        return self.is_terminal()

    def getLnStrats(self, timeout, probe_record):
        return self.get_ln_strats(timeout, probe_record)


class TacticNode(ASTNode):
    def __init__(
        self, name, params, s2actID=None
    ):  # tactic terminals do not have children
        super().__init__()
        self.name = name
        self.params = params
        self.s2actID = s2actID

    def _tactic_str(self):
        if not self.params:
            return f"{self.name}"
        tactic_str = f"(using-params {self.name}"
        for param in self.params:
            paramStr = " :" + param + " " + str(self.params[param])
            tactic_str += paramStr
        tactic_str += ")"
        return tactic_str

    def __str__(self):
        tactic_str = self._tactic_str()
        if self.isLeaf():
            return tactic_str
        elif self._is_first_then():
            return f"(then {tactic_str} {self.children[0]})"
        else:
            return f"{tactic_str} {self.children[0]}"

    def is_tactic(self):
        return True

    def _is_first_then(self):
        if self.is_leaf():
            return False
        if self.parent.is_tactic():
            return False
        return True

    def is_terminal(self):
        return True

    def get_ln_strats(self, precede_strats, probe_record):
        for strat_tup in precede_strats:
            strat_tup[0].append(self.s2actID)
        if self.is_leaf():
            return precede_strats
        return self.children[0].get_ln_strats(precede_strats, probe_record)

    # Backward-compatible aliases.
    def isTactic(self):
        return self.is_tactic()

    def isTerminal(self):
        return self.is_terminal()

    def getLnStrats(self, precede_strats, probe_record):
        return self.get_ln_strats(precede_strats, probe_record)
