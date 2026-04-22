class ASTNode:
    def __init__(self):
        self.children = []
        self.parent = None

    def isLeaf(self):
        return len(self.children) == 0

    def isTactic(self):
        return False

    # only used for leaf nodes
    def addChildren(self, children):
        assert self.isLeaf()
        assert len(children) > 0
        self.children = children
        for i in range(len(children)):
            children[i].parent = self
            children[i].pos = i

    # replace the pos'th child with the new children
    def replaceChild(self, child, pos):
        assert not self.isLeaf()
        assert pos < len(self.children)
        self.children[pos] = child
        child.parent = self
        child.pos = pos


class Root(ASTNode):
    def __init__(self):
        super().__init__()

    def __str__(self):
        assert len(self.children) == 1
        return str(self.children[0])

    def isTerminal(self):
        return True

    def getLnStrats(self, timeout, probe_record):
        return self.children[0].getLnStrats([([], timeout)], probe_record)


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

    def isTactic(self):
        return True

    def _is_first_then(self):
        if self.isLeaf():
            return False
        if self.parent.isTactic():
            return False
        return True

    def isTerminal(self):
        return True

    def getLnStrats(self, precede_strats, probe_record):
        for strat_tup in precede_strats:
            strat_tup[0].append(self.s2actID)
        if self.isLeaf():
            return precede_strats
        return self.children[0].getLnStrats(precede_strats, probe_record)
