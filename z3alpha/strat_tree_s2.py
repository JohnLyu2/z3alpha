import copy
from z3alpha.selector import search_next_action
from z3alpha.ast_nodes import ASTNode, TacticNode

MAX_IF_DEPTH = 3
TIMEOUTS = ["v2", "v8", "v32", "v128", "v512"]  # in seconds
PERCENTILES = ["90p", "70p", "50p"]


class OrElseNode(ASTNode):
    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout

    def __str__(self):
        return (
            f"(or-else (try-for {self.children[0]} {self.timeout * 1000}) "
            f"{self.children[1]})"
        )

    def isTerminal(self):
        return True

    def getLnStrats(self, precede_strats, probe_record):
        assert len(precede_strats) == 1
        precede_strat = precede_strats[0][0]
        precede_timeout = precede_strats[0][1]
        assert precede_timeout >= self.timeout
        prec_leftcp = [(copy.deepcopy(precede_strat), self.timeout)]
        prec_rightcp = [(copy.deepcopy(precede_strat), precede_timeout)]
        return self.children[0].getLnStrats(prec_leftcp, probe_record) + self.children[
            1
        ].getLnStrats(prec_rightcp, probe_record)


class PredicateNode(ASTNode):
    def __init__(self, name, prob_stats):
        super().__init__()
        self.name = name
        self.prob_stats = prob_stats
        self.is_selected = False

    def __str__(self):
        value_str = "<Value>"
        if self.is_selected:
            value_str = str(self.value)
        return f"(if (> {self.name} {value_str}) {self.children[0]} {self.children[1]})"

    def isTerminal(self):
        return self.is_selected

    def legalActions(self, rollout=False):
        return PERCENTILES

    def applyRule(self, action, params):
        assert not self.isTerminal()
        assert action in self.legalActions()
        self.value = int(self.prob_stats[self.name][action])
        self.is_selected = True

    def getLnStrats(self, precede_strats, probe_record):
        assert self.isTerminal()
        assert len(precede_strats) == 1
        bench_value = int(probe_record[self.name])
        if bench_value > self.value:
            return self.children[0].getLnStrats(precede_strats, probe_record)
        else:
            return self.children[1].getLnStrats(precede_strats, probe_record)


class ProbeNode(ASTNode):
    def __init__(self, depth, timeout, s2dict):
        super().__init__()
        self.if_depth = depth
        self.timeout = timeout
        self.s2dict = s2dict
        self.probe_stats = s2dict["probe_stats"]
        self.action_dict = {50: "num-consts", 51: "num-exprs", 52: "size"}

    def __str__(self):
        return f"<ProbeNode>"

    def isTerminal(self):
        return False

    def legalActions(self, rollout=False):
        return list(self.action_dict.keys())

    def applyRule(self, action, params):
        assert self.isLeaf()
        assert action in self.legalActions()
        probe_name = self.action_dict[action]
        pred_node = PredicateNode(probe_name, self.probe_stats)
        self.parent.replaceChild(pred_node, self.pos)
        left_strat = S2Strategy(self.timeout, self.s2dict, self.if_depth)
        right_strat = S2Strategy(self.timeout, self.s2dict, self.if_depth)
        pred_node.addChildren([left_strat, right_strat])


class S2Strategy(ASTNode):
    def __init__(self, timeout, s2dict, if_depth):
        super().__init__()
        self.timeout = timeout
        self.s2dict = s2dict
        self.solver_action_dict = s2dict["solver_dict"]
        self.preprocess_action_dict = s2dict["preprocess_dict"]
        self.s1strat_lst = s2dict["s1_strats"]
        self.if_depth = if_depth

    def __str__(self):
        return f"<S2Strategy>"

    def isTerminal(self):
        return False

    def getCurActPath(self):
        reverse_path = []
        cur_node = self.parent
        while cur_node:
            if cur_node.isTactic():
                reverse_path.append(cur_node.s2actID)
            cur_node = cur_node.parent
        return list(reversed(reverse_path))

    def legalTimeoutActions(self):
        return [c for c in TIMEOUTS if int(c[1:]) <= self.timeout]

    def legalActions(self, rollout=False):
        cur_path = self.getCurActPath()
        tac_actions = search_next_action(cur_path, self.s1strat_lst)
        legal_actions = list(tac_actions)
        if (not rollout) and (len(tac_actions) > 1):
            legal_actions += self.legalTimeoutActions()
        if (not rollout) and (self.if_depth < MAX_IF_DEPTH):
            legal_actions.append(3)
        return legal_actions

    def applySolverRule(self, action):
        tactic, tac_params = self.solver_action_dict[action]
        tac_node = TacticNode(tactic, tac_params, action)
        self.parent.replaceChild(tac_node, self.pos)

    def applyThenRule(self, action):
        tactic, tac_params = self.preprocess_action_dict[action]
        tac_node = TacticNode(tactic, tac_params, action)
        self.parent.replaceChild(tac_node, self.pos)
        s2strat = S2Strategy(self.timeout, self.s2dict, MAX_IF_DEPTH)
        tac_node.addChildren([s2strat])

    def applyTimeoutRule(self, action):
        timeout_value = int(action[1:])
        branching_node = OrElseNode(timeout_value)
        tryout_strat = S2Strategy(timeout_value, self.s2dict, MAX_IF_DEPTH)
        default_strat = S2Strategy(self.timeout - timeout_value, self.s2dict, MAX_IF_DEPTH)
        self.parent.replaceChild(branching_node, self.pos)
        branching_node.addChildren([tryout_strat, default_strat])

    def applyIfRule(self):
        self.parent.replaceChild(
            ProbeNode(self.if_depth + 1, self.timeout, self.s2dict), self.pos
        )

    def applyRule(self, action, params):  # params is no use here
        assert self.isLeaf()
        assert action in self.legalActions()
        if action == 3:  # if rule
            self.applyIfRule()
        elif action in TIMEOUTS:
            self.applyTimeoutRule(action)
        elif action in self.solver_action_dict:
            self.applySolverRule(action)
        elif action in self.preprocess_action_dict:
            self.applyThenRule(action)
        else:
            raise Exception("unexpected action")
