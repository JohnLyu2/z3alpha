import copy
from z3alpha.selector import search_next_action

MAX_IF_DEPTH = 3 
TIMEOUTS = ["v2", "v8", "v32", "v128", "v512"] # in seconds
PERCENTILES = ["90p", "70p", "50p"]

class ASTNode():
    def __init__(self):
        self.children = []
        self.parent = None
        
    def isLeaf(self):
        return len(self.children) == 0

    def isTactic(self):
        return False

    # make change this to unify with the way in S2Strategy
    def applyRule(self, action, params):
        assert (self.isLeaf())
        assert (action in self.legalActions())
        func = self.action_dict[action]
        func(params)

    # only used for leaf nodes
    def addChildren(self, children):
        assert (self.isLeaf())
        assert (len(children) > 0)
        self.children = children
        for i in range(len(children)):
            children[i].parent = self
            children[i].pos = i

    # replace the pos'th child with the new children
    def replaceChild(self, child, pos):
        assert (not self.isLeaf())
        assert (pos < len(self.children))
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
    
class TimeOutNode0(ASTNode):
    def __init__(self, remain_time, s2dict):
        super().__init__()
        self.remain_time = remain_time
        self.s2dict = s2dict

    def __str__(self):
        return f"<TimeOutNode>"

    def isTerminal(self):
        return False

    def legalActions(self, rollout = False):
        candidates = TIMEOUTS
        return [c for c in candidates if int(c[1:]) <= self.remain_time]

    def applyRule(self, action, params):
        assert (self.isLeaf())
        assert (action in self.legalActions())
        a_value = int(action[1:])
        branchingNode = TimeOutNode1(a_value)
        # currently no branching after timeout tryout
        tryout_strat = S2Strategy(a_value, self.s2dict, MAX_IF_DEPTH)
        default_strat = S2Strategy(self.remain_time - a_value, self.s2dict, MAX_IF_DEPTH)
        self.parent.replaceChild(branchingNode, self.pos)
        branchingNode.addChildren([tryout_strat, default_strat])

class TimeOutNode1(ASTNode):
    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout

    def __str__(self):
        return f"(or-else (try-for {self.children[0]} {self.timeout * 1000}) {self.children[1]})"

    def isTerminal(self):
        return True

    def getLnStrats(self, precede_strats, probe_record):
        assert len(precede_strats) == 1
        precede_strat = precede_strats[0][0]
        precede_timeout = precede_strats[0][1]
        assert (precede_timeout >= self.timeout)
        prec_leftcp = [(copy.deepcopy(precede_strat), self.timeout)]
        prec_rightcp = [(copy.deepcopy(precede_strat), precede_timeout)]
        return self.children[0].getLnStrats(prec_leftcp, probe_record) + self.children[1].getLnStrats(prec_rightcp, probe_record)

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
    
    def legalActions(self, rollout = False):
        return PERCENTILES
    
    def applyRule(self, action, params):
        assert (not self.isTerminal())
        assert (action in self.legalActions())
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
        self.probe_stats = s2dict['probe_stats']
        # for now no binary predicate; do not consider different predicates for different logics
        self.action_dict = {
            50: "num-consts", 
            51: "num-exprs",
            52: "size"
        }

    def __str__(self):
        return f"<ProbeNode>"
    
    def isTerminal(self):
        return False
    
    def legalActions(self, rollout = False):
        return list(self.action_dict.keys())

    def applyRule(self, action, params):
        assert (self.isLeaf())
        assert (action in self.legalActions())
        probe_name = self.action_dict[action]
        pred_node = PredicateNode(probe_name, self.probe_stats)
        self.parent.replaceChild(pred_node, self.pos)
        left_strat = S2Strategy(self.timeout, self.s2dict, self.if_depth)
        right_strat = S2Strategy(self.timeout, self.s2dict, self.if_depth)
        pred_node.addChildren([left_strat, right_strat])

class TacticNode(ASTNode):
    def __init__(self, name, params, s2actID = None): # tactic terminals do not have children 
        super().__init__()
        self.name = name
        self.params = params
        self.s2actID = s2actID
        # self._setParamMABs()

    def _tactic_str(self):
        if not self.params: return f"{self.name}"
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
        if self.isLeaf(): return False
        if self.parent.isTactic(): return False
        return True

    # in stage 2, tactic parameters are part of the name string; always return False for stage 2
    def hasParams(self):
        return not self.params

    def isTerminal(self):
        return True

    def getLnStrats(self, precede_strats, probe_record):
        for strat_tup in precede_strats:
            strat_tup[0].append(self.s2actID)
        if self.isLeaf():
            return precede_strats
        return self.children[0].getLnStrats(precede_strats, probe_record)

class PreprocessTactic(ASTNode):
    def __init__(self, logic):
        super().__init__()
        self.logic = logic
        self.action_dict = {
            20: "simplify", 
            21: "propagate-values",
            22: "ctx-simplify",
            23: "elim-uncnstr",
            24: "solve-eqs",
            # 25 - 31 are QF_BV only
            25: "purify-arith",
            26: "max-bv-sharing",
            27: "aig",
            28: "reduce-bv-size",
            29: "ackermannize_bv",
            # 30: "bit-blast", # require simplification beforehand; otherwise report error
            # 32 - 34 are QF_NIA only
            32: "lia2card",
            33: "card2bv",
            34: "cofactor-term-ite",
            # 35 - 38 are QF_LIA only
            35: "propagate-ineqs",
            36: "add-bounds",
            37: "normalize-bounds",
            38: "lia2pb",
            # 40 - 43 are QF_S only
            40: "ext_str",
            41: "ext_strSimplify",
            42: "ext_strToRegex",
            43: "ext_strToWE",

        }

    def __str__(self):
        # if self.isLeaf():
        #     return f"<PreprocessTactic>({self.logic})"
        return f"<PreprocessTactic>({self.logic}) {self.children[0]}"

    def isTerminal(self):
        return False

    def legalActions(self, rollout = False):
        actions = [i for i in range(20,25)]
        if self.logic == "default":
            return actions
        elif self.logic == "QF_BV":
            return actions + [i for i in range(25,30)]
        elif self.logic == "QF_NIA":
            return actions + [i for i in range(32,35)]
        elif self.logic == "QF_NRA" or self.logic == "SAT":
            return actions
        elif self.logic == "QF_LIA":
            return actions + [i for i in range(35,39)]
        elif self.logic == "QF_LRA":
            return actions
        elif self.logic == "QF_S":
            return actions + [40, 41, 42, 43]
        else: 
            raise Exception("unexpected smt logic")

    def applyRule(self, action, params):
        assert (action in self.legalActions())
        tactic_name = self.action_dict[action]
        # params = TACTIC_PARAMS[tactic_name] if tactic_name in TACTIC_PARAMS else None
        selected = TacticNode(tactic_name, params, action)
        self.parent.replaceChild(selected, self.pos)
        selected.addChildren(self.children)

class SolverTactic(ASTNode):
    def __init__(self, logic):
        super().__init__()
        self.logic = logic
        self.action_dict = {
            10: "smt",
            11: "qfnra-nlsat", # for QF_NIA and QF_NRA
            12: "sat",
            13: "qfbv", # only for QF_BV
            14: "qfnia", # only for QF_NIA
            15: "qfnra", # only for QF_NRA
            16: "qflia", # only for QF_LIA
            17: "qflra", # only for QF_LRA
            18: "arr", # only for QF_S
            19: "las", # only for QF_UFBV

        }

    def __str__(self):
        return f"<SolverTactic>({self.logic})"

    def isTerminal(self):
        return False

    def legalActions(self, rollout = False):
        actions = [10]
        if self.logic == "default":
            return actions
        elif self.logic == "QF_BV":
            return actions + [13]
        elif self.logic == "QF_NIA":
            return actions + [11, 14]
        elif self.logic == "QF_NRA":
            return actions + [11, 15]
        elif self.logic == "QF_LIA":
            return actions + [16]
        elif self.logic == "QF_LRA":
            return actions + [17]
        elif self.logic == "SAT":
            return actions + [12]
        elif self.logic == "QF_S":
            return actions + [18, 19]
        else: 
            raise Exception("unexpected smt logic")

    def applyRule(self, action, params):
        assert (self.isLeaf())
        assert (action in self.legalActions())
        tactic_name = self.action_dict[action]
        selected = TacticNode(tactic_name, params, action)
        self.parent.replaceChild(selected, self.pos)

class S1Strategy(ASTNode):
    def __init__(self, logic):
        super().__init__()
        self.logic = logic
        self.action_dict = {
            0: self.applySolverRule,  # <S1Strategy> := <SolverTactic>
            1: self.applyThenRule,  # <S1Strategy> := (then <PreprocessTactic> <S1Strategy>)
            5: self.applyNla2BVRule,  # <S1Strategy>(QF_NIA/QF_NRA) := (then nla2bv <S1Strategy>(QF_BV))
            7: self.applyBitBlastRule,  # <S1Strategy>(BV) := (then simplify bit-blast <S1Strategy>(SAT))
            8: self.applyPb2BvRule # <S1Strategy>(QF_LIA) := (then pb2bv <S1Strategy>(QF_BV))
        }

    def __str__(self):
        return f"<S1Strategy>({self.logic})"

    def isTerminal(self):
        return False
        
    def legalActions(self, rollout = False):
        actions = [0, 1]
        if (not rollout) and (self.logic == "QF_NIA" or self.logic == "QF_NRA"):
            actions.append(5)
        if (self.logic == "QF_BV"):
            actions.append(7)
        if (self.logic == "QF_LIA"):
            # actions.append(8)
            pass
        return actions

    def applySolverRule(self, params):
        assert self.parent
        self.parent.replaceChild(SolverTactic(self.logic), self.pos)

    def applyThenRule(self, params):
        assert self.parent
        preprocessor = PreprocessTactic(self.logic)
        self.parent.replaceChild(preprocessor, self.pos)
        preprocessor.addChildren([S1Strategy(self.logic)])

    def applyNla2BVRule(self, params):
        nla2bv_node = TacticNode("nla2bv", params)
        self.parent.replaceChild(nla2bv_node, self.pos)
        qf_bv_strat = S1Strategy("QF_BV")
        nla2bv_node.addChildren([qf_bv_strat])

    def applyPb2BvRule(self, params):
        pb2bv_node = TacticNode("pb2bv", params)
        self.parent.replaceChild(pb2bv_node, self.pos)
        qf_bv_strat = S1Strategy("QF_BV")
        pb2bv_node.addChildren([qf_bv_strat])

    def applyBitBlastRule(self, params):
        simplify_node = TacticNode("simplify", None)
        bit_blast_node = TacticNode("bit-blast", params)
        self.parent.replaceChild(simplify_node, self.pos)
        simplify_node.addChildren([bit_blast_node])
        bit_blast_node.addChildren([S1Strategy("SAT")])

class S2Strategy(ASTNode):
    def __init__(self, timeout, s2dict, if_depth):
        super().__init__()
        self.timeout = timeout
        self.action_lst = [2, # timeout rule
                           3, # if rule
                           ]
        self.s2dict = s2dict
        self.solver_action_dict = s2dict['solver_dict']
        self.preprocess_action_dict = s2dict['preprocess_dict']
        self.s1strat_lst = s2dict['s1_strats']
        # self.probe_stats = s2dict['probe_stats']
        self.if_depth = if_depth
        # self.probe_records = probe_records

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

    def legalActions(self, rollout = False): 
        cur_path = self.getCurActPath()
        tac_actions = search_next_action(cur_path, self.s1strat_lst)
        legal_actions = tac_actions
        if (not rollout) and (len(tac_actions) > 1) and (self.timeout > int(TIMEOUTS[0][1:])):
            legal_actions.append(2) # timeout rule
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

    def applyTimeoutRule(self):
        self.parent.replaceChild(TimeOutNode0(self.timeout, self.s2dict), self.pos)

    def applyIfRule(self):
        self.parent.replaceChild(ProbeNode(self.if_depth+1, self.timeout, self.s2dict), self.pos)

    def applyRule(self, action, params): # params is no use here
        assert (self.isLeaf())
        assert (action in self.legalActions())
        if action == 2: # timeout rule
            self.applyTimeoutRule()
        elif action == 3: # if rule
            self.applyIfRule()
        elif action in self.solver_action_dict:
            self.applySolverRule(action)
        elif action in self.preprocess_action_dict:
            self.applyThenRule(action)
        else:
            raise Exception("unexpected action")


class StrategyAST():
    def __init__(self, stage, logic, timeout, s2config = None):
        self.logic = logic
        self.timeout = timeout
        self.root = Root()
        if stage == 1:
            self.root.addChildren([S1Strategy(logic)])
        else:
            assert s2config
            s2dict = s2config['s2dict']
            self.root.addChildren([S2Strategy(timeout, s2dict, if_depth = 0)])
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

    def legalActions(self, rollout = False):
        if self.isTerminal(): return []
        return self.findFstNonTerm().legalActions(rollout)

    def applyRule(self, action, params):
        assert (not self.isTerminal())
        node = self.findFstNonTerm()
        node.applyRule(action, params)

    def get_linear_strategies(self, probe_record):
        assert self.isTerminal()
        return self.root.getLnStrats(self.timeout, probe_record)
         