import copy
from dataclasses import dataclass
from typing import Any

from z3alpha.ast_nodes import ASTNode, Root, TacticNode
from z3alpha.stage2.utils import next_actions_from_prefix
from z3alpha.strategy_tree import SearchTreeBase


@dataclass(frozen=True)
class Stage2Context:
    seed_action_sequences: list[list[int]]
    solver_actions: dict[int, tuple]
    preprocess_actions: dict[int, tuple]
    result_cache: dict[tuple[int, ...], list[tuple[bool, float, str]]]
    probe_stats: dict[str, dict[str, int]]
    probe_records: list[dict[str, int]]


@dataclass(frozen=True)
class Stage2Action:
    kind: str
    value: int | str | None = None

    def __str__(self) -> str:
        if self.kind == "if_rule":
            return "if_rule"
        return f"{self.kind}:{self.value}"


Action = Stage2Action

MAX_IF_DEPTH = 3
TIMEOUTS = ["v2", "v8", "v32", "v128", "v512"]  # in seconds
PERCENTILES = ["90p", "70p", "50p"]
ACTION_IF_RULE = 3
ACTION_PROBE_NUM_CONSTS = 50
ACTION_PROBE_NUM_EXPRS = 51
ACTION_PROBE_SIZE = 52


class OrElseNode(ASTNode):
    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout

    def __str__(self):
        return (
            f"(or-else (try-for {self.children[0]} {self.timeout * 1000}) "
            f"{self.children[1]})"
        )

    def is_terminal(self):
        return True

    def get_ln_strats(self, precede_strats, probe_record):
        assert len(precede_strats) == 1
        precede_strat = precede_strats[0][0]
        precede_timeout = precede_strats[0][1]
        assert precede_timeout >= self.timeout
        prec_leftcp = [(copy.deepcopy(precede_strat), self.timeout)]
        prec_rightcp = [(copy.deepcopy(precede_strat), precede_timeout)]
        return self.children[0].get_ln_strats(prec_leftcp, probe_record) + self.children[
            1
        ].get_ln_strats(prec_rightcp, probe_record)


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

    def is_terminal(self):
        return self.is_selected

    def legal_actions(self, rollout: bool = False) -> list[str]:
        return PERCENTILES

    def apply_rule(self, action: str, params: Any) -> None:
        assert not self.is_terminal()
        assert action in self.legal_actions()
        self.value = int(self.prob_stats[self.name][action])
        self.is_selected = True

    def get_ln_strats(self, precede_strats, probe_record):
        assert self.is_terminal()
        assert len(precede_strats) == 1
        bench_value = int(probe_record[self.name])
        if bench_value > self.value:
            return self.children[0].get_ln_strats(precede_strats, probe_record)
        return self.children[1].get_ln_strats(precede_strats, probe_record)


class ProbeNode(ASTNode):
    def __init__(self, depth, timeout, stage2_context: Stage2Context):
        super().__init__()
        self.if_depth = depth
        self.timeout = timeout
        self.stage2_context = stage2_context
        self.probe_stats = stage2_context.probe_stats
        self.action_dict = {
            ACTION_PROBE_NUM_CONSTS: "num-consts",
            ACTION_PROBE_NUM_EXPRS: "num-exprs",
            ACTION_PROBE_SIZE: "size",
        }

    def __str__(self):
        return "<ProbeNode>"

    def is_terminal(self):
        return False

    def legal_actions(self, rollout: bool = False) -> list[int]:
        return list(self.action_dict.keys())

    def apply_rule(self, action: int, params: Any) -> None:
        assert self.is_leaf()
        assert action in self.legal_actions()
        probe_name = self.action_dict[action]
        pred_node = PredicateNode(probe_name, self.probe_stats)
        self.parent.replace_child(pred_node, self.pos)
        left_strat = Stage2StrategyNode(self.timeout, self.stage2_context, self.if_depth)
        right_strat = Stage2StrategyNode(self.timeout, self.stage2_context, self.if_depth)
        pred_node.add_children([left_strat, right_strat])


class Stage2StrategyNode(ASTNode):
    def __init__(self, timeout, stage2_context: Stage2Context, if_depth):
        super().__init__()
        self.timeout = timeout
        self.stage2_context = stage2_context
        self.solver_action_dict = stage2_context.solver_actions
        self.preprocess_action_dict = stage2_context.preprocess_actions
        self.s1strat_lst = stage2_context.seed_action_sequences
        self.if_depth = if_depth

    def __str__(self):
        return "<Stage2StrategyNode>"

    def is_terminal(self):
        return False

    def get_cur_act_path(self):
        reverse_path = []
        cur_node = self.parent
        while cur_node:
            if cur_node.is_tactic():
                reverse_path.append(cur_node.branched_action_id)
            cur_node = cur_node.parent
        return list(reversed(reverse_path))

    def legal_timeout_actions(self) -> list[str]:
        return [c for c in TIMEOUTS if int(c[1:]) <= self.timeout]

    def legal_actions(self, rollout: bool = False) -> list[Action]:
        cur_path = self.get_cur_act_path()
        tac_actions = next_actions_from_prefix(cur_path, self.s1strat_lst)
        legal_actions: list[Stage2Action] = [
            Stage2Action("tactic", tac_action) for tac_action in tac_actions
        ]
        if (not rollout) and (len(tac_actions) > 1):
            legal_actions += [
                Stage2Action("timeout", timeout_value)
                for timeout_value in self.legal_timeout_actions()
            ]
        if (not rollout) and (self.if_depth < MAX_IF_DEPTH):
            legal_actions.append(Stage2Action("if_rule"))
        return legal_actions

    def apply_solver_rule(self, action: int) -> None:
        tactic, tac_params = self.solver_action_dict[action]
        tac_node = TacticNode(tactic, tac_params, action)
        self.parent.replace_child(tac_node, self.pos)

    def apply_then_rule(self, action: int) -> None:
        tactic, tac_params = self.preprocess_action_dict[action]
        tac_node = TacticNode(tactic, tac_params, action)
        self.parent.replace_child(tac_node, self.pos)
        s2strat = Stage2StrategyNode(self.timeout, self.stage2_context, MAX_IF_DEPTH)
        tac_node.add_children([s2strat])

    def apply_timeout_rule(self, action: str) -> None:
        timeout_value = int(action[1:])
        branching_node = OrElseNode(timeout_value)
        tryout_strat = Stage2StrategyNode(timeout_value, self.stage2_context, MAX_IF_DEPTH)
        default_strat = Stage2StrategyNode(
            self.timeout - timeout_value, self.stage2_context, MAX_IF_DEPTH
        )
        self.parent.replace_child(branching_node, self.pos)
        branching_node.add_children([tryout_strat, default_strat])

    def apply_if_rule(self) -> None:
        self.parent.replace_child(
            ProbeNode(self.if_depth + 1, self.timeout, self.stage2_context), self.pos
        )

    def apply_rule(self, action: Action, params: Any) -> None:
        assert self.is_leaf()
        assert action in self.legal_actions()
        if action.kind == "if_rule":
            self.apply_if_rule()
        elif action.kind == "timeout":
            self.apply_timeout_rule(str(action.value))
        elif action.kind == "tactic" and action.value in self.solver_action_dict:
            self.apply_solver_rule(int(action.value))
        elif action.kind == "tactic" and action.value in self.preprocess_action_dict:
            self.apply_then_rule(int(action.value))
        else:
            raise Exception("unexpected action")


class BranchedStrategyTree(SearchTreeBase):
    """MCTS search tree for conditional (if / probe) strategies."""

    def __init__(self, logic, timeout, *, context: Stage2Context):
        self.logic = logic
        self.timeout = timeout
        self.root = Root()
        self.root.add_children(
            [Stage2StrategyNode(timeout, context, if_depth=0)]
        )
