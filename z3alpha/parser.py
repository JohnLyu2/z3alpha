from z3alpha.tactics.catalog import (
    PREPROCESS_TACTICS,
    SOLVER_TACTICS,
    SUPPORTED_TACTIC_PARAMS,
)

TACTIC_LST = SOLVER_TACTICS + PREPROCESS_TACTICS

TACTICAL_LST = ["then", "if", "using-params"]
PARAM_LST = SUPPORTED_TACTIC_PARAMS
PRED_PROBES = [">", "is-qfbv-eq"]
NUM_PROBES = ["size", "num-consts"]
MARKS = ["(", ")", ":"]

# from alphasmt.strat_tree import TacticNode


class Token:
    def __init__(self, name, type):
        self.name = name
        self.type = type

    def __str__(self):
        return f'Token "{self.name}": {self.type}'


def Strategy_Tokenizer(s):
    # tokenize the strategy string
    # return a list of tokens
    s = s.strip()
    token_lst = []
    i = 0
    while i < len(s):
        if s[i] in MARKS:
            token_name = s[i]
            token_type = "mark"
        else:
            # token_name is the substring of s[i:] that does not contain a space or parenthesis or :
            token_name = s[i:].split(" ")[0].split("(")[0].split(")")[0].split(":")[0]
            if token_name in TACTIC_LST:
                token_type = "tactic"
            elif token_name in TACTICAL_LST:
                token_type = "tactical"
            elif token_name in PARAM_LST:
                token_type = "param"
            elif token_name in PRED_PROBES or token_name in NUM_PROBES:
                token_type = "probe"
            # if token_name is a number
            elif token_name.lstrip("-").isdigit():
                token_type = "number"
            elif token_name == "true" or token_name == "false":
                token_type = "bool"
            else:
                raise Exception(f'Token "{token_name}" is not supported')
        token = Token(token_name, token_type)
        # print(f"{token}, start at index {i}")
        token_lst.append(token)
        i += len(token_name)
        while i < len(s) and s[i] == " ":
            i += 1
    return token_lst


def _is_tactic(t_lst, i):
    if t_lst[i].type == "tactic":
        return True
    if (
        t_lst[i].name == "("
        and t_lst[i + 1].name == "using-params"
        and t_lst[i + 2].type == "tactic"
    ):
        return True
    return False


def parse_tactic(t_lst, i):
    if t_lst[i].type == "tactic":
        return (t_lst[i].name, {}), i + 1
    params = {}
    tactic = t_lst[i + 2].name
    next_pos = i + 3
    while True:
        if t_lst[next_pos].name != ":":
            raise Exception(f"Invalid strategy")
        if t_lst[next_pos + 1].type != "param":
            raise Exception(f"Invalid strategy")
        param = t_lst[next_pos + 1].name
        if t_lst[next_pos + 2].type == "bool" or t_lst[next_pos + 2].type == "number":
            value = t_lst[next_pos + 2].name
        else:
            raise Exception(f"Invalid strategy")
        params[param] = value
        next_pos += 3
        if t_lst[next_pos].name == ")":
            return (tactic, params), next_pos + 1


def s1_strat_parse(s1_strat_str):
    # parse a s1 strategy string into a list of TacticNode
    assert len(s1_strat_str) > 0
    tokens = Strategy_Tokenizer(s1_strat_str)
    pos = 0
    if _is_tactic(tokens, pos):
        tactic, pos = parse_tactic(tokens, pos)
        assert pos == len(tokens)
        return [tactic]
    assert tokens[pos].name == "(" and tokens[pos + 1].name == "then"
    pos += 2
    tactic_lst = []
    while _is_tactic(tokens, pos):
        tactic, pos = parse_tactic(tokens, pos)
        # print(tactic)
        tactic_lst.append(tactic)
    return tactic_lst
