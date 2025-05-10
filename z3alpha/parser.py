SOLVER_TACTICS = ['smt', 'qfnra-nlsat', 'sat', 'qfbv', 'qfnia', 'qfnra', 'qflia', 'qflra', 'arr', 'las']
PREPROCESS_TACTICS = ['simplify', 'propagate-values', 'ctx-simplify', 'elim-uncnstr', 
                      'solve-eqs', 'purify-arith', 'max-bv-sharing', 'aig', 
                      'reduce-bv-size', 'ackermannize_bv', 'lia2card', 'card2bv',
                      'cofactor-term-ite', 'nla2bv', 'bv1-blast', 'bit-blast', 'pb2bv',
                      'propagate-ineqs', 'add-bounds', 'normalize-bounds', 'lia2pb', 
                      'ext_str', 'ext_strSimplify', 'ext_strToRegex', 'ext_strToWE']
TACTIC_LST = SOLVER_TACTICS + PREPROCESS_TACTICS

TACTICAL_LST = ['then', 'if', 'using-params']
PARAM_LST = ['inline_vars', 'seed', 'factor', 'elim_and', 'som', 'blast_distinct', 
             'flat', 'hi_div0', 'local_ctx', 'hoist_mul', 'push_ite_bv', 
             'pull_cheap_ite', 'nla2bv_max_bv_size', 'add_bound_lower', 
             'add_bound_upper', 'pb2bv_all_clauses_limit', 'lia2pb_max_bits', 'random_seed',
             'push_ite_arith', 'hoist_ite', 'arith_lhs']
PRED_PROBES = ['>', 'is-qfbv-eq']
NUM_PROBES = ['size', 'num-consts']
MARKS = ['(', ')', ':']

# from alphasmt.strat_tree import TacticNode

class Token:
  def __init__(self, name, type):
    self.name = name
    self.type = type

  def __str__(self):
    return f"Token \"{self.name}\": {self.type}"

def Strategy_Tokenizer(s):
  # tokenize the strategy string
  # return a list of tokens
  s = s.strip()
  token_lst = []
  i = 0
  while i < len(s):
    if s[i] in MARKS:
      token_name = s[i]
      token_type = 'mark'
    else:
      # token_name is the substring of s[i:] that does not contain a space or parenthesis or :
      token_name = s[i:].split(' ')[0].split('(')[0].split(')')[0].split(':')[0]
      if token_name in TACTIC_LST:
        token_type = 'tactic'
      elif token_name in TACTICAL_LST:
        token_type = 'tactical'
      elif token_name in PARAM_LST:
        token_type = 'param'
      elif token_name in PRED_PROBES or token_name in NUM_PROBES:
        token_type = 'probe'
      # if token_name is a number
      elif token_name.lstrip('-').isdigit():
        token_type = 'number'
      elif token_name == 'true' or token_name == 'false':
        token_type = 'bool'
      else:
        raise Exception(f"Token \"{token_name}\" is not supported")
    token = Token(token_name, token_type)
    # print(f"{token}, start at index {i}")
    token_lst.append(token)
    i += len(token_name)
    while i < len(s) and s[i] == ' ':
      i += 1
  return token_lst

def _is_tactic(t_lst, i):
  if t_lst[i].type == "tactic":
    return True
  if t_lst[i].name == "(" and t_lst[i+1].name == "using-params" and t_lst[i+2].type == "tactic":
    return True
  return False

def parse_tactic(t_lst, i):
  if t_lst[i].type == "tactic":
    return (t_lst[i].name, {}), i+1
  params = {}
  tactic = t_lst[i+2].name
  next_pos = i+3
  while True:
    if t_lst[next_pos].name != ":":
      raise Exception(f"Invalid strategy")
    if t_lst[next_pos+1].type != "param":
      raise Exception(f"Invalid strategy")
    param = t_lst[next_pos+1].name
    if t_lst[next_pos+2].type == "bool" or t_lst[next_pos+2].type == "number":
      value = t_lst[next_pos+2].name
    else:
      raise Exception(f"Invalid strategy")
    params[param] = value
    next_pos += 3
    if t_lst[next_pos].name == ")":
      return (tactic, params), next_pos+1
      

def s1_strat_parse(s1_strat_str):
    # parse a s1 strategy string into a list of TacticNode
    assert len(s1_strat_str) > 0
    tokens = Strategy_Tokenizer(s1_strat_str)
    pos = 0
    if _is_tactic(tokens, pos):
        tactic, pos = parse_tactic(tokens, pos)
        assert pos == len(tokens)
        return [tactic]
    assert tokens[pos].name == "(" and tokens[pos+1].name == "then"
    pos += 2
    tactic_lst = []
    while _is_tactic(tokens, pos):
        tactic, pos = parse_tactic(tokens, pos)
        # print(tactic)
        tactic_lst.append(tactic)
    return tactic_lst

# def _parse_then(t_lst, i):
#   assert t_lst[i].name == "(" and t_lst[i+1].name == "then"
#   next_pos = i+2
#   assert _is_tactic(t_lst, next_pos)
#   root, next_pos = parse_tactic(t_lst, next_pos)
#   prev_tactic = root
#   while _is_tactic(t_lst, next_pos):
#     tactic, next_pos = parse_tactic(t_lst, next_pos)
#     prev_tactic.add_children(tactic)
#     prev_tactic = tactic
#   if t_lst[next_pos].name == "(":
#     strat, next_pos = _parse_strat_rec(t_lst, next_pos)
#     prev_tactic.add_children(strat)
#   if t_lst[next_pos].name != ")":
#     raise Exception(f"Invalid strategy")
#   return root, next_pos+1



# # a recursive function that parses the strategy starting from index i in t_lst
# def _parse_strat_rec(t_lst, i):
#   if _is_tactic(t_lst, i):
#     tactic, next_pos = parse_tactic(t_lst, i)
#     return tactic, next_pos
#   else:
#     if t_lst[i].name != "(" or t_lst[i+1].type != "tactical":
#       raise Exception(f"Invalid strategy")
#     if t_lst[i+1].name == "then":
#       return _parse_then(t_lst, i)
#     if t_lst[i+1].name == "if":
#       if t_lst[i+2].name != "(":
#         pred = t_lst[i+2].name
#         value = None
#         next_pos = i+3
#         if pred not in PRED_PROBES:
#           raise Exception(f"Invalid strategy")
#       else:
#         if t_lst[i+3].name != ">" or t_lst[i+4].name not in NUM_PROBES or t_lst[i+5].type != "number" or t_lst[i+6].name != ")": # for now only allows >
#           raise Exception(f"Invalid strategy")
#         pred = t_lst[i+4].name
#         value = t_lst[i+5].name
#         next_pos = i+7
#       if_node = IfPredicate(pred, value)
#       left_strat, next_pos = _parse_strat_rec(t_lst, next_pos)
#       right_strat, next_pos = _parse_strat_rec(t_lst, next_pos)
#       if_node.add_children(left_strat, right_strat)
#       if t_lst[next_pos].name != ")":
#         raise Exception(f"Invalid strategy")
#       next_pos += 1
#       return if_node, next_pos


# def Strategy_Parse(s):
#   assert len(s) > 0
#   # parse the strategy string into a tree
#   # return the root node
#   tokens = Strategy_Tokenizer(s)
#   strat, p = _parse_strat_rec(tokens, 0)
#   return strat