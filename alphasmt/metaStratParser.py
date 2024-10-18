SOLVER = ['z3', 'cvc4', 'cvc5', 'z3Noodler']
SOLVER_TACTICS_z3 = ['smt', 'qfnra-nlsat', 'sat', 'qfbv', 'qfnia', 'qfnra', 'qflia', 'qflra', 'arr', 'las']
PREPROCESS_TACTICS_z3 = ['simplify', 'propagate-values', 'ctx-simplify', 'elim-uncnstr',
                      'solve-eqs', 'purify-arith', 'max-bv-sharing', 'aig',
                      'reduce-bv-size', 'ackermannize_bv', 'lia2card', 'card2bv',
                      'cofactor-term-ite', 'nla2bv', 'bv1-blast', 'bit-blast', 'pb2bv',
                      'propagate-ineqs', 'add-bounds', 'normalize-bounds', 'lia2pb',
                      'ext_str', 'ext_strSimplify', 'ext_strToRegex', 'ext_strToWE']
TACTIC_LST_z3 = SOLVER_TACTICS_z3 + PREPROCESS_TACTICS_z3

TACTICAL_LST_z3 = ['then', 'if', 'using-params']
PARAM_LST_z3 = ['inline_vars', 'seed', 'factor', 'elim_and', 'som', 'blast_distinct',
             'flat', 'hi_div0', 'local_ctx', 'hoist_mul', 'push_ite_bv',
             'pull_cheap_ite', 'nla2bv_max_bv_size', 'add_bound_lower',
             'add_bound_upper', 'pb2bv_all_clauses_limit', 'lia2pb_max_bits', 'random_seed',
             'push_ite_arith', 'hoist_ite', 'arith_lhs']
PRED_PROBES_z3 = ['>', 'is-qfbv-eq']
NUM_PROBES_z3 = ['size', 'num-consts']



MARKS = ['(', ')', ':', '[', ']']

class Token:
  def __init__(self, name, type):
    self.name = name
    self.type = type

  def __str__(self):
    return f"Token \"{self.name}\": {self.type}"


def parseForZ3(token_lst, s, i):
    # tokenize the strategy string
    # return a list of tokens
    while i < len(s):
        if s[i] in MARKS:
            if s[i] == '[':
                return token_lst, i
            token_name = s[i]
            token_type = 'mark'
        else:
            # token_name is the substring of s[i:] that does not contain a space or parenthesis or :
            token_name = s[i:].split(' ')[0].split('(')[0].split(')')[0].split(':')[0].split(':')[0].split(':')[0]
            if token_name in TACTIC_LST_z3:
                token_type = 'tactic'
            elif token_name in TACTICAL_LST_z3:
                token_type = 'tactical'
            elif token_name in PARAM_LST_z3:
                token_type = 'param'
            elif token_name in PRED_PROBES_z3 or token_name in NUM_PROBES_z3:
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


def Tokenizer(s):
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
      token_name = s[i:].split(' ')[0].split('(')[0].split(')')[0].split(':')[0].split('[')[0].split(']')[0]
      if token_name in SOLVER:
        token_type = 'solver'
        if token_name == 'z3Noodler':
            token = Token(token_name, token_type)
            token_lst.append(token)
            i += len(token_name)
            token_lst , i = parseForz3Noodler(token_lst, s, i)
        elif token_name == 'cvc4':
            token = Token(token_name, token_type)
            # print(f"{token}, start at index {i}")
            token_lst.append(token)
            i += len(token_name)
            token_lst , i = parseForCvc4(token_lst, s, i)
        elif token_name == 'cvc5':
            token = Token(token_name, token_type)
            # print(f"{token}, start at index {i}")
            token_lst.append(token)
            i += len(token_name)
            token_lst , i = parseForCvc5(token_lst, s, i)
        elif token_name == 'z3':
            token = Token(token_name, token_type)
            # print(f"{token}, start at index {i}")
            token_lst.append(token)
            i += len(token_name)
            token_lst , i = parseForZ3(token_lst, s, i)
      else:
        raise Exception(f"Token \"{token_name}\" is not supported")
    while i < len(s) and s[i] == ' ':
      i += 1
  return token_lst