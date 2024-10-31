from pyparsing import Word, alphas, alphanums, oneOf, Forward, Group, Suppress, Literal

solvers = oneOf("Z3STRAT CVC5 STP Bitwuzla")
predicate = Word(alphanums + "_")  # Allows alphanumeric predicates and underscores
time = Word(alphanums)             # Allows alphanumeric time representation

MetaStrat = Forward()
Solver = solvers

TryExpr = Group(Suppress('(') + Suppress("TRY") + MetaStrat + time + MetaStrat + Suppress(')'))
IfExpr = Group(Suppress('(') + Suppress("IF") + predicate + MetaStrat + MetaStrat + Suppress(')'))

# Recursive MetaStrat to allow nested structures
MetaStrat <<= Solver | TryExpr | IfExpr

# Test the parser
def parse_expression(expression):
    try:
        parsed = MetaStrat.parseString(expression, parseAll=True)
        return parsed.asList()
    except Exception as e:
        return f"Parse error: {e}"

# Sample expressions
expressions = [
    "Z3STRAT",
    "(TRY Z3STRAT 5 STP)",
    "(IF condition (TRY Z3STRAT 5 STP) CVC5)",
    "(IF condition (TRY (IF another_condition STP Z3STRAT) 10 CVC5) Bitwuzla)"
]
for expr in expressions:
    print(f"Expression: {expr}")
    print("Parsed:", parse_expression(expr))
    print()
