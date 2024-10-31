from pyparsing import Word, alphas, alphanums, oneOf, Forward, Group, Suppress, Literal, infixNotation, opAssoc

# Define the terminals
solvers = oneOf("Z3STRAT CVC5 STP Bitwuzla")
time = Word(alphanums)  # Represents a numeric or string time value

# Define boolean operators and comparisons for the predicate
variable = Word(alphas, alphanums + "_")  # Alphanumeric with underscore for variables
comparison_operator = oneOf("== != > >= < <=")

# Build the comparison expression
condition_expr = Forward()
comparison_expr = Group(variable + comparison_operator + variable)
condition_expr <<= infixNotation(comparison_expr, [
    ("and", 2, opAssoc.LEFT),
    ("or", 2, opAssoc.LEFT),
    ("not", 1, opAssoc.RIGHT)
])

# Define the grammar rules
MetaStrat = Forward()
Solver = solvers

# Define TRY and IF expressions with parentheses
TryExpr = Group(Suppress('(') + Literal("TRY") + MetaStrat + time + MetaStrat + Suppress(')'))
IfExpr = Group(Suppress('(') + Literal("IF") + condition_expr + MetaStrat + MetaStrat + Suppress(')'))

# Define recursive MetaStrat to allow nested structures
MetaStrat <<= Solver | TryExpr | IfExpr


# Function to evaluate conditions
def evaluate_condition(parsed_condition, variables={}):
    if isinstance(parsed_condition, str):
        return variables.get(parsed_condition, False)

    op = parsed_condition[1]
    left = parsed_condition[0]
    right = parsed_condition[2]

    if op == "==":
        return variables.get(left) == variables.get(right)
    elif op == "!=":
        return variables.get(left) != variables.get(right)
    elif op == ">":
        return variables.get(left) > variables.get(right)
    elif op == ">=":
        return variables.get(left) >= variables.get(right)
    elif op == "<":
        return variables.get(left) < variables.get(right)
    elif op == "<=":
        return variables.get(left) <= variables.get(right)
    elif op == "and":
        return evaluate_condition(parsed_condition[0], variables) and evaluate_condition(parsed_condition[2], variables)
    elif op == "or":
        return evaluate_condition(parsed_condition[0], variables) or evaluate_condition(parsed_condition[2], variables)
    elif op == "not":
        return not evaluate_condition(parsed_condition[1], variables)
    return False



def run(solver, time=20):
    print(f"Running {solver} for {time} seconds")


# Recursive function to evaluate MetaStrat expressions
def evaluate_meta_strat(parsed_expr, variables={}):
    if isinstance(parsed_expr, str):
        # It's a simple solver with no time; run with default time
        run(parsed_expr)
    elif parsed_expr[0] == "IF":
        # Handle IF condition
        condition = parsed_expr[1]
        true_branch = parsed_expr[2]
        false_branch = parsed_expr[3]

        if evaluate_condition(condition, variables):
            evaluate_meta_strat(true_branch, variables)
        else:
            evaluate_meta_strat(false_branch, variables)
    elif parsed_expr[0] == "TRY":
            # Handle TRY sequence
        first_expr = parsed_expr[1]  # This can be a solver or another expression
        try_time = parsed_expr[2]
        next_strat = parsed_expr[3]

            # Check if first_expr is a solver
        if isinstance(first_expr, str):
            run(first_expr, try_time)  # Run the solver
        else:
                # If it's not a string, it's a nested expression
            evaluate_meta_strat(first_expr, variables)  # Evaluate the first expression

            # Continue with the next strategy after TRY
        evaluate_meta_strat(next_strat, variables)
    else:
        # Handle a direct Solver case with time
        solver = parsed_expr[0]
        time_value = parsed_expr[1]
        if len(parsed_expr) > 1:
            run(solver, time_value)
        else:
            run(solver)


# Test the parser
def parse_expression(expression, variables={}):
    try:
        parsed = MetaStrat.parseString(expression, parseAll=True)
        return parsed.asList()
    except Exception as e:
        return f"Parse error: {e}"


# Sample expressions
expressions = [
    "(TRY Z3STRAT 5 STP)",
    "(IF (x == y) (TRY Z3STRAT 5 STP) CVC5)",
    "(IF (a > b and c != d) (TRY (IF (a == b) STP Z3STRAT) 10 CVC5) Bitwuzla)"
]

# Parse and execute results
variables = {"x": 1, "y": 1, "a": 3, "b": 2, "c": 4, "d": 5}  # Sample values for evaluation
for expr in expressions:
    print(f"Evaluating Expression: {expr}")
    parsed_expr = parse_expression(expr)
    if isinstance(parsed_expr, list):
        evaluate_meta_strat(parsed_expr[0], variables)
    print()
