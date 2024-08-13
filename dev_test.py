from alphasmt.evaluator import SolverRunner

solver0 = SolverRunner("z3", "z3", "smtcomp24/test_instances/qfnia0.smt2", 10, 0)
solver0.start()
solver0.join(10)
id, resTask, timeTask, pathTask = solver0.collect()