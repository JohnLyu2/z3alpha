# Evaluation Configuration Help
  
**"solvers"**: list all competing solvers with their invocation command

**"strat_files"**: list Z3 strategy file path to be tested (there must be a "Z3" entry in "solvers"); put *null* for the default Z3 solver

**"timeout"**: wallclock timeout for a solver-instance pair

**"batch_size"**: number of instances to be evaluated in a batch

**"res_dir"**: the directory where the results are saved

**"eval_dir"**: directory to search for `*.smt2` (exactly one of eval_dir or eval_list_file)

**"eval_list_file"**: path to a text file listing instance paths, one per line (blank lines and lines starting with `#` are ignored). Use this for large instance sets.

**"tmp_dir"**: the temp file directory

