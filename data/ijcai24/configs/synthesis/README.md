# Synthesis Configuration Help

**"logic"**: the SMT logic of the tested benchmarks, e.g., QF_BV, QF_LIA, etc

**"batch_size"**: number of instances to be evaluated in a batch

**"train_dir"**: path to a single root directory for training benchmarks. All `*.smt2` files under that directory (recursively) are used for both linear search and branched (stage-2) synthesis.

**"timeout"**: wall-clock timeout (seconds) for each solver–instance pair in both linear search and stage-2 search

**"mcts_sims"**: number of MCTS simulations for linear strategy search

**"branched_sims"**: number of MCTS simulations for branched (conditional) strategy search

**"ln_strat_num"**: number of linear strategies to select for the shortlist after linear MCTS

Optional keys (see `z3alpha.config`): **`z3path`**, **`parent_log_dir`**, **`log_level`**, **`logic_config_dir`**

Unknown keys in the JSON cause an error. Do not add `mcts_config` or `random_seed` to this file.

## MCTS hyperparameters and random seed

Defaults live in `z3alpha/config/synthesis.py` (`DEFAULT_C_UCT`, `DEFAULT_C_UCB`, `DEFAULT_RANDOM_SEED`). Override per run on the CLI:

`python -m z3alpha.synthesize CONFIG.json --c-uct 0.6 --c-ucb 0.1 --random-seed 42`
