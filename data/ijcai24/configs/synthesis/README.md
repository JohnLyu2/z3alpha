# Synthesis Configuration Help

**"logic"**: the SMT logic of the tested benchmarks, e.g., QF_BV, QF_LIA, etc

**"batch_size"**: number of instances to be evaluated in a batch

**"train_dir"**: path to a single root directory for training benchmarks. All `*.smt2` files under that directory (recursively) are used for both linear search and branched (stage-2) synthesis.

**"timeout"**: wall-clock timeout (seconds) for each solver–instance pair in both linear search and stage-2 search

**"mcts_sims"**: number of MCTS simulations for linear strategy search

**"s2_sims"**: number of MCTS simulations for branched (stage-2) strategy search

**"ln_strat_num"**: number of linear strategies to select for the shortlist after linear MCTS

## Defaults and overrides

Package defaults (UCT/UCB constants, random seed) load from `z3alpha/synthesis_defaults.json` in the repository and are merged with this JSON (your file overrides on a per-key basis if you add optional keys there).

- Optional keys in the JSON, same shape as the defaults: **`mcts_config`** (with `c_uct` and `c_ucb`) and **`random_seed`**.

- CLI overrides (highest priority): `python -m z3alpha.synthesize CONFIG.json --c-uct 0.6 --c-ucb 0.1 --random-seed 42`
