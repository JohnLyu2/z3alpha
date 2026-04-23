# Synthesis Configuration Help

**"logic"**: the SMT logic of the tested benchmarks, e.g., QF_BV, QF_LIA, etc

**"batch_size"**: number of instances to be evaluated in a batch

**"train_dir"**: path to a single root directory for training benchmarks. All `*.smt2` files under that directory (recursively) are used for both linear search and branched (stage-2) synthesis.

**"s1config"**: configurations for the linear strategy MCTS

**"s2config"**: configurations for the branched (stage-2) MCTS

**"sim_num"**: number of MCTS simulations

**"timeout"**: wallclock timeout for each solver-instance pair

**"c_uct"**: the c constant in UCT of MCTS

**"c_ucb"**: the c constant in UCB1 of the layered search

**"ln_strat_num"**: number of linear strategies to be selected after the Stage-1 MCTS

**"random_seed"**: the random seed for rollout
