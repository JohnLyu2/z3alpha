# Synthesis Configuration Help
  
**"logic"**: the SMT logic of the tested benchmarks, e.g., QF_BV, QF_LIA, etc

**"batch_size"**: number of instances to be evaluated in a batch

**"s1config"**: configurations for the stage-1 MCTS

**"s2config"**: configurations for the stage-2 MCTS

**"bench_dirs"**: list all training set directorys (will recursively include all instances in the listed directories as training instances)

**"sim_num"**: number of MCTS simulations

**"timeout"**: wallclock timeout for each solver-instance pair

**"c_uct"**: the c constant in UCT of MCTS

**"c_ucb"**: the c constant in UCB1 of the layered search

**"is_log"**: whether log the info

**"ln_strat_num"**: number of linear strategies to be selected after the Stage-1 MCTS

**"temp_folder"**: the temp file direcotry

**"random_seed"**: the random seed for rollout
