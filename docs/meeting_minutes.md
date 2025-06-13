## May 26, 2025
### General Info
* New SMT-LIB benchmarks: non-incremental: https://zenodo.org/records/15493090
* Let's all use the latest stable version, Z3-4.15.0: https://github.com/Z3Prover/z3/releases/tag/z3-4.15.0
* Thursday 8:00am EDT

### To-Dos
#### John
* Complete parallel strategy synthesis in 2-3 days
* Write docs for how to do the synthesis
* Review the new evaluator

#### Jiahao
* Change the arguments/logics: specify per-process core & memory requirement, instead of batch size
* Check whether the new evaluation works correctly with parallel strategies
* Node-level parallelization (less priority)

#### Paul
* Look into tactic/parameter pools for some interesting logics


## Jun 5, 2025
### Agenda
* (John) I was invited by Joel to make a presentation about Z3alpha at [MOSCA](https://mosca2025.github.io/).
  I want make this talk more focusing on strings, especially what tactics and parameters work better on
  what families, etc, and maybe incorporate a bit about our recent progress on parallel and ML theory.
  We can discuss what are be done in this ~2 months and about potential future publication.
* Go through the guideline
* Let's each one try one logic to get familiar with the process
* Paul's progress on logic-specific tactic/parameter (which code is used to run the evaluation and what is the time limit)
* Let's look at all logics and decide how to proceed
* New evaluator: Is the new logic/arguments implemented? Later merge with the exisiting evaluator.
  * Memory requiement for each execution: take the minmum of core and memory requirement; can be optional, keep the same logic as now 
* [the posted [issue](https://github.com/Z3Prover/z3/issues/7674)] Jiahao found `par-or` may return unknown/error if one stratey in the portfolio returns so; and the setting for parallel may need to be explicitly turned on; will follow up.

## Jun 12, 2025

* John: First solver submission
* Paul: Will work on some logics, and look into the z3str3 tactic in z3 code
* Jiahao: Make a new binary to execute Z3 strategies

