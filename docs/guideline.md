# Guidelines for Preparing SMT-COMP'25

## Summary of Important Notes

* Download benchmarks from the SMT-LIB'25 Zenodo repo: https://zenodo.org/records/15493090**
* Make sure using the Z3 latest stable version, **Z3-4.15.0**: https://github.com/Z3Prover/z3/releases/tag/z3-4.15.0, for both synthesis and evaluaiton.
* Syntheize config json:
    * timeout (s1config): 30s
    * sim_num (s1config): 600
    * ln_strat_num: 4


## Training Dataset Preparation

### SMT-LIB Benchmark Downloading

Download benchmarks from the SMT-LIB'25 Zenodo repo: https://zenodo.org/records/15493090.
You may use the downloading script at `scripts/smtlib25_download.sh` 
(Usage:`scripts/smtlib25_download.sh <logic-name> [<save-path>]`)

**(TBD)** We will use a 30-second time limit (for each instance-solver-pair execution)
and run simulations for 600 times when synthesizing strategies. 
Depending on the computation resources (e.g., #cores), we can decide how many instances we want to include in the training set for each logic.

For example, I use a machine with 64 cores (nodes at Narvel, Compute Canada).
If I want to 

## Z3alpha Environment Setup

Let's all 


## Synthesize Strategies