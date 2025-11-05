**Z3alpha** synthesizes efficient Z3 strategies tailored to your problem set! See our IJCAI'24 paper [Layered and Staged Monte Carlo Tree Search for SMT Strategy Synthesis](https://arxiv.org/abs/2401.17159) for technical details. 

Our tool is built on top of the [Z3 SMT solver](https://github.com/Z3Prover/z3).

## Setup

We recommend using a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
```

Install Z3alpha and its dependencies:

1. **Z3 Solver**

Z3alpha depends both on the Z3 binary and the Z3 Python bindings. The easiest way to install both is to run:

```bash
pip install z3-solver
```

2. **Install Z3alpha**

```bash
pip3 install -e .
```

## A Synthesis Example

Here, we provide an example of synthesizing a tailored Z3 strategy for a toy benchmark set `data/sample/benchmarks/`. 


The command for this toy example is as follows:

```bash
python -m z3alpha.synthesize data/sample/configs/synthesis.json
```

The configuration file specifies settings such as the number of MCTS simulations, training datasets, and timeouts. A sample configuration is provided at `data/sample/configs/synthesis.json`.

After this command finishes, the synthesized strategy is saved under `experiments/synthesis/` in a directory named `out-<starting time:%Y-%m-%d_%H-%M-%S>`.

## IJCAI-24 Reproduction

### Benchmarks

The scripts in `data/ijcai24/benchmarks/download_scripts/` will download all benchmark sets used in the paper into the `data/ijcai24/benchmarks/` directory. For example, to download the original FastSMT benchmarks for the experiments in Section 5.3, run:

```bash
./data/ijcai24/benchmarks/download_scripts/download_fastsmt_benchmarks.sh
```

### Z3alpha Strategy Synthesis

We provide sample configuration JSON files for the experiments in `data/ijcai24/configs/synthesis/`. When under the repository root, run the following command with the corresponding configuration file. 
For example, to synthesize a strategy for *leipzig*, run:

```bash
python -m z3alpha.synthesize data/ijcai24/configs/synthesis/leipzig.json
```

Before running the script, ensure to adjust the JSON configuration file to match your computer's specifications.

### Evaluation

We provide the script `scripts/exp_tester.py` for evaluating solvers and strategies. Sample configuration files for evaluating our synthesized strategies are provided in `data/ijcai24/configs/eval`. For example, to evaluate all solvers on the *leipzig* test, run:

```bash
python scripts/exp_tester.py data/ijcai24/configs/eval/leipzig.json
```

The evaluation outcomes are saved in the directory specified by the `res_dir` entry in the configuration JSON file.


### Results

All experimental result data are included in `data/ijcai24/`. For each experiment, there is a subfolder (e.g., `data/ijcai24/results/QF_BV/core/`) containing all competing solvers' evaluation statistics and sample strategies synthesized by Z3alpha and FastSMT.



## SMT-COMP'24
As a derived solver, Z3alpha entered [SMT-COMP 2024 (Single Query Track)](https://smt-comp.github.io/2024/results/results-single-query/)  and won [some awards](https://drive.google.com/file/d/1dEeJFfzjJz4vp-mU5XiGnR-hHJdsU1QZ/view?usp=sharing). See our submitted solver along with synthesized strategies in `smtcomp24/`. Note that the executables in `smtcomp24/z3bin/` are compiled for the [competition evnvironment](https://smt-comp.github.io/2024/specs/).

## License
Z3alpha is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).


### Citing this Work
```bibtex
@inproceedings{ijcai2024p211,
  title     = {Layered and Staged Monte Carlo Tree Search for SMT Strategy Synthesis},
  author    = {Lu, Zhengyang and Siemer, Stefan and Jha, Piyush and Day, Joel and Manea, Florin and Ganesh, Vijay},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {1907--1915},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/211},
  url       = {https://doi.org/10.24963/ijcai.2024/211},
}

