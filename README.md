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

A smoke-test configuration for QF\_NIA is provided at `data/smoke/configs/synthesis.json`, with benchmarks in `data/smoke/benchmarks/QF_NIA/`. Run it with:

```bash
python -m z3alpha.synthesize data/smoke/configs/synthesis.json
```

This runs Z3alpha on 10 QF_NIA benchmarks with 10 stage-1 MCTS simulations, 50 stage-2 simulations, and a 2-second per-benchmark timeout.

**Machine-local settings** (`env_config.json` at the repo root):
- `workers` — parallel benchmark evaluations per simulation (default: 4)
- `z3_path` — path to the Z3 binary (default: `z3` on `PATH`)
- `z3_version` — if set, the binary version is verified at startup
- `machine_name` — informational label for result logs

All fields are optional; missing fields use defaults.

**Output files** (in `experiments/synthesis/out-<timestamp>/`):
- `linear_strategy_summary.csv` — per-strategy: simulation id, strategy string, `n_solved`, `par2_avg`, `par10_avg`
- `linear_strategy_per_benchmark.csv` — per-instance outcomes (`sat`/`unsat`/`timeout`/`error`, solve time)
- `linear_selected_strategies.csv` — shortlist of strategies passed to stage 2
- `selector.pkl` — trained PWC selector (default mode); or `synthesized_strategy.txt` (with `--stage2`)

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

We provide the script `scripts/eval_solvers.py` for evaluating solvers and strategies. Sample configuration files for evaluating our synthesized strategies are provided in `data/ijcai24/configs/eval`. For example, to evaluate all solvers on the *leipzig* test, run:

```bash
python scripts/eval_solvers.py data/ijcai24/configs/eval/leipzig.json
```

The evaluation outcomes are saved in the directory specified by the `res_dir` entry in the configuration JSON file.


### Results

All experimental result data are included in `data/ijcai24/`. For each experiment, there is a subfolder (e.g., `data/ijcai24/results/QF_BV/core/`) containing all competing solvers' evaluation statistics and sample strategies synthesized by Z3alpha and FastSMT.



## SMT-COMP'24
As a derived solver, Z3alpha participated in [SMT-COMP 2024 (Single Query Track)](https://smt-comp.github.io/2024/results/results-single-query/) and won [some awards](https://drive.google.com/file/d/1dEeJFfzjJz4vp-mU5XiGnR-hHJdsU1QZ/view?usp=sharing).

## License
Z3alpha is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).


### Citing this Work

**IJCAI'24 (conference):**
```bibtex
@inproceedings{z3alpha_ijcai2024,
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
```

**Acta Informatica (journal, 2025):**
```bibtex
@article{z3alpha_acta_informatica_2025,
  title   = {Novel tree-search method for synthesizing SMT strategies},
  author  = {Lu, Zhengyang John and Day, Joel and Jha, Piyush and Sarnighausen-Cahn, Paul and Siemer, Stefan and Manea, Florin and Ganesh, Vijay},
  journal = {Acta Informatica},
  volume  = {62},
  number  = {3},
  pages   = {28},
  year    = {2025},
  doi     = {10.1007/s00236-025-00495-x},
  url     = {https://doi.org/10.1007/s00236-025-00495-x},
}
```

