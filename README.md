**Z3alpha** synthesizes efficient Z3 strategies tailored to your problem set! See our IJCAI'24 paper [Layered and Staged Monte Carlo Tree Search for SMT Strategy Synthesis](https://arxiv.org/abs/2401.17159) for technical details. 

Our tool is built on top of the [Z3 SMT solver](https://github.com/Z3Prover/z3). 

## SMT-COMP'24
As a derived solver, Z3alpha entered SMT-COMP 2024 (Single Query Track)  and won [some awards](https://smt-comp.github.io/2024/results/results-single-query/). See our submitted solver along with synthesized strategies in `smtcomp24/`. Note that the executables in `smtcomp24/z3bin/` are compiled for the [competition evnvironment](https://smt-comp.github.io/2024/specs/).


## Prerequisites

The only environmental prerequisites are Z3 and its Python binding (Z3Py). After installing Z3 from the [Z3 GitHub repository](https://github.com/Z3Prover/z3), verify the setup:

**Z3 Executable**

Open a terminal and run:
  ```bash
  z3 -h
  ```

If you see the help message, Z3 is installed correctly. If not, add Z3 to your system's PATH.

**Z3Py**

In a Python environment, check the Z3 Python binding (Z3Py):
```python
try:
    import z3
except ImportError:
    raise Exception("Z3 Python binding not found.")
```
If no errors occur, Z3Py is ready for use.

## A Synthesis Example

Here, we provide an example of synthesizing a tailored Z3 strategy for a toy benchmark set `benchmarks/small`. The `synthesize_full.py` script performs the staged MCTS, which takes a configuration JSON file as an argument. The configuration file specifies settings such as the MCTS simulation number, training datasets, timeouts, etc. The configuration file for this toy example is provided at `experiments/syn_configs/sample.json`. 

The command for this toy example is as follows:

```bash
$ python ./synthesize_full.py experiments/syn_configs/sample.json
```

After this command terminates, the synthesized strategy is saved to a result directory under `experiments/results/`, along with the logs (if enabled). The result directory is named as `out-<starting time:%Y-%m-%d_%H-%M-%S>`.

## IJCAI-24 Experiment Data
We have included all experimental result data in `ijcai24_data/`. For each experiment, there is a subfolder (e.g., `ijcai24_data/QF_BV/core/`) containing all competing solvers' testing statistics and sample Z3alpha and FastSMT synthesized strategies. The experiments were conducted on a high-performance CentOS 7 cluster equipped with Intel E5-2683 v4 (Broadwell) processors running at 2.10 GHz.

## Reproduce IJCAI-24 Experiments

The experiments in our IJCAI'24 paper are based on Z3 4.12.2.

### Benchmark Download
Navigate to the `benchmarks/` directory and run the corresponding benchmark downloading script for each experiment. For example, to download the original FastSMT benchmarks for the experiments in Section 5.3, run:

```bash
$ ./download_fastsmt_benchmarks.sh
```
Note: the previous script for downloading SMT-LIB benchmarks from the official GitLab is not working now due to the SMT-LIB organizers changing the storage location of the SMT-LIB benchmarks from GitLab to Zenodo. We will update the downloading script soon.

### Z3alpha Strategy Synthesis

We provide sample configuration JSON files for the experiments in `experiments/syn_configs/`. When under the repository root, run the `synthesize_full.py` script with the corresponding configuration file to start the synthesis. For example, to synthesize a strategy for *leipzig*, run:

```bash
$ python ./synthesize_full.py experiments/syn_configs/leipzig.json
```

Before running the script, ensure to adjust the JSON configuration file to match your computer's specifications.

### Competing Solvers

**FastSMT**

For installation and operation of FastSMT, please refer to the guidance available in the [FastSMT GitHub repository](https://fastsmt.ethz.ch/). In our IJCAI experiemnts, minor modifications to FastSMT are made to facilitate its compatibility with Z3 4.12.2. We have also updated the tactic and parameter canadidates for each tested SMT logic.

**cvc5**

Please check the [cvc5 webpage](https://cvc5.github.io/) for installation and operation instructions. We use CVC5-1.0.5 as one baseline solver in our experiments. 

### Evaluation

We provide the script `scripts/exp_tester.py` for evaluating solvers and strategies. Sample configuration files for evaluating our synthesized strategies are provided in `experiments/eva_configs/`. For example, to evaluate all solvers on the *leipzig* test, run:

```bash
$ python scripts/exp_tester.py experiments/eva_configs/leipzig.json
```

The evaluation outcomes are saved in the directory specified by the `res_dir` entry in the configuration JSON file.

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

