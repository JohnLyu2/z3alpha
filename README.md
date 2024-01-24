Z3alpha is a tool to synthesize tailored Z3 stragegy to your problem set.
Our tool is built on top of Z3 SMT solver (https://github.com/Z3Prover/z3). Currently we support Z3 12.2.2.

# Prerequisites

The only environmental prerequisites are Z3 and its Python binding (Z3Py). After installing Z3 from the [Z3 GitHub repository](https://github.com/Z3Prover/z3), verify the setup:

### Z3 Executable

Open a terminal and run:
  ```bash
  z3 -h
  ```

If you see the help message, Z3 is installed correctly. If not, add Z3 to your system's PATH.

### Z3Py
In a Python environment, check the Z3 Python binding (Z3Py):
```python
try:
    import z3
except ImportError:
    raise Exception("Z3 Python binding not found.")
```
If no errors occur, Z3Py is ready for use.

# IJCAI-24 Submission Experiment Data
We have included all experimental result data in `ijcai24_data/`. For each experiment, there is a subfolder (e.g., `ijcai24_data/QF_BV/core/`) containing all competing solvers' testing statistics and sample Z3alpha and FastSMT synthesized strategies. The experiments were conducted on a high-performance CentOS 7 cluster equipped with Intel E5-2683 v4 (Broadwell) processors running at 2.10 GHz, accompanied by 75 gigabytes of memory.

# Reproduce IJCAI-24 Experiments

## Benchmark Download
Navigate to the `benchmarks/` directory and run the corresponding benchmark downloading script for each experiment. For example, to download the original FastSMT benchmarks for the experiments in Section 5.3, run:

```bash
$ ./download_fastsmt_benchmarks.sh
```

## FastSMT Strategy Synthesis
For installation and operation of FastSMT, please refer to the guidance available in the [FastSMT GitHub repository](https://fastsmt.ethz.ch/). We have made minor modifications to the FastSMT source code to facilitate compatibility with Z3 12.2.2. We have also updated the tactic and parameter canadidates for each tested SMT logic. All these changes are provided in [link]. 

## Z3str4
To reproduce the QF_S experiment, please use the Z3 version at https://anonymous.4open.science/r/z3str-60AF/README.md. Z3str4 is invoked from the command line with:

```bash
$ z3 smt.string_solver=z3str3 smt.str.tactic=3probe <smt2file>
```