# AlgoSelect

AlgoSelect is a research project that combines machine learning-based algorithm selection with automated strategy synthesis for SMT (Satisfiability Modulo Theories) solvers. It integrates two powerful tools:

- **Z3alpha**: Synthesizes efficient Z3 solver strategies using layered and staged Monte Carlo Tree Search (MCTS)
- **MachSMT**: Uses machine learning to select the best solver/strategy for individual benchmarks

## Overview

AlgoSelect implements a two-stage workflow:

1. **Stage 1 (Z3alpha)**: Synthesizes a portfolio of optimized Z3 strategies for a given set of benchmarks using MCTS
2. **Stage 2 (MachSMT)**: Trains a machine learning model to predict which synthesized strategy is best for each individual benchmark

This hybrid approach leverages the strengths of both automated strategy synthesis and empirical algorithm selection.

## Main Scripts

AlgoSelect provides three main scripts, each designed for different use cases:

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `bin/run_MachZ3alpha.py` | **Full two-stage workflow** | When you want to synthesize new strategies (Stage 1) and then train a selector (Stage 2) from scratch |
| `bin/run_MachSMT.py` | **Stage 2 with Z3 execution** | When you already have synthesized strategies and want to evaluate them by actually running Z3 on benchmarks |
| `bin/run_MachSMT_solvers.py` | **Stage 2 with precomputed results** | When you have precomputed CSV results and want to quickly evaluate MachSMT's selection performance without running Z3 |

### Script Details

#### 1. `run_MachZ3alpha.py` - Full Workflow

Orchestrates the complete two-stage pipeline:
- Runs Z3alpha to synthesize strategies on training benchmarks
- Automatically configures and runs MachSMT using the synthesized strategies
- Optionally skip Stage 2 with `--skip-stage2` flag

**Use this when:** You're starting a new experiment from scratch and need to synthesize strategies.

#### 2. `run_MachSMT.py` - Stage 2 with Actual Execution

Trains MachSMT and evaluates performance with real Z3 execution:
- Loads pre-synthesized strategies from Stage 1 output
- Trains MachSMT model to predict which strategy to use
- Actually runs Z3 with selected strategies on test benchmarks
- Compares MachSMT selections against a default strategy
- Supports result caching for efficiency
- Evaluates benchmarks in parallel batches

**Use this when:** You have synthesized strategies and want to measure actual solving performance.

#### 3. `run_MachSMT_solvers.py` - Stage 2 with Precomputed Results

Lightweight evaluation using precomputed solver results:
- Loads pre-synthesized strategies and precomputed results from CSV
- Trains MachSMT model to predict solver selection
- Retrieves performance from CSV instead of running Z3
- Supports additional features: description embeddings, family information
- Faster evaluation for testing different ML configurations

**Use this when:** You want to quickly experiment with different MachSMT settings without waiting for Z3 execution.

## Installation

### Prerequisites

- Python 3.9 or higher
- Z3 SMT Solver

### Setup

1. Clone the repository and navigate to the AlgoSelect directory

2. Install MachSMT dependencies:

```bash
cd MachSMT
make dev  # In a virtual environment
# Or if make dev doesn't work:
pip install -e .
cd ..
```

3. Install Z3alpha dependencies:

```bash
pip install z3-solver
```

4. Verify MachSMT installation points to source code (not site-packages):

```bash
python -c "import machsmt; print(machsmt.__file__)"
```

This ensures code changes take effect without reinstalling.

5. Ensure Z3 is available in your PATH or configure the path in your config file

## Usage

### 1. Running the Full Workflow

```bash
python bin/run_MachZ3alpha.py --config config/MachZ3alpha/examples/sample_config.json
```

This will:
- Run Stage 1 to synthesize strategies
- Automatically run Stage 2 to train and evaluate MachSMT
- Output results to `results/synthesis/out-TIMESTAMP/`

To run only Stage 1:
```bash
python bin/run_MachZ3alpha.py --config config/MachZ3alpha/examples/sample_config.json --skip-stage2
```

### 2. Running Stage 2 with Actual Z3 Execution

```bash
python bin/run_MachSMT.py --config config/MachSMT/examples/sample_config.json
```

This requires:
- Training data CSV (`ln_res.csv` from Stage 1)
- Strategy mapping CSV (`strat_mapping.csv` from Stage 1)
- Test benchmarks directory
- Z3 solver path

### 3. Running Stage 2 with Precomputed Results

```bash
# Basic usage
python bin/run_MachSMT_solvers.py --config config/MachSMT/examples/sample_config.json

# With description embeddings
python bin/run_MachSMT_solvers.py --config config/MachSMT/examples/sample_config.json --use_description

# With family features
python bin/run_MachSMT_solvers.py --config config/MachSMT/examples/sample_config.json --use_family
```

This requires:
- Training data CSV
- Precomputed test results CSV
- Strategy mapping CSV
- Test benchmarks directory (for feature extraction)

### Quick Start with Sample Script

```bash
./scripts/slurm/sample/run_MachZ3alpha_sample.sh
```

Results will be saved in the `results/` folder.

## Configuration

Configuration files are in JSON format and organized by tool:

### Z3alpha Configuration (`config/MachZ3alpha/`)

```json
{
  "logic": "QF_NIA",
  "batch_size": 2,
  "s1config": {
    "num_iters": 100,
    "timeout": 10
  },
  "s2config": {
    "num_iters": 50,
    "timeout": 10
  },
  "ln_strat_num": 5,
  "machsmt_config": "config/MachSMT/examples/sample_config.json"
}
```

Key parameters:
- `logic`: SMT logic to target
- `s1config`: Stage 1 MCTS parameters
- `s2config`: Stage 2 MCTS parameters
- `ln_strat_num`: Number of strategies to synthesize
- `machsmt_config`: Path to MachSMT config for Stage 2

### MachSMT Configuration (`config/MachSMT/`)

```json
{
  "paths": {
    "training_file": "data/ln_res.csv",
    "mapping_file": "data/strat_mapping.csv",
    "benchmark_dir": "benchmarks/test/",
    "test_res_file": "data/test_results.csv",
    "output_dir": "results/",
    "z3": "z3",
    "tmp_dir": "/tmp/",
    "default_strat": "data/default_strategy.txt"
  },
  "settings": {
    "timeout": 10,
    "num_benchmarks": -1,
    "batch_size": 4,
    "selector": "EHMLogic",
    "ml_core": "scikit",
    "use_gpu": false,
    "description": false,
    "run_z3alpha_strat": false
  }
}
```

Key parameters:
- `timeout`: Solver timeout in seconds
- `num_benchmarks`: Number of benchmarks to evaluate (-1 for all)
- `batch_size`: Parallel evaluation batch size
- `selector`: Selection method (`EHM`, `EHMLogic`, `PWC`, `PWCLogic`)
- `ml_core`: ML backend (`scikit`, `xgboost`, `pytorch`)

### Selector Methods

- **EHM (Empirical Hardness Model)**: Regression-based runtime prediction
- **EHMLogic**: EHM with logic-aware filtering
- **PWC (Pairwise Comparison)**: Classification-based pairwise solver comparison
- **PWCLogic**: PWC with logic-aware filtering

## Directory Structure

```
AlgoSelect/
├── README.md                # Project documentation
│
├── bin/                     # Main executable scripts
│   ├── run_MachZ3alpha.py      # Full two-stage workflow
│   ├── run_MachSMT.py          # Stage 2 with Z3 execution
│   └── run_MachSMT_solvers.py  # Stage 2 with precomputed results
│
├── benchmarks/              # SMT benchmark files
│   ├── SMTCOMP24/          # Organized by logic (NIA, QF_BV, etc.)
│   └── tmp_bench/          # Temporary benchmarks
│
├── config/                  # Configuration files
│   ├── MachSMT/
│   │   ├── examples/       # Template/example configs
│   │   ├── SMTCOMP24/      # Competition-specific configs
│   │   └── cached/         # Cached/generated configs
│   └── MachZ3alpha/
│       └── examples/       # Template/example configs
│
├── data/                    # Training data and synthesized strategies
│   └── 5-strats/           # 5-strategy experiment data
│
├── results/                 # All experimental outputs
│   ├── SMTCOMP24/          # Competition results
│   └── prev_experiments/   # Archived experiments
│
├── scripts/                 # Utility scripts
│   ├── dataset_create.py           # Dataset creation utilities
│   ├── dataset_create_with_csv.py
│   ├── embeddings/                 # Description embedding scripts
│   └── slurm/                      # HPC/SLURM job scripts
│
├── docs/                    # Documentation
│   └── notes/              # Development notes
│
└── MachSMT/                # MachSMT submodule/dependency
```

## Output

All scripts create timestamped output directories containing:
- `output.txt`: Detailed results for each benchmark
- Configuration file copy
- Performance metrics (PAR2 scores, solving rates)
- Model training time

Example output location:
```
results/run_20250119_143022/output.txt
```

## Performance Metrics

- **PAR2 Score**: Penalized Average Runtime (2x timeout for unsolved instances)
- **Solving Rate**: Percentage of benchmarks solved within timeout
- **Model Training Time**: Time to train the MachSMT model

## Research Background

This project is based on the following research:

- **Z3alpha**: Hu et al., "Layered and Staged Monte Carlo Tree Search for SMT Strategy Synthesis", IJCAI 2024
- **MachSMT**: Scott et al., "Algorithm selection for SMT", STTT 2023

AlgoSelect successfully competed in SMT-COMP 2024 (Single Query Track).

## Troubleshooting

### MachSMT Installation Issues

If `make dev` doesn't work, try:
```bash
pip install -e .
```

Make sure the MachSMT module points to the source code directory, not site-packages. This ensures changes to the code take effect without reinstalling.

Verify with:
```bash
python -c "import machsmt; print(machsmt.__file__)"
```

### Common Issues

- **Configuration file not found**: Check that paths in config files are absolute or relative to the script location
- **No benchmarks found**: Verify `benchmark_dir` path and that `.smt2` files exist
- **Z3 not found**: Ensure Z3 is in PATH or set the correct path in config `paths.z3`
- **Missing CSV files**: For `run_MachSMT.py` and `run_MachSMT_solvers.py`, ensure Stage 1 has been run first to generate required CSV files

## License

See individual submodules (Z3alpha, MachSMT) for their respective licenses.
