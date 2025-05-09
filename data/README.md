# Data Directory

This directory contains the data used or produced by the project, organized by research paper/competition.

## Directory Structure

### IJCAI 2024 Data (`ijcai24/`)
Contains data related to the IJCAI 2024 paper:
- `benchmarks/`: Scripts to download the SMT benchmarks used in the paper
- `results/`: Sample synthesized strategies and experimental results
- `configs/`: Configuration files for experiments
  - `synthesis/`: Configuration JSON files for strategy synthesis
  - `eval/`: Configuration JSON files for evaluation

### SMT-COMP 2024 Data (`smtcomp24/`)
Contains our submission to SMT-COMP 2024:
- `z3bin/`: Z3 binary files and related executables
- `strats/`: Strategy files and configurations
- `z3alpha.py`: Main script/executable of our submitted solver