#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=1G
#SBATCH --account=def-vganesh
#SBATCH --output=JHC/experiments/solver_test_%j.out
#SBATCH --error=JHC/experiments/solver_test_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

# Add parent directory to PYTHONPATH
export PYTHONPATH="/lustre06/project/6001884/jchen688/github/z3alpha:$PYTHONPATH"

# Create monitoring directories
MONITOR_DIR="JHC/experiments/monitoring_$SLURM_JOB_ID"
mkdir -p $MONITOR_DIR

# Run the script with your benchmarks
python z3alpha/new_evaluator.py \
  --solver z3 \
  --batch-size 5 \
  --benchmark-dir data/ijcai24/benchmarks/samples \
  --timeout 10 \
  --output JHC/experiments/results_$SLURM_JOB_ID.csv \
  --debug \
  --strategy "(then simplify smt)" \
  --monitor-output $MONITOR_DIR

echo "End time: $(date)"