#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=def-vganesh
#SBATCH --output=scripts/JHC/SBS_VBS_%j.out
#SBATCH --error=scripts/JHC/SBS_VBS_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"  
echo "Start time: $(date)"

# Run the script with z3-noodler solver
python queries/SBS_VBS.py \

echo "End time: $(date)"