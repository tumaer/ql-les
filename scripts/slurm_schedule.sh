#!/bin/bash
#SBATCH --job-name=simple_kolm_u0
#SBATCH --output=slogs/vel_solvers_%A_%a.out
#SBATCH --error=slogs/vel_solvers_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p slogs

# Launch with: sbatch scripts/slurm_schedule.sh
bash scripts/schedule.sh
