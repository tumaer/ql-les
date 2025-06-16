#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/slogs/%x_%A_%a.out
#SBATCH --error=logs/slogs/%x_%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs/slogs

if [ $# -lt 1 ]; then
    echo "Usage: sbatch -J <job_name> scripts/slurm_train.sh <experiment_configs>"
    exit 1
fi

# shellcheck disable=SC2068
python src/train.py $@
