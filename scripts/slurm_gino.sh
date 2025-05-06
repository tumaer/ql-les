#!/bin/bash
#SBATCH --job-name=gino
#SBATCH --output=logs/slogs/gino_%A_%a.out
#SBATCH --error=logs/slogs/gino_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Warning: Running script outside of SLURM."
else
  # If using Slurm: create slogs directory if it doesn't exist already
  echo "Running on SLURM job ID: $SLURM_JOB_ID"
  mkdir -p logs/slogs
fi

# Ensure that 3 arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: sbatch scripts/slurm_gino.sh <EVERY_N> <SEED> <EXTRAS>"
  exit 1
fi

# DOF:
# TODO:

# Read command-line arguments
EVERY_N="$1"  # can be specified as "1" or "10"
SEED="$2"  # for reproducibility
EXTRAS="$3"  # for additional arguments

echo "Training GINO on every_n=${EVERY_N} with seed=${SEED} and extras='${EXTRAS}'"
# shellcheck disable=SC2086
python src/train.py experiment=gino_kolm2d_every"${EVERY_N}".yaml model.net.model_name=gino \
  seed="${SEED}" +logger.wandb.name=gino ${EXTRAS}

### Runs

#################### hyperparameter tuning every1 ####################
# sbatch scripts/slurm_gino.sh 1 12345 ""
