#!/bin/bash
#SBATCH --job-name=lag
#SBATCH --output=logs/slogs/lag_%A_%a.out
#SBATCH --error=logs/slogs/lag_%A_%a.out
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
  echo "Usage: sbatch scripts/slurm_lag.sh <EVERY_N> <SEED> <EXTRAS>"
  exit 1
fi

# DOF:

# Read command-line arguments
EVERY_N="$1"  # can be specified as "1" or "10"
SEED="$2"  # for reproducibility
EXTRAS="$3"  # for additional arguments

echo "Training v2u-GNSF on every_n=${EVERY_N} with seed=${SEED} and extras='${EXTRAS}'"
# shellcheck disable=SC2086
python src/train.py experiment=lag_kolm2d_every"${EVERY_N}".yaml \
  seed="${SEED}" +logger.wandb.name=lag ${EXTRAS}
### Runs

#################### hyperparameter tuning every1 ####################
### Runs 06.05.25
# sbatch scripts/slurm_lag.sh 1 12345 "model.optimizer.lr=0.0003"
# sbatch scripts/slurm_lag.sh 1 12345 ""  # DEFAULTS
# sbatch scripts/slurm_lag.sh 1 12345 "model.optimizer.lr=0.00003"
# sbatch scripts/slurm_lag.sh 1 12345 "model.net.noise_std=0.00003"
# sbatch scripts/slurm_lag.sh 1 12345 "model.net.noise_std=0.000003"
# sbatch scripts/slurm_lag.sh 1 12345 "model.optimizer.lr=0.0001 model.net.noise_std=0.00003 model.v2u_solver=none" # -> has to be same as two runs above
# sbatch scripts/slurm_lag.sh 1 12345 "model.net.noise_std=0.0"

# Summary:
# lr=1e-4 is a good choice, but we start with 1e-3 and reduce every 100k steps
# noise is not needed and even degrades performance -> understandable with num_epochs=2.5

### Runs 06.05.25 -> add more v2u_solvers
