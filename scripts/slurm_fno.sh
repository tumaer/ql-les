#!/bin/bash
#SBATCH --job-name=fno
#SBATCH --output=slogs/fno_%A_%a.out
#SBATCH --error=slogs/fno_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Warning: Running script outside of SLURM."
else
  # If using Slurm: create slogs directory if it doesn't exist already
  echo "Running on SLURM job ID: $SLURM_JOB_ID"
  mkdir -p slogs
fi

# Ensure that 3 arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: sbatch scripts/slurm_fno.sh <EVERY_N> <SEED> <EXTRAS>"
  exit 1
fi

# DOF:
# model.optimizer.lr=0.0001
# model.net.nbrs_condition="knn"/"radius"
# model.net.nbrs_k=3 / +model.net.nbrs_cutoff=0.2
# +model.net.nbrs_kernel="1/x^2"/"quintic"
# model.scheduler.step_size=100_000
# model.net.noise_std=0.0
# model.net.return_x_grid=True
# model.net.fno_n_modes=[32,32]
# model.net.fno_hidden_channels=32
# model.net.fno_n_layers=4

# Read command-line arguments
EVERY_N="$1"  # can be specified as "1" or "10"
SEED="$2"  # for reproducibility
EXTRAS="$3"  # for additional arguments

echo "Training InterpFNO on every_n=${EVERY_N} with seed=${SEED} and extras='${EXTRAS}'"
python src/train.py experiment=gino_kolm2d_every"${EVERY_N}".yaml model.net.model_name=interp_fno \
  seed="${SEED}" +logger.wandb.name=fno "${EXTRAS}"

### Runs

# Code validation:
# ./scripts/slurm_fno.sh 1 0 "model.optimizer.lr=0.0001 model.net.nbrs_condition=radius model.net.nbrs_k=null +model.net.nbrs_cutoff=0.2 +model.net.nbrs_kernel=quintic model.scheduler.step_size=100_000 model.net.noise_std=0.0 model.net.return_x_grid=True model.net.fno_n_modes=[32,32] model.net.fno_hidden_channels=32 model.net.fno_n_layers=4 logger.wandb.offline=True"

#################### hyperparameter tuning every1 ####################
# sbatch scripts/slurm_fno.sh 1 12345 "model.net.return_x_grid=True model.optimizer.lr=0.0003"
# sbatch scripts/slurm_fno.sh 1 12345 "model.net.return_x_grid=True"  # DEFAULTS
# sbatch scripts/slurm_fno.sh 1 12345 "model.net.return_x_grid=True model.optimizer.lr=0.00003"

# sbatch scripts/slurm_fno.sh 1 12345 "model.net.return_x_grid=True model.net.nbrs_k=10"
# sbatch scripts/slurm_fno.sh 1 12345 "model.net.return_x_grid=True model.net.nbrs_k=30"
# sbatch scripts/slurm_fno.sh 1 12345 "model.net.return_x_grid=True model.net.nbrs_condition=radius +model.net.nbrs_cutoff=0.2 model.net.nbrs_k=null"
# sbatch scripts/slurm_fno.sh 1 12345 "model.net.return_x_grid=True model.net.nbrs_condition=radius +model.net.nbrs_cutoff=0.2 model.net.nbrs_k=null +model.net.nbrs_kernel=quintic"

# sbatch scripts/slurm_fno.sh 1 12345 "model.net.return_x_grid=True model.net.noise_std=0.001"
# sbatch scripts/slurm_fno.sh 1 12345 "model.net.return_x_grid=True model.net.noise_std=0.0001"


# Every1 summary:
# best noise: 0.00003
# lr: 200k-0.001 = 200k-0.0003 = 500k-0.0001; 200k-0.0001 sucks
# new defaults: noise_std=0.00003, transition_steps=250000, init_value=0.001
