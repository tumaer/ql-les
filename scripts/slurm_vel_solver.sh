#!/bin/bash
#SBATCH --job-name=vel_solvers
#SBATCH --output=slogs/vel_solvers_%A_%a.out
#SBATCH --error=slogs/vel_solvers_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Warning: Running script outside of SLURM."
else
  # If using Slurm: create logs directory if it doesn't exist
  mkdir -p slogs
fi

# Read command-line arguments and set default alpha_u if not provided
CASE_NAME="$1"
ALPHA_U="${2:-1.0}"  # can be specified as "1" or "1.0"
EVERY_N="${3:-1}"  # can be specified as "1" or "10"

# Check if case name is provided
if [ -z "$CASE_NAME" ]; then
  echo "Usage: sbatch scripts/slurm_vel_solver.sh <case_name> [alpha_u]"
  exit 1
fi

echo "Running case '${CASE_NAME}' with alpha_u=${ALPHA_U}"
if (( $(echo "$ALPHA_U == 0.0" | bc -l) )); then
  # If alpha_u == 0, fall back to predicting only acceleration for v
  echo "Predicting only acceleration for v"
  python src/train.py experiment=gns_kolm2d_every${EVERY_N}.yaml model.alpha_u=${ALPHA_U} \
    +logger.wandb.name=koml2d_${CASE_NAME} model.vel_solver=${CASE_NAME} \
    model.net.node_in=26 model.net.node_out=2
else
  # If alpha_u != 0, predict both accelerations for u and v
  echo "Predicting accelerations for u and v"
  python src/train.py experiment=gns_kolm2d_every${EVERY_N}.yaml model.alpha_u=${ALPHA_U} \
    +logger.wandb.name=koml2d_${CASE_NAME} model.vel_solver=${CASE_NAME}
fi

# # Runs to start:
# sbatch scripts/slurm_vel_solver.sh simple 0.0
# sbatch scripts/slurm_vel_solver.sh simple 0.000001
# sbatch scripts/slurm_vel_solver.sh simple 1
# sbatch scripts/slurm_vel_solver.sh tvf 1
# sbatch scripts/slurm_vel_solver.sh simple_u 1
# sbatch scripts/slurm_vel_solver.sh simple_u_closure 1 - really bad
# sbatch scripts/slurm_vel_solver.sh simple_rlx 1
# 
# sbatch scripts/slurm_vel_solver.sh simple 1 10
# sbatch scripts/slurm_vel_solver.sh tvf 1 10
# sbatch scripts/slurm_vel_solver.sh simple_u 1 10
# sbatch scripts/slurm_vel_solver.sh simple_rlx 1 10