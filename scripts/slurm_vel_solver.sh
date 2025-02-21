#!/bin/bash
#SBATCH --job-name=vel_solvers
#SBATCH --output=slogs/vel_solvers_%A_%a.out
#SBATCH --error=slogs/vel_solvers_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Warning: Running script outside of SLURM."
else
  # If using Slurm: create logs directory if it doesn't exist already
  mkdir -p slogs
fi

# Check if 4 arguments are provided
if [ "$#" -ne 4 ]; then
  echo "Usage: sbatch scripts/slurm_vel_solver.sh <MODEL_NAME> <VEL_SOLVER> <ALPHA_U> <EVERY_N>"
  exit 1
fi

# Read command-line arguments
MODEL_NAME="$1"  # "gns" or "segnn"
VEL_SOLVER="$2"  # "simple", "tvf", "simple_u", "simple_u_closure", "simple_rlx"
ALPHA_U="$3"  # can be specified as "1" or "1.0"
EVERY_N="$4"  # can be specified as "1" or "10"

echo "Training ${MODEL_NEME} with vel_solver=${VEL_SOLVER}, alpha_u=${ALPHA_U}, and every_n=${EVERY_N}"
python src/train.py experiment=${MODEL_NAME}_kolm2d_every${EVERY_N}.yaml model.alpha_u=${ALPHA_U} \
  +logger.wandb.name=${MODEL_NAME}_${VEL_SOLVER} model.vel_solver=${VEL_SOLVER}

### Runs

# Code validation:
# sbatch scripts/slurm_vel_solver.sh gns simple 0.0 1
# sbatch scripts/slurm_vel_solver.sh gns simple 0.000001 1

# gns & every1
# sbatch scripts/slurm_vel_solver.sh gns simple 1 1
# sbatch scripts/slurm_vel_solver.sh gns tvf 1 1
# sbatch scripts/slurm_vel_solver.sh gns simple_u 1 1
# sbatch scripts/slurm_vel_solver.sh gns simple_u_closure 1 1  # really bad
# sbatch scripts/slurm_vel_solver.sh gns simple_rlx 1 1
# gns & every10
# sbatch scripts/slurm_vel_solver.sh gns simple 1 10
# sbatch scripts/slurm_vel_solver.sh gns tvf 1 10
# sbatch scripts/slurm_vel_solver.sh gns simple_u 1 10
# sbatch scripts/slurm_vel_solver.sh gns simple_rlx 1 10

# segnn & every1
# sbatch scripts/slurm_vel_solver.sh segnn simple 1 1
# sbatch scripts/slurm_vel_solver.sh segnn tvf 1 1
# sbatch scripts/slurm_vel_solver.sh segnn simple_u 1 1
# sbatch scripts/slurm_vel_solver.sh segnn simple_u_closure 1 1  # really bad
# sbatch scripts/slurm_vel_solver.sh segnn simple_rlx 1 1
# segnn & every10
# sbatch scripts/slurm_vel_solver.sh segnn simple 1 10
# sbatch scripts/slurm_vel_solver.sh segnn tvf 1 10
# sbatch scripts/slurm_vel_solver.sh segnn simple_u 1 10
# sbatch scripts/slurm_vel_solver.sh segnn simple_rlx 1 10