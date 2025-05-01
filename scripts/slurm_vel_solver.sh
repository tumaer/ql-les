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

# Ensure that 6 arguments are provided
if [ "$#" -ne 6 ]; then
  echo "Usage: sbatch scripts/slurm_vel_solver.sh <MODEL_NAME> <VEL_SOLVER> <ALPHA_U> <EVERY_N> <SEED> [EXTRAS]"
  exit 1
fi

# Read command-line arguments
MODEL_NAME="$1"  # "gns" or "segnn"
VEL_SOLVER="$2"  # "simple", "tvf", "simple_u", "simple_u_closure", "simple_rlx"
ALPHA_U="$3"  # can be specified as "1" or "1.0"
EVERY_N="$4"  # can be specified as "1" or "10"
SEED="$5"  # for reproducibility
EXTRAS="$6"  # for additional arguments

echo "Training ${MODEL_NAME} with vel_solver=${VEL_SOLVER}, alpha_u=${ALPHA_U}, every_n=${EVERY_N}," \
  "seed=${SEED}, and extras='${EXTRAS}'"
python src/train.py experiment="${MODEL_NAME}"_kolm2d_every"${EVERY_N}".yaml model.alpha_u="${ALPHA_U}" \
  +logger.wandb.name="${MODEL_NAME}"_"${VEL_SOLVER}" model.vel_solver="${VEL_SOLVER}" seed="${SEED}" "${EXTRAS}"

### Runs

# Code validation:
# sbatch scripts/slurm_vel_solver.sh gns simple 0.0 1 12345
# sbatch scripts/slurm_vel_solver.sh gns simple 0.000001 1 12345

# gns & every1
# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345
# sbatch scripts/slurm_vel_solver.sh gns tvf 1 1 12345
# sbatch scripts/slurm_vel_solver.sh gns simple_u 1 1 12345
# sbatch scripts/slurm_vel_solver.sh gns simple_u_closure 1 1 12345  # really bad
# sbatch scripts/slurm_vel_solver.sh gns simple_rlx 1 1 12345
# gns & every10
# sbatch scripts/slurm_vel_solver.sh gns simple 1 10 12345
# sbatch scripts/slurm_vel_solver.sh gns tvf 1 10 12345
# sbatch scripts/slurm_vel_solver.sh gns simple_u 1 10 12345
# sbatch scripts/slurm_vel_solver.sh gns simple_rlx 1 10 12345

#################### hyperparameter tuning every1 ####################
# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.net.noise_std=0.0001"  - better
# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.net.noise_std=0.0006"  - worse
# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.optimizer.weight_decay=0.01"  - no impact
# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.scheduler.transition_steps=200000"  - better

# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.net.noise_std=0.0001 model.scheduler.transition_steps=200000"
# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.net.noise_std=0.0001 model.scheduler.transition_steps=200000 model.scheduler.init_value=0.0003"
# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.net.noise_std=0.00003 model.scheduler.transition_steps=200000"
# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.net.noise_std=0.00000 model.scheduler.transition_steps=200000"
# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.net.noise_std=0.0001 model.scheduler.transition_steps=500000"

# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.net.noise_std=0.00003 model.scheduler.transition_steps=200000 model.scheduler.init_value=0.0003"
# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.net.noise_std=0.00003 model.scheduler.transition_steps=500000"

# sbatch scripts/slurm_vel_solver.sh gns simple 1 1 12345 "model.net.noise_std=0.00003 model.scheduler.transition_steps=200000 model.scheduler.init_value=0.001"

# Every1 summary:
# best noise: 0.00003
# lr: 200k-0.001 = 200k-0.0003 = 500k-0.0001; 200k-0.0001 sucks
# new defaults: noise_std=0.00003, transition_steps=250000, init_value=0.001

#################### hyperparameter tuning every1 ####################
# sbatch scripts/slurm_vel_solver.sh gns simple 1 10 12345 "model.net.noise_std=0.00003 model.scheduler.transition_steps=200000 model.scheduler.init_value=0.001"
# sbatch scripts/slurm_vel_solver.sh gns simple 1 10 12345 "model.net.noise_std=0.00003 model.scheduler.transition_steps=200000 model.scheduler.init_value=0.0003"
# sbatch scripts/slurm_vel_solver.sh gns simple 1 10 12345 "model.net.noise_std=0.0001 model.scheduler.transition_steps=200000 model.scheduler.init_value=0.001"
# sbatch scripts/slurm_vel_solver.sh gns simple 1 10 12345 "model.net.noise_std=0.0003 model.scheduler.transition_steps=200000 model.scheduler.init_value=0.001"
# sbatch scripts/slurm_vel_solver.sh gns simple 1 10 12345 "model.net.noise_std=0.0003 model.scheduler.transition_steps=200000 model.scheduler.init_value=0.0003"

# Every10 summary:
# best noise: 0.0001. similar to 0.00003 and 0.0003
# best lr: 1e-3 same as 3e-4 over first 300k steps
# new defaults: noise_std=0.0001, transition_steps=250000, init_value=0.001

#################### segnn & every1 ####################
# sbatch scripts/slurm_vel_solver.sh segnn simple 1 1 12345
# sbatch scripts/slurm_vel_solver.sh segnn tvf 1 1 12345
# sbatch scripts/slurm_vel_solver.sh segnn simple_u 1 1 12345
# sbatch scripts/slurm_vel_solver.sh segnn simple_u_closure 1 1 12345  # really bad
# sbatch scripts/slurm_vel_solver.sh segnn simple_rlx 1 1 12345
# segnn & every10
# sbatch scripts/slurm_vel_solver.sh segnn simple 1 10 12345
# sbatch scripts/slurm_vel_solver.sh segnn tvf 1 10 12345
# sbatch scripts/slurm_vel_solver.sh segnn simple_u 1 10 12345
# sbatch scripts/slurm_vel_solver.sh segnn simple_rlx 1 10 12345
