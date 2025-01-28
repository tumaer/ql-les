#!/bin/bash
#SBATCH --job-name=vel_solvers
#SBATCH --output=slogs/vel_solvers_%A_%a.out
#SBATCH --error=slogs/vel_solvers_%A_%a.err
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p slogs

# Define the cases array
cases=("simple" "tvf" "neural_sph")

# Get the current case based on array task ID
case=${cases[$SLURM_ARRAY_TASK_ID]}

# Launch with: sbatch scripts/slurm_vel_solver.sh
echo "Running case: ${case}"
python src/train.py experiment=gns_train_eval_2d_kolm.yaml \
  +logger.wandb.name=koml2d_${case} model.alpha_u=1.0 model.vel_solver=${case}
