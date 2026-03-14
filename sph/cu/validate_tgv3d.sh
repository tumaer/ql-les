#!/bin/bash
#SBATCH --job-name=tgv3d_cuda
#SBATCH --output=slogs/tgv3d_cuda_%A_%a.out
#SBATCH --error=slogs/tgv3d_cuda_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# End-to-end validation of the 2-D SPH TGV baseline simulation.
#
# Steps:
#   1. Build the CUDA binary.
#   2. Run a short relaxation pass to settle particle positions.
#   3. Re-inject the analytic TGV velocity field (tgv2d_init.py).
#   4. Run production simulations with and without TVF.
#   5. Produce diagnostic plots for both runs.

set -e

source /usr/local/cuda-12.1.sh

mkdir -p build
nvcc -O3 -std=c++17 solver.cu -o build/solver

# Step 1: relaxation pass
./build/solver --config cfg/tgv3d_64_init.conf
python init_u_tgv.py --type tgv3d --src "res/tgv3d_64_init/state_step_00002500.bin"

# # Step 2: production runs
./build/solver --config cfg/tgv3d_64_tvf.conf

# # Step 3: analysis
python analyse.py --type tgv3d --path res/tgv3d_64_init
python analyse.py --type tgv3d --path res/tgv3d_64_tvf
