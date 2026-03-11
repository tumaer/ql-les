#!/bin/bash
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
./build/solver --config cfg/tgv2d_init.conf
python init_u.py --type tgv2d --src "res/tgv2d_init/state_step_00006000.bin"

# Step 2: production runs
./build/solver --config cfg/tgv2d_tvf.conf
./build/solver --config cfg/tgv2d_notvf.conf

# Step 3: analysis
python analyse.py --type tgv2d --path res/tgv2d_init
python analyse.py --type tgv2d --path res/tgv2d_tvf
python analyse.py --type tgv2d --path res/tgv2d_notvf
