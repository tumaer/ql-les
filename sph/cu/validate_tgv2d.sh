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
./build/solver --config cfg/tgv2d_100_init.conf
python init_u_tgv.py --case tgv2d --src "res/tgv2d_100_init/state_step_00002000.bin"

# Step 2: production runs
./build/solver --config cfg/tgv2d_100_tvf.conf
./build/solver --config cfg/tgv2d_100_notvf.conf
./build/solver --config cfg/tgv2d_100_tvf.conf --is_tvf_stress 1 --save_dir res/withA/tgv2d_100_tvf

# Step 3: analysis
python analyse.py --case tgv2d --path res/tgv2d_100_init
python analyse.py --case tgv2d --path res/tgv2d_100_tvf
python analyse.py --case tgv2d --path res/tgv2d_100_notvf
python analyse.py --case tgv2d --path res/withA/tgv2d_100_tvf

# Step 4: validation
python validate_tgv2d.py
