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
nvcc -O3 -std=c++17 tgv_cuda.cu -o build/tgv_cuda

# Step 1: relaxation pass
./build/tgv_cuda --config tgv_cuda_init.conf
python tgv2d_init.py

# Step 2: production runs
./build/tgv_cuda --config tgv_cuda_tvf.conf
./build/tgv_cuda --config tgv_cuda_notvf.conf

# Step 3: analysis
python analyse.py res_tgv2d_init
python analyse.py res_tgv2d_tvf
python analyse.py res_tgv2d_notvf
