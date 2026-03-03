#!/bin/bash
#SBATCH --job-name=lag
#SBATCH --output=logs/slogs/%x_%A.out
#SBATCH --error=logs/slogs/%x_%A.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1

# sbatch scripts/kolm_everyn.sh

#### Train
# python src/train.py experiment=kolm100/lag.yaml +logger.wandb.name=lag1 \
#   data.name=kolm1 data.every_n=1 data.metadata_file=metadata_every1.json
# python src/train.py experiment=kolm100/lag.yaml +logger.wandb.name=lag10 \
#   data.name=kolm10 data.every_n=10 data.metadata_file=metadata_every10.json
# python src/train.py experiment=kolm100/lag.yaml +logger.wandb.name=lag20 \
#   data.name=kolm20 data.every_n=20 data.metadata_file=metadata_every20.json
# python src/train.py experiment=kolm100/lag.yaml +logger.wandb.name=lag50 \
#   data.name=kolm50 data.every_n=50 data.metadata_file=metadata_every50.json

#### Train with noise
# python src/train.py experiment=kolm100/lag.yaml +logger.wandb.name=lag10 \
#   data.name=kolm10 data.every_n=10 data.metadata_file=metadata_every10.json \
#   model.net.noise_std=0.0001
# python src/train.py experiment=kolm100/lag.yaml +logger.wandb.name=lag10 \
#   data.name=kolm10 data.every_n=10 data.metadata_file=metadata_every10.json \
#   model.net.noise_std=0.0003
# python src/train.py experiment=kolm100/lag.yaml +logger.wandb.name=lag10 \
#   data.name=kolm10 data.every_n=10 data.metadata_file=metadata_every10.json \
#   model.net.noise_std=0.001

#### Train with 5 history steps
# python src/train.py experiment=kolm100/lag.yaml +logger.wandb.name=lag10 \
#   data.name=kolm10 data.every_n=10 data.metadata_file=metadata_every10.json \
#   vars.input_seq_length=6

#### Train with 5 history steps and noise
# python src/train.py experiment=kolm100/lag.yaml +logger.wandb.name=lag10 \
#   data.name=kolm10 data.every_n=10 data.metadata_file=metadata_every10.json \
#   model.net.noise_std=0.0001 vars.input_seq_length=6
# python src/train.py experiment=kolm100/lag.yaml +logger.wandb.name=lag50 \
#   data.name=kolm50 data.every_n=50 data.metadata_file=metadata_every50.json \
#   model.net.noise_std=0.0001 vars.input_seq_length=6

#### Inference
run_basic() {
    python src/eval.py ckpt_path=logs/train/runs/"$1"/checkpoints/last.ckpt \
    model.visualize.vis_test.rollout_dir=logs/train/runs/"$1"/rlt/"$2" \
    logger.wandb.offline=True model.visualize.vis_test.out_type=pkl \
    model.visualize.vis_test.animate_first_n=0 \
    trainer.limit_test_batches=5 data.only_beginning=True \
    model.neuralsph.test.num_steps="$3" vars.num_rollout_steps="$4" "${@:5}"
}

#### 1000 physical
# run_basic "2025-07-07_11-43-25" 11 0 11 model.v2u_solver=same # 100th
# run_basic "2026-03-01_01-43-27" 21 0 21 model.v2u_solver=same # 50th
# run_basic "2026-03-01_01-43-06" 51 0 51 model.v2u_solver=same # 20th
# run_basic "2026-03-01_02-03-36" 101 0 101 model.v2u_solver=same # 10th
# run_basic "2026-03-01_02-02-03" 1001 0 1001 model.v2u_solver=same # 1th

# #### 2000 physical
# run_basic "2025-07-07_11-43-25" 21 0 21 model.v2u_solver=same # 100th
# run_basic "2026-03-01_01-43-27" 41 0 41 model.v2u_solver=same # 50th
# run_basic "2026-03-01_01-43-06" 101 0 101 model.v2u_solver=same # 20th
# run_basic "2026-03-01_02-03-36" 201 0 201 model.v2u_solver=same # 10th
# run_basic "2026-03-01_02-02-03" 2001 0 2001 model.v2u_solver=same # 1th

# #### 2000 physical with NSPH
# run_basic "2025-07-07_11-43-25" 21_nsph3 3 21 model.v2u_solver=same # 100th
# run_basic "2026-03-01_01-43-27" 41_nsph3 3 41 model.v2u_solver=same # 50th
# run_basic "2026-03-01_01-43-06" 101_nsph3 3 101 model.v2u_solver=same # 20th
# run_basic "2026-03-01_02-03-36" 201_nsph3 3 201 model.v2u_solver=same # 10th
# run_basic "2026-03-01_02-02-03" 2001_nsph3 3 2001 model.v2u_solver=same # 1th

# #### 10000 physical with NSPH
# run_basic "2026-03-01_01-43-27" 201_nsph3 3 201 model.v2u_solver=same # 50th
# run_basic "2026-03-01_01-43-06" 501_nsph3 3 501 model.v2u_solver=same # 20th
# run_basic "2026-03-01_02-03-36" 1001_nsph3 3 1001 model.v2u_solver=same # 10th

# Runs with noise on every 10th

# # 2000 steps, no nsph
# run_basic "2026-03-02_01-42-18" 201_00001 0 201 model.v2u_solver=same # std=0.0001
# run_basic "2026-03-02_01-43-03" 201_00003 0 201 model.v2u_solver=same # std=0.0003
# run_basic "2026-03-02_01-43-30" 201_0001 0 201 model.v2u_solver=same # std=0.001
# # 10000 steps
# run_basic "2026-03-02_01-42-18" 1001_n3_00001 3 1001 model.v2u_solver=same # std=0.0001
# run_basic "2026-03-02_01-43-03" 1001_n3_00003 3 1001 model.v2u_solver=same # std=0.0003
# run_basic "2026-03-02_01-43-30" 1001_n3_0001 3 1001 model.v2u_solver=same # std=0.001

# Runs with 5 historic steps

# # 2000 steps, no nsph, on every 10th
# run_basic "2026-03-02_09-32-50" 201_h5 0 201 model.v2u_solver=same # std=0, h5
# run_basic "2026-03-02_14-52-38" 201_00001_h5 0 201 model.v2u_solver=same # std=0.0001, h5
# # 10000 steps, with nsph, on every 10th
# run_basic "2026-03-02_09-32-50" 1001_n3_h5 3 1001 model.v2u_solver=same
# run_basic "2026-03-02_14-52-38" 1001_n3_00001_h5 3 1001 model.v2u_solver=same
# 2000 steps, no nsph, on every 50th
# run_basic "2026-03-02_23-14-55" 41_h5 0 41 model.v2u_solver=same # std=00001, h5
# # 10000 steps, with nsph, on every 50th
# run_basic "2026-03-02_23-14-55" 201_n1_h5 1 201 model.v2u_solver=same # std=00001, h5
# run_basic "2026-03-02_23-14-55" 201_n3_h5 3 201 model.v2u_solver=same # std=00001, h5
# run_basic "2026-03-02_23-14-55" 201_n5_h5 5 201 model.v2u_solver=same # std=00001, h5

# python notebooks/everyn.py
