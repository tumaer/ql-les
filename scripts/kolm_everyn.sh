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

# python notebooks/everyn.py
