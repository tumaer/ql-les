#!/bin/bash
#SBATCH --job-name=kolm10
#SBATCH --output=logs/slogs/%x_%A.out
#SBATCH --error=logs/slogs/%x_%A.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# sbatch scripts/kolm10.sh

source /usr/local/cuda-12.1.sh
export TORCH_CUDA_ARCH_LIST=8.6 # Set CUDA arch to avoid compilation warning

### Overfit runs
overfit_basic() { python src/train.py experiment=kolm10/"$1".yaml debug=overfit \
    trainer.accelerator=gpu model.optimizer.lr=0.001 "${@:3}"; }
# overfit_basic lag model.net.num_message_passing_steps=2
# overfit_basic gns model.net.num_message_passing_steps=2
# overfit_basic gns model.net.num_message_passing_steps=2 model.vel_solver=tvf
# overfit_basic gns model.net.num_message_passing_steps=2 model.vel_solver=simple_u
# overfit_basic gino model.neuralsph.val.num_steps=3 model.neuralsph.val.dt_factor=2 \
#     model.neuralsph.test.num_steps=3 model.neuralsph.test.dt_factor=2

### Training runs
train_basic() { python src/train.py experiment=kolm10/"$1".yaml +logger.wandb.name="$2" "${@:3}"; }
# >>> Variants
train_basic lag lag
train_basic gns simple model.vel_solver=simple
train_basic gns tvf model.vel_solver=tvf
train_basic gns simple_u model.vel_solver=simple_u
train_basic gino gino model.neuralsph.val.num_steps=3 model.neuralsph.val.dt_factor=2 \
    model.neuralsph.test.num_steps=3 model.neuralsph.test.dt_factor=2
train_basic v2u_gnn v2u_gnn

# >>> Noise
train_basic lag lag_noise1e5 model.net.noise_std=1e-5
train_basic lag lag_noise3e5 model.net.noise_std=3e-5
train_basic lag lag_noise1e4 model.net.noise_std=1e-4
train_basic lag lag_noise3e4 model.net.noise_std=3e-4

# # >>> Weight decay -> no clear effect, so skipping for now
# train_basic lag lag_wd1e5 model.optimizer.weight_decay=1.e-5
# train_basic lag lag_wd1e4 model.optimizer.weight_decay=1.e-4
# train_basic lag lag_wd1e3 model.optimizer.weight_decay=1.e-3

### Inference
# For timing, set `model.visualize.vis_test.out_type=none`, `model.v2u_solver=none`, and:
# export TIME_ROLLOUT=1
run_basic() {
    python src/eval.py ckpt_path=logs/train/runs/"$1"/checkpoints/last.ckpt \
    model.visualize.vis_test.rollout_dir=logs/train/runs/"$1"/rlt/"$2" \
    model.visualize.vis_test.animate_first_n=0 model.visualize.vis_test.out_type=pkl \
    model.neuralsph.test.is_tvf=False model.neuralsph.test.num_steps="$3" \
    model.neuralsph.test.dt_factor=2 trainer.limit_test_batches=5 data.only_beginning=True \
    logger.wandb.offline=True vars.num_rollout_steps="$4" "${@:5}"
}

# 100 steps, all variants
# run_basic 2026-04-02_00-02-07 101_2n3 3 101 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-04_01-02-04/checkpoints/last.ckpt # LAG
# run_basic 2026-04-02_00-03-14 101_2n3 3 101 # GNS simple
# run_basic 2026-04-02_00-03-36 101_2n3 3 101 # GNS tvf
# run_basic 2026-04-02_00-03-57 101_2n3 3 101 # GNS simple_u
# run_basic 2026-04-02_00-04-22 101_2n3 3 101 # GINO

# Full run
# run_basic 2026-04-02_00-02-07 1399_2n3 3 1399 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-04_01-02-04/checkpoints/last.ckpt # LAG
# run_basic 2026-04-02_00-04-22 1399_2n3 3 1399 # GINO

# Noise comparisons
# run_basic 2026-04-02_00-13-43 1399_2n3 3 1399 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-04_01-02-04/checkpoints/last.ckpt # LAG noise=1e-5
# run_basic 2026-04-02_00-13-54 1399_2n3 3 1399 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-04_01-02-04/checkpoints/last.ckpt # LAG noise=3e-5
# run_basic 2026-04-02_16-34-44 1399_2n3 3 1399 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-04_01-02-04/checkpoints/last.ckpt # LAG noise=1e-4
# run_basic 2026-04-02_04-59-47 1399_2n3 3 1399 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-04_01-02-04/checkpoints/last.ckpt # LAG noise=3e-4

# Weight decay comparisons
# run_basic 2026-04-02_21-44-51 1399_2n3 3 1399 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-04_01-02-04/checkpoints/last.ckpt # LAG wd=1e-5
# run_basic 2026-04-02_16-35-08 1399_2n3 3 1399 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-04_01-02-04/checkpoints/last.ckpt # LAG wd=1e-4
# run_basic 2026-04-02_22-06-01 1399_2n3 3 1399 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-04_01-02-04/checkpoints/last.ckpt # LAG wd=1e-3
