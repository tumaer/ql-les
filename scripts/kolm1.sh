#!/bin/bash
#SBATCH --job-name=kolm1
#SBATCH --output=logs/slogs/%x_%A.out
#SBATCH --error=logs/slogs/%x_%A.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# sbatch scripts/kolm1.sh

source /usr/local/cuda-12.1.sh
export TORCH_CUDA_ARCH_LIST=8.6 # Set CUDA arch to avoid compilation warning

### Overfit
overfit_basic() { python src/train.py experiment=kolm1/"$1".yaml debug=overfit \
    trainer.accelerator=gpu model.optimizer.lr=0.001 "${@:3}"; }
# overfit_basic lag model.net.num_message_passing_steps=2
# overfit_basic gns model.net.num_message_passing_steps=2 model.vel_solver=simple
# overfit_basic gns model.net.num_message_passing_steps=2 model.vel_solver=tvf
# overfit_basic gns model.net.num_message_passing_steps=2 model.vel_solver=simple_u
# overfit_basic gino

### Training
train_basic() { python src/train.py experiment=kolm1/"$1".yaml +logger.wandb.name="$2" "${@:3}"; }
# >>> Variants runs
train_basic lag lag
train_basic gns simple model.vel_solver=simple
train_basic gns tvf model.vel_solver=tvf
train_basic gns simple_u model.vel_solver=simple_u
train_basic gino gino
train_basic v2u_gnn v2u_gnn

### Inference
run_basic() {
    python src/eval.py ckpt_path=logs/train/runs/"$1"/checkpoints/last.ckpt \
    model.visualize.vis_test.rollout_dir=logs/train/runs/"$1"/rlt/"$2" \
    model.visualize.vis_test.animate_first_n=0 model.visualize.vis_test.out_type=pkl \
    model.neuralsph.test.is_tvf=False model.neuralsph.test.num_steps="$3" \
    trainer.limit_test_batches=5 data.only_beginning=True \
    logger.wandb.offline=True vars.num_rollout_steps="$4" "${@:5}"
}

# # 500 steps, no NSPH
# run_basic 2026-04-02_20-40-23 501 0 501 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-03_03-15-00/checkpoints/last.ckpt # LAG
# run_basic 2026-04-03_01-42-23 501 0 501 # GNS simple
# run_basic 2026-04-03_12-32-13 501 0 501 # GNS tvf
# run_basic 2026-04-03_17-49-09 501 0 501 # GNS simple_u
# run_basic 2026-04-02_20-40-40 501 0 501 # GINO

# # 500 steps, 1n3
# run_basic 2026-04-02_20-40-23 501_1n3 3 501 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-03_03-15-00/checkpoints/last.ckpt # LAG
# run_basic 2026-04-03_01-42-23 501_1n3 3 501 # GNS simple
# run_basic 2026-04-03_12-32-13 501_1n3 3 501 # GNS tvf
# run_basic 2026-04-03_17-49-09 501_1n3 3 501 # GNS simple_u
# run_basic 2026-04-02_20-40-40 501_1n3 3 501 # GINO

# # 2000 steps, 1n3
# run_basic 2026-04-02_20-40-23 2001_1n3 3 2001 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-03_03-15-00/checkpoints/last.ckpt # LAG
# run_basic 2026-04-03_01-42-23 2001_1n3 3 2001 # GNS simple
# run_basic 2026-04-03_12-32-13 2001_1n3 3 2001 # GNS tvf
# run_basic 2026-04-03_17-49-09 2001_1n3 3 2001 # GNS simple_u
# run_basic 2026-04-02_20-40-40 2001_1n3 3 2001 # GINO

# # Full run, 1n3
# run_basic 2026-04-02_20-40-23 13999_1n1 1 13999 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-03_03-15-00/checkpoints/last.ckpt # LAG
# run_basic 2026-04-02_20-40-23 13999_1n3 3 13999 model.v2u_solver=gnn \
#   model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-03_03-15-00/checkpoints/last.ckpt # LAG
# run_basic 2026-04-02_20-40-23 13999_1n1_nov2u 1 13999 model.v2u_solver=same
# run_basic 2026-04-02_20-40-40 13999 0 13999 # GINO
# run_basic 2026-04-02_20-40-40 13999_1n3 3 13999 # GINO
