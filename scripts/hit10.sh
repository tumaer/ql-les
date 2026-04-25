#!/bin/bash
#SBATCH --job-name=hit10
#SBATCH --output=logs/slogs/%x_%A.out
#SBATCH --error=logs/slogs/%x_%A.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# sbatch scripts/hit10.sh

source /usr/local/cuda-12.1.sh
export TORCH_CUDA_ARCH_LIST=8.6 # Set CUDA arch to avoid compilation warning

### Overfit
# python src/train.py experiment=hit10/lag.yaml debug=overfit trainer.accelerator=gpu

### Train
train_basic() { python src/train.py experiment=hit10/"$1".yaml +logger.wandb.name="$2" "${@:3}"; }
# >> Noise runs
train_basic lag lag
train_basic lag lag model.net.noise_std=1e-5
train_basic lag lag model.net.noise_std=3e-5
train_basic lag lag model.net.noise_std=1e-4
train_basic lag lag model.net.noise_std=3e-4
train_basic v2u_gnn v2u_gnn

### Inference
# For timing, set `model.visualize.vis_test.out_type=none`, `model.v2u_solver=none`, and:
# export TIME_ROLLOUT=1
run_basic() {
    python src/eval.py ckpt_path=logs/train/runs/"$1"/checkpoints/last.ckpt \
    model.visualize.vis_test.rollout_dir=logs/train/runs/"$1"/rlt/"$2" \
    model.visualize.vis_test.animate_first_n=0 model.visualize.vis_test.out_type=pkl \
    model.neuralsph.test.is_tvf=False model.neuralsph.test.num_steps="$3" \
    model.neuralsph.test.dt_factor=2 trainer.limit_test_batches=5 data.only_beginning=True \
    model.net.v2u_gnn.ckpt_path=logs/train/runs/2026-04-04_03-51-01/checkpoints/last.ckpt \
    logger.wandb.offline=True vars.num_rollout_steps="$4" "${@:5}"
}

# # LAG noise ablation
# run_basic 2026-04-03_12-50-53 249_2n1_debug 1 249 model.v2u_solver=gnn # LAG noise=0
# run_basic 2026-04-05_05-02-10 249_2n1 1 249 model.v2u_solver=gnn # LAG noise=1e-5
# run_basic 2026-04-03_18-37-18 249_2n1 1 249 model.v2u_solver=gnn # LAG noise=3e-5
# run_basic 2026-04-03_22-04-27 249_2n1 1 249 model.v2u_solver=gnn # LAG noise=1e-4
# run_basic 2026-04-04_03-52-00 249_2n1 1 249 model.v2u_solver=gnn # LAG noise=3e-4
# run_basic 2026-04-03_22-05-17 249_2n1 1 249 model.v2u_solver=gnn # LAG noise=1e-3

# run_basic 2026-04-19_13-35-06 249_2n1 1 249 model.v2u_solver=gnn # LAG noise=3e-5 seed=124
# run_basic 2026-04-19_13-38-16 249_2n1 1 249 model.v2u_solver=gnn # LAG noise=3e-5 seed=125
