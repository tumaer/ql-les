#!/bin/bash
#SBATCH --job-name=lag
#SBATCH --output=logs/slogs/%x_%A.out
#SBATCH --error=logs/slogs/%x_%A.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1

source .env

# sbatch scripts/paper/kolm100_lag_v2.sh

#### Train
# python src/train.py experiment=kolm100/lag.yaml +logger.wandb.name=lag

#### Inference
run_basic() {
    python src/eval.py ckpt_path=logs/train/runs/"$1"/checkpoints/last.ckpt \
    model.visualize.vis_test.rollout_dir=logs/train/runs/"$1"/rlt/"$2" \
    logger.wandb.offline=True model.visualize.vis_test.out_type=pkl \
    model.visualize.vis_test.animate_first_n=0 \
    trainer.limit_test_batches=5 data.only_beginning=True \
    model.neuralsph.test.num_steps="$3" vars.num_rollout_steps="$4" "${@:5}"
}

#### 21 - non of the below really worked -> explore smaller coarse-graining factors!
# run_basic "2025-07-07_11-43-25" 21 0 21 model.v2u_solver=same # lag
# run_basic "2025-07-07_11-43-25" 21_nsph1 1 21 model.v2u_solver=same # lag
# run_basic "2025-07-07_11-43-25" 21_nsph5 5 21 model.v2u_solver=same # lag
# run_basic "2025-07-07_11-43-25" 11_nsph5 5 11 model.v2u_solver=same # lag

# python notebooks/spectra_and_ekin.py
