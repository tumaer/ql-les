#!/bin/bash
#SBATCH --job-name=lag
#SBATCH --output=logs/slogs/%x_%A.out
#SBATCH --error=logs/slogs/%x_%A.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

# sbatch scripts/paper/kolm1_lag_v2.sh

#### Train
# python src/train.py experiment=kolm1/lag.yaml +logger.wandb.name=lag
# python src/train.py experiment=kolm1/u2v_gns.yaml +logger.wandb.name=u2v_gns
# python src/train.py experiment=kolm1/gino.yaml +logger.wandb.name=gino
# python src/train.py experiment=kolm1/gns.yaml +logger.wandb.name=gns_simple
# python src/train.py experiment=kolm1/gino.yaml +logger.wandb.name=gino model.optimizer.lr=0.02
# python src/train.py experiment=kolm1/gino.yaml +logger.wandb.name=gino model.optimizer.lr=0.01
# python src/train.py experiment=kolm1/gino.yaml +logger.wandb.name=gino model.optimizer.lr=0.001

#### Inference
run_basic() {
    python src/eval.py ckpt_path=logs/train/runs/"$1"/checkpoints/last.ckpt \
    model.visualize.vis_test.rollout_dir=logs/train/runs/"$1"/rlt/"$2" \
    logger.wandb.offline=True model.visualize.vis_test.out_type=pkl \
    model.visualize.vis_test.animate_first_n=0 \
    trainer.limit_test_batches=5 data.only_beginning=True \
    model.neuralsph.test.num_steps="$3" vars.num_rollout_steps="$4" "${@:5}"
}

#### 101
# run_basic "2025-07-06_06-19-37" 101 0 101 model.v2u_solver=same # lag
# run_basic "2025-07-06_11-05-20" 101 0 101 # u2v_gns
# run_basic "2025-07-06_11-14-14" 101 0 101 # gns_simple
# run_basic "2025-07-06_11-22-57" 101 0 101 # gino

#### 1001
# run_basic "2025-07-06_06-19-37" 1001 0 1001 model.v2u_solver=same # lag
# run_basic "2025-07-06_11-05-20" 1001 0 1001 # u2v_gns
# run_basic "2025-07-06_11-14-14" 1001 0 1001 # gns_simple

# run_basic "2025-07-06_06-19-37" 1001_nsph1 1 1001 model.v2u_solver=same # lag
# run_basic "2025-07-06_11-05-20" 1001_nsph1 1 1001 # u2v_gns
# run_basic "2025-07-06_11-05-20" 1001_nsph1_nu0001 1 1001 # u2v_gns
# run_basic "2025-07-06_11-05-20" 1001_nsph1_nu001 1 1001 # u2v_gns
# run_basic "2025-07-06_11-14-14" 1001_nsph1 1 1001 # gns_simple


# python notebooks/spectra_and_ekin.py
