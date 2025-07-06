#!/bin/bash
#SBATCH --job-name=lag
#SBATCH --output=logs/slogs/%x_%A_%a.out
#SBATCH --error=logs/slogs/%x_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

# nohup bash scripts/paper/kolm10_lag.sh > lag_kolm10.log 2>&1 &
# or
# sbatch scripts/paper/kolm10_lag.sh 101 0 101 same

#### Train
# python src/train.py experiment=kolm_every_10/lag.yaml +logger.wandb.name=lag \
#     model.net.num_message_passing_steps=5
# # change default_connectivity_radius=0.2 from 0.15
# python src/train.py experiment=kolm_every_10/lag.yaml +logger.wandb.name=lag \
#     model.net.num_message_passing_steps=5
# python src/train.py experiment=kolm_every_10/lag.yaml +logger.wandb.name=lag
# Summary: 5 layer and cutoff=0.2 is a bit worse than 10 layer and cutoff=0.15 (default)
#     but 5 layer with cutoff=0.15 is almost 2x worse on val error than 10/0.15

# python src/train.py experiment=kolm_every_10/v2u_gnn.yaml +logger.wandb.name=v2u10
# or
# sbatch -J v2u10 scripts/slurm_train.sh "experiment=kolm_every_10/v2u_gnn.yaml +logger.wandb.name=v2u10"

#### Inference
# model with 3e-5 noise: 2025-06-15_23-18-37
# v2u1 model: 2025-06-16_03-27-47
run_basic() {
    python src/eval.py ckpt_path=logs/train/runs/2025-06-15_23-15-04/checkpoints/last.ckpt \
    model.visualize.vis_test.rollout_dir=logs/train/runs/2025-06-15_23-15-04/rlt/"$1" \
    logger.wandb.offline=True model.visualize.vis_test.out_type=pkl \
    model.visualize.vis_test.animate_first_n=0 model.v2u_solver="$4" \
    trainer.limit_test_batches=5 data.only_beginning=True \
    model.net.v2u_gnn.ckpt_path="logs/train/runs/2025-06-16_03-25-44/checkpoints/last.ckpt" \
    model.neuralsph.test.num_steps="$2" vars.num_rollout_steps="$3" "${@:5}"
}
# run_basic "$1" "$2" "$3" "$4" ${@:5}

# run_noisy() {
#     python src/eval.py ckpt_path=logs/train/runs/2025-06-15_23-18-37/checkpoints/last.ckpt \
#     model.visualize.vis_test.rollout_dir=logs/train/runs/2025-06-15_23-18-37/rlt/"$1" \
#     logger.wandb.offline=True model.visualize.vis_test.out_type=pkl \
#     model.visualize.vis_test.animate_first_n=0 model.v2u_solver="$4" \
#     trainer.limit_test_batches=5 data.only_beginning=True \
#     model.net.v2u_gnn.ckpt_path="logs/train/runs/2025-06-16_03-25-44/checkpoints/last.ckpt" \
#     model.neuralsph.test.num_steps="$2" vars.num_rollout_steps="$3" "${@:5}"
# }
# run_noisy "$1" "$2" "$3" "$4" ${@:5}

#### 101
# run_basic 101 0 101 same
# run_basic 101_nsph1 1 101 same
# run_basic 101_nsph1tvf1 1 101 same model.neuralsph.test.is_tvf=True
# run_basic 101_gnn 0 101 gnn
# run_basic 101_gnn_nsph1 1 101 gnn

#### 500
# sbatch scripts/paper/kolm10_lag.sh 500_rho 1 501 gnn
# sbatch scripts/paper/kolm10_lag.sh 500_rhotvf1 1 501 gnn model.neuralsph.test.is_tvf=True
# sbatch scripts/paper/kolm10_lag.sh 500_rhotvf2 1 501 gnn model.neuralsph.cfg_every=2 \
#     model.neuralsph.test.cfg.is_tvf=True model.neuralsph.test.cfg.tvf_factor=1.0

# sbatch scripts/paper/kolm10_lag.sh 500_rho_nu 1 501 gnn  # nu=0.001
# sbatch scripts/paper/kolm10_lag.sh 500_rho_nu001 1 501 gnn  # nu=0.01

# sbatch scripts/paper/kolm10_lag.sh 500_rho_artif1 1 501 gnn

# sbatch scripts/paper/kolm10_lag.sh 500_rho_sph2 1 501 gnn
# sbatch scripts/paper/kolm10_lag.sh 500_rho101 1 501 gnn  # rho_max=1.01
# sbatch scripts/paper/kolm10_lag.sh 500_rho10 1 501 gnn  # 10 relaxation steps
