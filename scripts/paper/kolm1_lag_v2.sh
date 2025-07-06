#!/bin/bash
#SBATCH --job-name=lag
#SBATCH --output=logs/slogs/%x_%A.out
#SBATCH --error=logs/slogs/%x_%A.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

# nohup bash scripts/paper/kolm1_lag_v2.sh > lag_kolm1.log 2>&1 &
# or
# sbatch scripts/paper/kolm1_lag_v2.sh 101 0 101 same

#### Train
python src/train.py experiment=kolm1/lag.yaml +logger.wandb.name=lag
