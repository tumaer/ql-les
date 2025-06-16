#!/bin/bash
# nohup bash scripts/paper/lag.sh > lag_kolm2d_every1_20000_nsph.log 2>&1 &

#### Train
# python src/train.py experiment=kolm_every_10/lag.yaml +logger.wandb.name=lag
# python src/train.py experiment=kolm_every_10/v2u_gnn.yaml +logger.wandb.name=v2u10
# or
# sbatch -J v2u10 scripts/slurm_train.sh "experiment=kolm_every_10/v2u_gnn.yaml +logger.wandb.name=v2u10"
