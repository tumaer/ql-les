#!/bin/bash
# Schedule execution of many runs
# nohup bash scripts/schedule.sh > koml2d.log 2>&1 &
# or
# sbatch slurm_schedule.sh

python src/train.py experiment=gns_train_eval_2d_kolm.yaml \
  +logger.wandb.name=koml2d_u0 model.alpha_u=0.0000001 model.vel_solver=simple