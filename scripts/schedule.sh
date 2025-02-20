#!/bin/bash
# Schedule execution of many runs
# nohup bash scripts/schedule.sh > gns_kolm2d_every1_simple.log 2>&1 &
# or
# sbatch slurm_schedule.sh

python src/train.py experiment=gns_kolm2d_every1.yaml \
  +logger.wandb.name=gns_simple model.alpha_u=1 model.vel_solver=simple