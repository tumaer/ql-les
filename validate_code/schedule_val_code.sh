#!/bin/bash
# Schedule execution of many runs
# nohup bash scripts/schedule.sh > gns_kolm2d_every1_simple.log 2>&1 &
# or
# sbatch slurm_schedule.sh

python src/train.py experiment=gns_tgv2d.yaml \
  +logger.wandb.name=gns_tgv_val_lagbench, model.alpha_u=0 \
  model.net.vel_solver=simple ++model.pushforward=null ++vars.max_pushforward_steps=0 \
  ++model.net.noise_std=0 data.shuffle=False

python src/train.py experiment=gns_tgv2d.yaml \
  +logger.wandb.name=gns_tgv_val_lagbench, model.alpha_u=0.000001 \
  model.net.vel_solver=simple ++model.pushforward=null ++vars.max_pushforward_steps=0 \
  ++model.net.noise_std=0 data.shuffle=False

python /home/atoshev/code/lagrangebench/main.py\
  config=validate_code/LB_tgv2d/gns.yaml
