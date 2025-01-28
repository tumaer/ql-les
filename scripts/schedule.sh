#!/bin/bash
# Schedule execution of many runs
# nohup bash scripts/schedule.sh > koml2d.log 2>&1 &

python src/train.py trainer.devices=[0] experiment=gns_train_eval_2d_kolm.yaml \
  +logger.wandb.name=koml2d model.alpha_u=1.0 model.vel_solver=simple

python src/train.py trainer.devices=[0] experiment=gns_train_eval_2d_kolm.yaml \
  +logger.wandb.name=koml2d model.alpha_u=0.0000001 model.vel_solver=simple