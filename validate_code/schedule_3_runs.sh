#!/bin/bash
# Unified script for running multiple experiments with different environments
# Run this script using: nohup bash schedule.sh > gns_kolm2d_every1_simple.log 2>&1 &

echo "Starting scheduled ML runs..."

# =============================
# Run first training script in GNS environment
# =============================
echo "Activating GNS environment..."
source /home/tkalinov/code/sph_les_directory/sph_les/.venv/bin/activate

echo "Running first training script..."
python /home/tkalinov/code/sph_les_directory/sph_les/src/train.py experiment=gns_tgv2d \
  "+logger.wandb.name=gns_tgv_val_lagbench_u0" "model.alpha_u=0" \
  "model.net.vel_solver=simple" "++model.pushforward=null" "++vars.max_pushforward_steps=0" \
  "++model.net.noise_std=0" "data.shuffle=False"

echo "First training run completed."

# =============================
# Run second training script in GNS environment
# =============================

echo "Running second training script..."
python /home/tkalinov/code/sph_les_directory/sph_les/src/train.py experiment=gns_tgv2d \
  "+logger.wandb.name=gns_tgv_val_lagbench_unot0" "model.alpha_u=0.000001" \
  "model.net.vel_solver=simple" "++model.pushforward=null" "++vars.max_pushforward_steps=0" \
  "++model.net.noise_std=0" "data.shuffle=False"

echo "Second training run completed."
echo "Deactivating GNS environment..."
deactivate

# =============================
# Run validation script in LagrangeBench environment
# =============================
echo "Activating LagrangeBench environment..."
source /home/atoshev/code/lagrangebench/.venv/bin/activate

echo "Running validation script..."
python /home/atoshev/code/lagrangebench/main.py config=/home/tkalinov/code/sph_les_directory/sph_les/validate_code/LB_tgv2d/gns.yaml

echo "Validation run completed."
deactivate

echo "All tasks completed successfully."