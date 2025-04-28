# nohup bash /home/tkalinov/code/sph_les_directory/sph_les/validate_code/tgv_torchbench/schedule_torchbench.sh > gns_torch_64.log 2>&1 &
echo "start lagbench validation"

source /home/tkalinov/code/sph_les_directory/sph_les/.venv/bin/activate

python /home/tkalinov/code/sph_les_directory/sph_les/src/train.py experiment=gns_tgv2d_validation +logger.wandb.name=gns_tgv_single_traj_64 \
 model.net.vel_solver=simple
