echo "start lagbench validation"

source /home/tkalinov/code/sph_les_directory/sph_les/.venv/bin/activate

python sph_les/src/train.py experiment=gns_tgv2d_validation +logger.wandb.name=gns_tgv_single_traj \
 model.net.vel_solver=simple