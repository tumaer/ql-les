#!/bin/bash
#SBATCH --job-name=hit3d_valid
#SBATCH --output=slogs/hit3d_%A_%a.out
#SBATCH --error=slogs/hit3d_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

set -e

mkdir -p build
bash -c "source /usr/local/cuda-12.1.sh && nvcc -O3 -std=c++17 solver.cu -o build/solver"

for resolution in 32 64 128; do
    # Step 1: relaxation pass
    ./build/solver --config cfg/hit3d_${resolution}_init.conf
    python analyse.py --case hit3d --path res/hit3d_${resolution}_init

    # Step 2: inject the HIT velocity field onto the relaxed frames
    # Requires following data: res/hit3d/traj_{15..19}/u_512_04500_burnin.bin
    cp -r res/hit3d/. res/hit3d_${resolution}
    python init_u_kolm_or_hit.py --src_u "res/hit3d_${resolution}" \
      --src_pos "res/hit3d_${resolution}_init/state_step_00002000.bin"

    # Step 3: trajectory runs
    for traj in 15 16 17 18 19; do
        ./build/solver --config cfg/hit3d_${resolution}.conf \
          --save_dir res/hit3d_${resolution}/traj_${traj} \
          --init_state_file res/hit3d_${resolution}/traj_${traj}/relaxed_state.bin
        python analyse.py --case hit3d --path res/hit3d_${resolution}/traj_${traj}
        echo "    Finished Nx=${resolution} traj=${traj}"
    done

    python analyse_kolm_hit.py --case hit3d --path res/hit3d_${resolution} \
      --ref-path /local/disk/atoshev/dataset_hit/raw/3D_HIT_32768_20kevery1 \
      --burnin-steps-dns 250 --recompute-spectra
    echo "Finished Nx=${resolution}"
done

python analyse_kolm_hit_2.py --case hit3d --path res --burnin-steps-dns 250 \
  --ref-path /local/disk/atoshev/dataset_hit/raw/3D_HIT_32768_20kevery1
