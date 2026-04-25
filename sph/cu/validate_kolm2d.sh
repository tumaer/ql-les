#!/bin/bash
#SBATCH --job-name=kolm2d_valid
#SBATCH --output=slogs/kolm2d_%A_%a.out
#SBATCH --error=slogs/kolm2d_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

set -e

mkdir -p build
bash -c "source /usr/local/cuda-12.1.sh && nvcc -O3 -std=c++17 solver.cu -o build/solver"

for resolution in 64 128 256 512; do
    # Step 1: relaxation pass
    ./build/solver --config cfg/kolm2d_${resolution}_init.conf
    python analyse.py --case kolm2d --path res/kolm2d_${resolution}_init

    # Step 2: inject the Kolmogorov velocity field onto the relaxed frames
    # Requires following data: res/kolm2d/traj_{15..19}/u_512_04500_burnin.bin
    cp -r res/kolm2d/. res/kolm2d_${resolution}
    python init_u_kolm_or_hit.py --src_u "res/kolm2d_${resolution}" \
      --src_pos "res/kolm2d_${resolution}_init/state_step_00002000.bin"

    # Step 3: trajectory runs
    for traj in 15 16 17 18 19; do
        # to include correction matrix A, add: --is_tvf_stress 1
        ./build/solver --config cfg/kolm2d_${resolution}.conf \
          --save_dir res/kolm2d_${resolution}/traj_${traj} \
          --init_state_file res/kolm2d_${resolution}/traj_${traj}/relaxed_state.bin
        python analyse.py --case kolm2d --path res/kolm2d_${resolution}/traj_${traj}
        echo "    Finished Nx=${resolution} traj=${traj}"
    done

    python analyse_kolm_hit.py --case kolm2d --path res/kolm2d_${resolution} \
      --ref-path /local/disk/atoshev/dataset_kolm/raw/2D_KOLM_4096_140kevery1 \
      --burnin-steps-dns 45 --recompute-spectra --recompute-corr
    echo "Finished Nx=${resolution}"
done

python analyse_kolm_hit_2.py --case kolm2d --path res --burnin-steps-dns 45 \
  --ref-path /local/disk/atoshev/dataset_kolm/raw/2D_KOLM_4096_140kevery1

python analyse_kolm_hit_3.py --case kolm2d --path res --burnin-steps-dns 45 \
  --ref-path /local/disk/atoshev/dataset_kolm/raw/2D_KOLM_4096_140kevery1
