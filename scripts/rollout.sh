#!/bin/bash
#SBATCH --job-name=rollout
#SBATCH --output=logs/%x.log
#SBATCH --error=logs/%x.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

if [ "$#" -ne 2 ]; then
  echo "Wrong number of arguments. Currently "$#", should be 2"
  echo "Usage: sbatch scripts/rollout.sh <EVERY_N> <NSPH_SUFFIX>"
  echo Currently supported: EVERY_N={1, 10}, NSPH_SUFFIX={"", "_neuralsph"}
  exit 1
fi

EVERY_N=$1
nsph_suffix=$2

if [ $EVERY_N -eq 1 ]; then # every1
    rlt_len=50 # 1000
    ckpts=(
        "2025-02-20_11-51-24"  # simple
        "2025-02-20_22-13-57"  # tvf
        "2025-02-20_22-14-24"  # simple_u
        "2025-02-20_22-14-35"  # simple_rlx
    )
elif [ $EVERY_N -eq 10 ]; then # every10
    rlt_len=5 # 100
    ckpts=(
        "2025-02-20_11-51-35"  # simple
        "2025-02-20_22-14-49"  # tvf
        "2025-02-20_23-09-21"  # simple_u
        "2025-02-20_23-36-01"  # simple_rlx
    )
else
    echo "EVERY_N=${EVERY_N} not supported"
    exit
fi

run_base() {
    python src/eval.py \
        ckpt_path="logs/train/runs/${ckpt}/checkpoints/last.ckpt" \
        vars.num_rollout_steps=${rlt_len} \
        logger.wandb.offline=True "$@"
        # trainer.limit_test_batches=1 \
        # model.visualize.vis_test.rollout_dir="logs/paper/${ckpt}/rollouts_${rlt_len}${NSPH_SUFFIX}" \
        # model.visualize.vis_test.out_type=pkl \
}

if [ -z "$NSPH_SUFFIX" ]; then  # no NeuralSPH
    echo "Running without NeuralSPH"
    for ckpt in "${ckpts[@]}"; do
        run_base
    done
elif [ "$NSPH_SUFFIX" == "_neuralsph" ]; then
    echo "Running with NeuralSPH"
    for ckpt in "${ckpts[@]}"; do
        run_base model.neuralsph.test.dt_factor=2 model.neuralsph.test.num_steps=10
    done
else
    echo "NSPH_SUFFIX=${NSPH_SUFFIX} not supported. NSPH_SUFFIX={"", "_neuralsph"}"
    exit
fi

# Runs:
# nohup bash scripts/rollout.sh 10 "_neuralsph" >> logs/rollouts_5_10_neuralsph.log 2>&1 &
# nohup bash scripts/rollout.sh 10 "" >> logs/rollouts_5_10.log 2>&1 &
# nohup bash scripts/rollout.sh 1 "_neuralsph" >> logs/rollouts_50_1_neuralsph.log 2>&1 &
# nohup bash scripts/rollout.sh 1 "" >> logs/rollouts_50_1.log 2>&1 &

# equivalently:
# sbatch -J rollouts_50_1 scripts/rollout.sh 1 ""
