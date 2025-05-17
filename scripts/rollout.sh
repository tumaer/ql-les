#!/bin/bash
#SBATCH --job-name=rollout
#SBATCH --output=logs/slogs/%x.log
#SBATCH --error=logs/slogs/%x.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

if [ "$#" -ne 2 ]; then
  echo "Wrong number of arguments. Currently $#, should be 2"
  echo "Usage: sbatch scripts/rollout.sh <EVERY_N> <NSPH_SUFFIX>"
  echo "Currently supported: EVERY_N={1, 10}, NSPH_SUFFIX={'', '_neuralsph'}"
  exit 1
fi

EVERY_N=$1
NSPH_SUFFIX=$2

if [ "$EVERY_N" -eq 1 ]; then # every1
    rlt_len=200 # 1000
    ckpts=(
        # "2025-02-24_01-35-42"  # simple
        # "2025-02-24_01-36-13"  # tvf
        # "2025-02-24_02-06-00"  # simple_u
        # "2025-02-24_02-43-05"  # simple_rlx
        # "2025-05-05_04-52-59"  # GINO
        # "2025-05-05_02-40-18"  # InterpFNO
    )
elif [ "$EVERY_N" -eq 10 ]; then # every10
    rlt_len=20 # 100
    ckpts=(
        "2025-02-24_02-54-54"  # simple
        "2025-02-24_02-55-09"  # tvf
        "2025-02-24_02-55-18"  # simple_u
        "2025-02-24_02-55-24"  # simple_rlx
    )
else
    echo "EVERY_N=${EVERY_N} not supported"
    exit
fi

run_base() {
    python src/eval.py \
        ckpt_path="logs/train/runs/${ckpt}/checkpoints/last.ckpt" \
        vars.num_rollout_steps=${rlt_len} \
        model.visualize.vis_test.rollout_dir="logs/train/runs/${ckpt}/rlt/${rlt_len}${NSPH_SUFFIX}" \
        trainer.limit_test_batches=1 \
        model.visualize.vis_test.out_type=vtk \
        logger.wandb.offline=True "$@"
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
    echo "NSPH_SUFFIX=${NSPH_SUFFIX} not supported. NSPH_SUFFIX={\"\", \"_neuralsph\"}"
    exit
fi

# Runs:
# nohup bash scripts/rollout.sh 10 "_neuralsph" >> logs/rollouts_5_10_neuralsph.log 2>&1 &
# nohup bash scripts/rollout.sh 10 "" >> logs/rollouts_5_10.log 2>&1 &
# nohup bash scripts/rollout.sh 1 "_neuralsph" >> logs/rollouts_50_1_neuralsph.log 2>&1 &
# nohup bash scripts/rollout.sh 1 "" >> logs/rollouts_50_1.log 2>&1 &

# equivalently:
# sbatch -J rollouts_200_1 scripts/rollout.sh 1 ""
# sbatch -J rollouts_20_10 scripts/rollout.sh 10 ""
