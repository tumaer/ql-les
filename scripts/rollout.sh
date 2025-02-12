#!/bin/bash
# Usage: bash scripts/rollout.sh <every_n> <nsph_suffix>
# Currently supported: every_n={1,10}, nsph_suffix={"", "_neuralsph"}
# Example:
# nohup bash scripts/rollout.sh 10 "_neuralsph" >> logs/rollouts_5_10_neuralsph.log 2>&1 &
# nohup bash scripts/rollout.sh 10 "" >> logs/rollouts_5_10.log 2>&1 &
# nohup bash scripts/rollout.sh 1 "_neuralsph" >> logs/rollouts_50_1_neuralsph.log 2>&1 &
# nohup bash scripts/rollout.sh 1 "" >> logs/rollouts_50_1.log 2>&1 &

every_n=$1
nsph_suffix=$2

if [ -z "$every_n" ]; then
    echo "Usage: $0 <every_n> <nsph_suffix>"
    exit 1
fi

if [ $every_n -eq 1 ]; then # every1
    rlt_len=50 # 1000
    ckpts=(
        "2025-02-08_02-46-56"  # simple
        "2025-02-08_02-47-01"  # tvf
        "2025-02-08_02-47-06"  # simple_u
        "2025-02-09_06-22-19"  # simple_rlx
    )
elif [ $every_n -eq 10 ]; then # every10
    rlt_len=5 # 100
    ckpts=(
        "2025-02-09_17-04-48"  # simple
        "2025-02-09_17-04-52"  # tvf
        "2025-02-09_17-04-56"  # simple_u
        "2025-02-09_21-33-00"  # simple_rlx
    )
else
    echo "every_n=${every_n} not supported"
    exit
fi

run_base() {
    python src/eval.py experiment="gns_kolm2d_every${every_n}" \
        ckpt_path="logs/paper/${ckpt}/checkpoints/last.ckpt" \
        model.visualize.vis_test.rollout_dir="logs/paper/${ckpt}/rollouts_${rlt_len}${nsph_suffix}" \
        model.visualize.vis_test.out_type=pkl \
        vars.num_rollout_steps=${rlt_len} \
        logger.wandb.offline=True "$@"
        # +trainer.limit_test_batches=1
}

if [ -z "$nsph_suffix" ]; then  # no NeuralSPH
    echo "Running without NeuralSPH"
    for ckpt in "${ckpts[@]}"; do
        run_base
    done
elif [ "$nsph_suffix" == "_neuralsph" ]; then
    echo "Running with NeuralSPH"
    run_base model.neuralsph.test.dt_factor=2 model.neuralsph.test.num_steps=10
fi
