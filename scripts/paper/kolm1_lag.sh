#!/bin/bash
# nohup bash scripts/paper/lag.sh > lag_kolm2d_every1_20000_nsph.log 2>&1 &

#### Train
# python src/train.py experiment=kolm_every_1/lag.yaml +logger.wandb.name=lag
# python src/train.py experiment=kolm_every_1/v2u_gnn.yaml +logger.wandb.name=lag
# or
# sbatch -J v2u1 scripts/slurm_train.sh "experiment=kolm_every_1/v2u_gnn.yaml +logger.wandb.name=v2u1"

#### Inference
run_basic() {
    python src/eval.py ckpt_path=logs/train/runs/2025-06-02_03-21-52/checkpoints/last.ckpt \
    model.visualize.vis_test.rollout_dir=logs/train/runs/2025-06-02_03-21-52/rlt/"$1" \
    logger.wandb.offline=True model.visualize.vis_test.out_type=pkl \
    model.visualize.vis_test.animate_first_n=0 model.v2u_solver="$4" \
    trainer.limit_test_batches=5 data.only_beginning=True \
    model.net.v2u_gnn.ckpt_path="logs/train/runs/2025-05-10_05-09-50/checkpoints/last.ckpt" \
    model.neuralsph.test.num_steps="$2" vars.num_rollout_steps="$3" "${@:5}"
}

# run_basic test 0 11 same

#### 1001
# # run_basic RLT_DIR_NAME NUM_NSPH_STEPS NUM_ROLLOUT_STEPS [ADDITIONAL_ARGS]
# run_basic 1001_nsph2 2 1001 same
# run_basic 1001_nsph1 1 1001 same
# run_basic 1001_nsph05 1 1001 same model.neuralsph.test.dt_factor=0.5
# run_basic 1001_nsph02 1 1001 same model.neuralsph.test.dt_factor=0.2
# run_basic 1001_gnn 0 1001 gnn
# run_basic 1001_gnn_nsph1 1 1001 gnn
# run_basic 1001_nsph1tvf100 1 1001 same model.neuralsph.cfg_every=100 \
#     model.neuralsph.test.cfg.is_tvf=True model.neuralsph.test.cfg.tvf_factor=1 \

#### 101
# run_basic 101 0 101 same
# run_basic 101_nsph1 1 101 same
# run_basic 101_gnn 0 101 gnn
# run_basic 101_gnn_nsph1 1 101 gnn
# run_basic 101_nsph1tvf01 1 101 same model.neuralsph.test.is_tvf=True
# run_basic 101_nsph1tvf1 1 101 same model.neuralsph.test.is_tvf=True

#### 5001
# run_basic 5001_nsph1 1 5001 same
# run_basic 5001_gnn_nsph1 1 5001 gnn
# run_basic 5001_nsph2 2 5001 same
# run_basic 5001_nsph1tvf 1 5001 same model.neuralsph.test.is_tvf=True
# run_basic 5001_nsph1tvf001 1 5001 same model.neuralsph.test.is_tvf=True
# run_basic 5001_nsph1tvf0005 1 5001 same model.neuralsph.test.is_tvf=True
# run_basic 5001_nsph1tvf0002 1 5001 same model.neuralsph.test.is_tvf=True
# run_basic 5001_nsph1tvf0001 1 5001 same model.neuralsph.test.is_tvf=True
# run_basic 5001_nsph1tvf100 1 5001 same model.neuralsph.cfg_every=100 \
#     model.neuralsph.test.cfg.is_tvf=True model.neuralsph.test.cfg.tvf_factor=1
# run_basic 5001_nsph1tvf500 1 5001 same model.neuralsph.cfg_every=500 \
#     model.neuralsph.test.cfg.is_tvf=True model.neuralsph.test.cfg.tvf_factor=1
# run_basic 5001_nsph1tvf10f01 1 5001 same model.neuralsph.cfg_every=10 \
#     model.neuralsph.test.cfg.is_tvf=True model.neuralsph.test.cfg.tvf_factor=0.1
# run_basic 5001_nsph1f2 1 5001 same model.neuralsph.test.dt_factor=2

#### 20000
# run_basic 20000_nsph1tvf001 1 19999 same model.neuralsph.cfg_every=1 \
#     model.neuralsph.test.cfg.is_tvf=True model.neuralsph.test.cfg.tvf_factor=0.01
# run_basic 20000_nsph1tvf0005 1 19999 same model.neuralsph.cfg_every=1 \
#     model.neuralsph.test.cfg.is_tvf=True model.neuralsph.test.cfg.tvf_factor=0.005
# run_basic 20000_nsph1tvf100 1 19999 same model.neuralsph.cfg_every=100 \
#     model.neuralsph.test.cfg.is_tvf=True model.neuralsph.test.cfg.tvf_factor=1
