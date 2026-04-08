#!/bin/bash
# 小规模快速验证：2 个 epoch，每 epoch 100 步，用于检查依赖和流程是否正常
set -e
cd /app
python -m diffusion.trainer \
  --env=walker2d-medium-replay-v2 \
  --n_epochs=2 \
  --n_train_step_per_epoch=100 \
  --eval_period=1 \
  --eval_n_trajs=2 \
  --logging.output_dir=./experiment_output \
  --algo_cfg.loss_type=TD3
