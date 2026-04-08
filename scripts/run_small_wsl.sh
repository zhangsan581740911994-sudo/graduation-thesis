#!/bin/bash
# 在 WSL 下跑小规模训练：先设置 MuJoCo 环境变量再启动
set -e
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
python -m diffusion.trainer --env 'walker2d-medium-replay-v2' --n_epochs=2 --n_train_step_per_epoch=100 --eval_period=1 --eval_n_trajs=2 --logging.output_dir './experiment_output' --algo_cfg.loss_type=TD3
