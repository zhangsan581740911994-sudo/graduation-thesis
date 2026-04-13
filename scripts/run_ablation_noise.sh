#!/usr/bin/env bash
# 深度消融实验：混合噪声环境下的 EDP vs 纯 TD3
# 目的：证明在充满噪声、次优交易行为的数据集中，EDP（扩散模型）能比纯 TD3 更好地提取高回报策略。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export WANDB_DISABLED=1

# 开启 80% 的高噪声模式（行为策略中有 80% 概率执行纯随机交易）
export PORTFOLIO_NOISE_LEVEL=0.8

STAMP="$(date +%Y_%m_%d_%H_%M_%S)"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs_portfolio_ablation_noise}"
mkdir -p "$OUT_ROOT"

echo "================================================================="
echo "开始深度消融实验：高噪声环境 (Noise Level = 80%)"
echo "将分别运行 纯 TD3 和 EDP 模型，比较在嘈杂数据下的鲁棒性"
echo "================================================================="

# 1. 运行纯 TD3 (diff_coef=0.0)
echo ">>> [1/2] 正在运行：纯 TD3 (高噪声环境) ..."
for s in 42; do
  RUN_DIR="$OUT_ROOT/pure_td3_noise80_s${s}_${STAMP}"
  python -m diffusion.trainer \
    --dataset=portfolio \
    --env=portfolio-hs300-top20 \
    --portfolio_train_csv=data_thesis/hs300_top20_offline_train.csv \
    --portfolio_eval_csv=data_thesis/hs300_top20_offline_val.csv \
    --seed=$s \
    --n_epochs=1000 \
    --n_train_step_per_epoch=1200 \
    --batch_size=256 \
    --eval_period=10 \
    --eval_n_trajs=10 \
    --sample_method=ddpm \
    --algo_cfg.loss_type=TD3 \
    --algo_cfg.lr=4e-5 \
    --algo_cfg.diff_coef=0.0 \
    --save_model=false \
    --logging.output_dir="$RUN_DIR" | tee "$RUN_DIR.log"
  echo "纯 TD3 (seed=$s) 运行完成。日志：$RUN_DIR.log"
done

# 2. 运行 EDP (默认 diff_coef)
echo ">>> [2/2] 正在运行：EDP 扩散模型 (高噪声环境) ..."
for s in 42; do
  RUN_DIR="$OUT_ROOT/edp_noise80_s${s}_${STAMP}"
  python -m diffusion.trainer \
    --dataset=portfolio \
    --env=portfolio-hs300-top20 \
    --portfolio_train_csv=data_thesis/hs300_top20_offline_train.csv \
    --portfolio_eval_csv=data_thesis/hs300_top20_offline_val.csv \
    --seed=$s \
    --n_epochs=1000 \
    --n_train_step_per_epoch=1200 \
    --batch_size=256 \
    --eval_period=10 \
    --eval_n_trajs=10 \
    --sample_method=ddpm \
    --algo_cfg.loss_type=TD3 \
    --algo_cfg.lr=4e-5 \
    --save_model=false \
    --logging.output_dir="$RUN_DIR" | tee "$RUN_DIR.log"
  echo "EDP (seed=$s) 运行完成。日志：$RUN_DIR.log"
done

echo "================================================================="
echo "深度消融实验全部完成！"
echo "你可以使用汇总脚本查看 $OUT_ROOT 目录下的结果对比。"
echo "================================================================="
