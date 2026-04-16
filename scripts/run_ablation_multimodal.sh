#!/usr/bin/env bash
# 方案 A：多峰 + 低覆盖 离线行为数据上的 TD3 vs EDP 消融
# 需设置：PORTFOLIO_BEHAVIOR_MODE=multimodal（见 trading_env/portfolio_behavior.py）
# 数据：默认 top50；可改 TRAIN_CSV / EVAL_CSV

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export WANDB_DISABLED=1

# 与 multimodal 互斥，避免误用旧噪声混合
unset PORTFOLIO_NOISE_LEVEL || true

export PORTFOLIO_BEHAVIOR_MODE=multimodal
export PORTFOLIO_MULTIMODAL_CONSERVATIVE_FRAC="${PORTFOLIO_MULTIMODAL_CONSERVATIVE_FRAC:-0.7}"
export PORTFOLIO_MULTIMODAL_CONSERVATIVE_STD="${PORTFOLIO_MULTIMODAL_CONSERVATIVE_STD:-0.12}"
export PORTFOLIO_MULTIMODAL_EXTREME_K_MAX="${PORTFOLIO_MULTIMODAL_EXTREME_K_MAX:-3}"

TRAIN_CSV="${TRAIN_CSV:-data_thesis/hs300_top50_offline_train.csv}"
EVAL_CSV="${EVAL_CSV:-data_thesis/hs300_top50_offline_val.csv}"

STAMP="$(date +%Y_%m_%d_%H_%M_%S)"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs_portfolio_ablation_multimodal/$STAMP}"
mkdir -p "$OUT_ROOT"

echo "================================================================="
echo "消融：multimodal 行为数据 | TD3 vs EDP"
echo "OUT_ROOT=$OUT_ROOT"
echo "TRAIN_CSV=$TRAIN_CSV"
echo "EVAL_CSV=$EVAL_CSV"
echo "================================================================="

run_one() {
  local kind="$1"
  local seed="$2"
  local tag="multimodal_${kind}_s${seed}_${STAMP}"
  local run_dir="$OUT_ROOT/$tag"
  local log_file="$OUT_ROOT/${tag}.log"

  echo ">>> $kind seed=$seed"
  if [[ "$kind" == "td3" ]]; then
    python -m diffusion.trainer \
      --dataset=portfolio \
      --env=portfolio-hs300-top20 \
      --portfolio_train_csv="$TRAIN_CSV" \
      --portfolio_eval_csv="$EVAL_CSV" \
      --seed="$seed" \
      --behavior_seed="$seed" \
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
      --logging.output_dir="$run_dir" 2>&1 | tee "$log_file"
  else
    python -m diffusion.trainer \
      --dataset=portfolio \
      --env=portfolio-hs300-top20 \
      --portfolio_train_csv="$TRAIN_CSV" \
      --portfolio_eval_csv="$EVAL_CSV" \
      --seed="$seed" \
      --behavior_seed="$seed" \
      --n_epochs=1000 \
      --n_train_step_per_epoch=1200 \
      --batch_size=256 \
      --eval_period=10 \
      --eval_n_trajs=10 \
      --sample_method=ddpm \
      --algo_cfg.loss_type=TD3 \
      --algo_cfg.lr=4e-5 \
      --save_model=false \
      --logging.output_dir="$run_dir" 2>&1 | tee "$log_file"
  fi
}

SEEDS="${SEEDS:-42 43 44 45 46}"
for s in $SEEDS; do
  run_one td3 "$s"
  run_one edp "$s"
done

echo "完成。请对 progress.csv 汇总 best_normalized_return 等指标。"
