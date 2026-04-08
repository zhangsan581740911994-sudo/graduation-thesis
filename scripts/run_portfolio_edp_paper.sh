#!/usr/bin/env bash
# AutoDL / Linux：portfolio + EDP 毕设主实验（默认不存 model，只落盘 CSV/日志）。
# 规模：约 n_epochs × n_train_step_per_epoch 次梯度步（默认 1000×1000，与 EDP/D4RL 常见设置同量级）。
# 用法：bash scripts/run_portfolio_edp_paper.sh
# 覆盖示例：N_EPOCHS=500 EVAL_PERIOD=10 bash scripts/run_portfolio_edp_paper.sh
# 若 JAX 用 pip CUDA：请先 source venv，且 activate 里已配置 LD_LIBRARY_PATH。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT"
export WANDB_DISABLED=1
# 需要 float64 时可打开：export JAX_ENABLE_X64=1

# 每次运行单独目录，避免覆盖（时间戳）
STAMP="$(date +%Y_%m_%d_%H_%M_%S)"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs_portfolio_edp}"
RUN_DIR="$OUT_ROOT/run_${STAMP}"
mkdir -p "$RUN_DIR"

# 可选：export SAVE_MODEL=1 则保存 model_*.pkl（体积大）；默认不存，论文用 CSV 即可
SAVE_FLAGS=(--nosave_model)
if [[ "${SAVE_MODEL:-0}" == "1" ]]; then SAVE_FLAGS=(--save_model); fi

echo "指标 CSV：在 \$RUN_DIR 下会再出现一层 default_时间戳_s_seed--uuid/，其中有 progress.csv"
echo "本次终端完整输出：$RUN_DIR/console.log"
echo ""

python -m diffusion.trainer \
  --dataset=portfolio \
  --env=portfolio-hs300-top20 \
  --logging.output_dir="$RUN_DIR" \
  --logging.notes="thesis portfolio EDP TD3 ${STAMP}" \
  --seed="${SEED:-42}" \
  --behavior_seed="${BEHAVIOR_SEED:-42}" \
  --n_epochs="${N_EPOCHS:-1000}" \
  --n_train_step_per_epoch="${N_STEPS:-1000}" \
  --eval_period="${EVAL_PERIOD:-20}" \
  --eval_n_trajs="${EVAL_N_TRAJS:-10}" \
  --batch_size="${BATCH_SIZE:-256}" \
  --clip_action=0.999 \
  --sample_method=ddpm \
  --algo_cfg.loss_type=TD3 \
  --algo_cfg.discount=0.99 \
  --algo_cfg.nstep=1 \
  --algo_cfg.lr=0.0003 \
  --algo_cfg.tau=0.005 \
  --algo_cfg.num_timesteps=100 \
  --algo_cfg.guide_coef=1.0 \
  --algo_cfg.diff_coef=1.0 \
  --algo_cfg.alpha=2.0 \
  --algo_cfg.use_pred_astart=true \
  "${SAVE_FLAGS[@]}" \
  "$@" \
  2>&1 | tee "$RUN_DIR/console.log"

echo ""
echo "完成。请收集："
echo "  1) $RUN_DIR/console.log  — 终端完整记录"
echo "  2) $RUN_DIR/*/progress.csv — 每 epoch 指标（Excel/Origin 画图）"
echo "  3) $RUN_DIR/*/variant.json — 超参与配置"
echo "  4) $RUN_DIR/*/debug.log — 文本日志"
if [[ "${SAVE_MODEL:-0}" == "1" ]]; then
  echo "  5) $RUN_DIR/*/model_*.pkl 与 model_final.pkl（若 SAVE_MODEL=1）"
fi
