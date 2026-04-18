#!/usr/bin/env bash
# 使用中证1000 Top50 离线 CSV 跑 multimodal 消融（与 run_ablation_multimodal.sh 相同逻辑，仅换数据路径）

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TRAIN_CSV="${TRAIN_CSV:-data_thesis/csi1000_top50_offline_train.csv}"
export EVAL_CSV="${EVAL_CSV:-data_thesis/csi1000_top50_offline_val.csv}"

if [[ ! -f "$TRAIN_CSV" ]]; then
  echo "缺少 $TRAIN_CSV ，请先运行:"
  echo "  python data_thesis/get_csi1000_top50_by_weight.py"
  echo "  python data_thesis/preprocess_csi1000_top50_offline_dataset.py"
  exit 1
fi

exec bash "$ROOT/scripts/run_ablation_multimodal.sh"
