#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Action Approximation Ablation Runner

目标：比较 use_pred_astart=True / False 在
- 训练时间
- D4RL normalized return
方面的差异。

用法（在项目根目录执行终端命令）：
  cd e:\computer_learning\projects\edp
  python experiments\run_action_approx_ablation.py

注意：
- 这个脚本不会修改任何现有源码，只是通过子进程调用 `python -m diffusion.trainer`；
- 某些命令行参数名（如 --algo_cfg.use_pred_astart）需要与你本地的 trainer 定义对齐，
  如果报 unknown flag，可以根据实际代码稍作调整或先删掉该 flag。
"""

import csv
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import List


# 当前文件在 experiments/ 目录下，项目根目录是其上一级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class RunConfig:
  use_pred_astart: bool
  seed: int
  env: str = "walker2d-medium-replay-v2"
  loss_type: str = "TD3"
  epochs: int = 500          # 为了快速对比，这里先用 500 epochs
  steps_per_epoch: int = 1000
  output_dir: str = "./exp_ap_ablation"


@dataclass
class RunResult:
  use_pred_astart: bool
  seed: int
  train_time_sec: float
  best_normalized_return: float
  avg_normalized_return: float


def run_single_experiment(cfg: RunConfig) -> RunResult:
  """
  通过子进程调用 `python -m diffusion.trainer`，返回训练时间与评估结果。
  假设 trainer 会把 progress.csv / tabular 日志输出到 cfg.output_dir/{exp_name}/ 下。
  """
  exp_name = f"ap_{int(cfg.use_pred_astart)}_seed_{cfg.seed}"
  exp_dir = os.path.join(PROJECT_ROOT, cfg.output_dir, exp_name)
  os.makedirs(exp_dir, exist_ok=True)

  # 构造命令行。
  # 如果出现 unknown flag，可以先注释掉相应行，再根据实际代码调整。
  cmd = [
    "python", "-m", "diffusion.trainer",
    "--env", cfg.env,
    "--algo_cfg.loss_type", cfg.loss_type,
    "--logging.output_dir", exp_dir,
    "--trainer.n_epochs", str(cfg.epochs),
    "--trainer.n_train_step_per_epoch", str(cfg.steps_per_epoch),
    "--seed", str(cfg.seed),
  ]

  # 可选：如果工程里已经把 use_pred_astart 暴露为 algo_cfg 字段，可以打开这行
  # cmd.extend(["--algo_cfg.use_pred_astart", str(cfg.use_pred_astart)])

  print(f"Running: {' '.join(cmd)}")
  start = time.time()
  subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
  end = time.time()
  train_time = end - start

  # 从日志中读取 best_normalized_return / average_normalizd_return
  progress_path = os.path.join(exp_dir, "progress.csv")
  if not os.path.exists(progress_path):
    raise FileNotFoundError(f"progress.csv not found in {exp_dir}")

  best_norm = float("-inf")
  last_avg = float("nan")

  with open(progress_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      if not row:
        continue
      for key, val in row.items():
        if not val:
          continue
        if "best_normalized_return" in key:
          try:
            best_norm = max(best_norm, float(val))
          except ValueError:
            pass
        if "average_normalizd_return" in key:
          try:
            last_avg = float(val)
          except ValueError:
            pass

  return RunResult(
    use_pred_astart=cfg.use_pred_astart,
    seed=cfg.seed,
    train_time_sec=train_time,
    best_normalized_return=best_norm,
    avg_normalized_return=last_avg,
  )


def main():
  seeds = [0, 1]  # 可以先跑两个 seed 试试，确认流程没问题再加
  configs: List[RunConfig] = []
  for s in seeds:
    configs.append(RunConfig(use_pred_astart=True, seed=s))
    configs.append(RunConfig(use_pred_astart=False, seed=s))

  results: List[RunResult] = []
  for cfg in configs:
    res = run_single_experiment(cfg)
    print("Single run result:", asdict(res))
    results.append(res)

  # 汇总统计
  def summarize(use_pred_astart: bool):
    subset = [r for r in results if r.use_pred_astart == use_pred_astart]
    if not subset:
      return None
    import numpy as np
    t = np.array([r.train_time_sec for r in subset], dtype=float)
    b = np.array([r.best_normalized_return for r in subset], dtype=float)
    a = np.array([r.avg_normalized_return for r in subset], dtype=float)
    return {
      "use_pred_astart": use_pred_astart,
      "n_runs": len(subset),
      "train_time_sec_mean": float(t.mean()),
      "train_time_sec_std": float(t.std()),
      "best_norm_mean": float(b.mean()),
      "best_norm_std": float(b.std()),
      "avg_norm_mean": float(a.mean()),
      "avg_norm_std": float(a.std()),
    }

  summary_true = summarize(True)
  summary_false = summarize(False)

  print("\n=== Action Approximation Ablation Summary ===")
  for summ in [summary_true, summary_false]:
    if summ is None:
      continue
    print(summ)


if __name__ == "__main__":
  main()

