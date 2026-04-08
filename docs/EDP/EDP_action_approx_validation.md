## 把 Action Approximation 做对：既快又不把策略搞坏

> 视角：**我是这篇工作/项目的研发者**，要证明 action approximation（AP）这个设计既带来了训练加速，又不会明显破坏策略性能。

---

### 一、问题与困难

- **目标**：用单步预测的 \(\hat{x}_0\)（`pred_astart`）近似「完整扩散链采样得到的动作」，在训练时直接拿它算 guide loss，从而省掉整条采样链的计算开销。
- **困难点**：
  - 如果近似得太粗糙，**策略更新方向可能被系统性偏移**，尤其是在复杂环境（Antmaze / Adroit）上，可能表现为：收敛更慢、最终 normalized return 下降。
  - AP 的收益主要体现在**训练速度**上，但我们必须确保**性能（最终策略质量）不显著下降**——否则只是“快但差”，无法成为论文级贡献。
  - 在代码层面，AP 与很多模块交织在一起（`DiffusionPolicy`、`GaussianDiffusion`、`DiffusionQL` 的 loss），要设计 ablation 时 **尽量不破坏原有结构**。

**用一句话概括这个困难**：  

> 既要把 action approximation「插」进原来的 DiffusionQL 训练框架里，拿到明显的训练加速；又要用系统性的实验证明在不同环境上，策略性能没有明显被搞坏。

---

### 二、解决方案设计（高层思路）

我们希望在**不修改现有核心代码**的前提下，验证两件事：

1. **训练效率**：`use_pred_astart=True` 相比 `False`，在相同 hardware 下每秒训练步数明显提升；
2. **策略性能**：在相同训练总步数（或墙钟时间）下，两种设置的 **normalized return** 没有显著差距，甚至 AP 版本不劣。

为此，可以做一个**外部脚本式的 ablation**：

- 利用现有的命令行接口 `python -m diffusion.trainer ...`，通过 **配置/flags 控制** `use_pred_astart`；
- 不改 `diffusion/` 下的核心实现，仅在**新脚本中**：
  - 循环跑两组实验（有 AP / 无 AP）；
  - 读取每次实验输出目录下的日志（`progress.csv` 或 W&B 日志）中的 `best_normalized_return` / `average_normalizd_return`；
  - 记录运行时间，计算「性能–效率」对比；
  - 将结果汇总成一个简单的表格/打印输出。

这样做的好处：

- **不侵入原有代码**：只是在外面多了一个 “runner + 评估脚本”；
- **可重复**：任何人都可以在同样环境下跑同样脚本，得到相同的对比；
+- **方便扩展**：以后想对比别的超参（如 DPM 步数、EAS 样本数），只需在此脚本里多加几组配置即可。

---

### 三、实验设计细节

#### 3.1 对比维度

以 `walker2d-medium-replay-v2` 为例（也可以扩展到其他 D4RL 任务）：

- **有 AP**：`use_pred_astart=True`（项目默认做法，对应 EDP 论文中的设定）；
- **无 AP**：`use_pred_astart=False`（训练时真正跑完整反向链取动作算 guide loss，慢但更精确）。

对每个设定：

- 跑若干个 seed（例如 3～5 个）以减小方差；
- 记录：
  - **训练用时**（秒）
  - **best_normalized_return**
  - **average_normalizd_return**（或 RAT 风格的后 N 次平均）

最后输出形如：

| setting           | seeds | train_time (s) | best_norm_return | avg_norm_return |
|-------------------|-------|----------------|------------------|-----------------|
| use_pred_astart T | 3     | 12,345         | 0.93             | 0.91            |
| use_pred_astart F | 3     | 25,678         | 0.94             | 0.90            |

这样就能非常直观地回答：

- AP **大约加速了多少倍**；
- AP 对最终策略性能的影响是否可以接受（或者有时甚至更好）。

---

### 四、示例代码：外部 ablation 脚本（不改原有源码）

下面是一个**可行的 Python 脚本示例**，放在项目外部或新建例如 `experiments/run_action_approx_ablation.py`，不修改现有任何 `.py` 文件，仅通过命令行参数控制配置，并解析日志：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Action Approximation Ablation Runner

目标：比较 use_pred_astart=True / False 在
- 训练时间
- D4RL normalized return
方面的差异。

用法（在项目根目录执行）：
  python experiments/run_action_approx_ablation.py
"""

import csv
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import List


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class RunConfig:
  use_pred_astart: bool
  seed: int
  env: str = "walker2d-medium-replay-v2"
  loss_type: str = "TD3"
  epochs: int = 500          # 为了快速对比，这里可以先用 500，再视需要调大
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
  假设 trainer 会把 progress.csv / tabular 日志输出到 cfg.output_dir/{exp_id}/ 下。
  """
  # 为了区分不同设定/seed，拼一个简单的实验名
  exp_name = f"ap_{int(cfg.use_pred_astart)}_seed_{cfg.seed}"
  exp_dir = os.path.join(cfg.output_dir, exp_name)
  os.makedirs(exp_dir, exist_ok=True)

  # 构造命令行；这里假设 trainer 支持以下 flags：
  # --env, --algo_cfg.loss_type, --logging.output_dir,
  # --trainer.n_epochs, --trainer.n_train_step_per_epoch,
  # 以及一个额外的覆盖方式让 use_pred_astart 生效（例如通过 algo_cfg 或 config 文件）。
  #
  # 如果现有 trainer 不能直接从命令行改 use_pred_astart，可以：
  # - 在 configs/hps.py 里为 “ap ablation” 增加一个配置；
  # - 或者用环境变量/临时 config 文件的方式让 DiffusionQL 读取。
  #
  # 这里先给出一个“原则上可行”的写法，具体 flag 名可以根据项目实际调整。

  cmd = [
    "python", "-m", "diffusion.trainer",
    "--env", cfg.env,
    "--algo_cfg.loss_type", cfg.loss_type,
    "--algo_cfg.use_pred_astart", str(cfg.use_pred_astart),
    "--logging.output_dir", exp_dir,
    "--trainer.n_epochs", str(cfg.epochs),
    "--trainer.n_train_step_per_epoch", str(cfg.steps_per_epoch),
    "--seed", str(cfg.seed),
  ]

  print(f"Running: {' '.join(cmd)}")
  start = time.time()
  subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
  end = time.time()
  train_time = end - start

  # 从日志中读取 best_normalized_return / average_normalizd_return
  # 假设 trainer 使用 viskit 格式，在 exp_dir 下有 progress.csv。
  progress_path = os.path.join(exp_dir, "progress.csv")
  if not os.path.exists(progress_path):
    raise FileNotFoundError(f"progress.csv not found in {exp_dir}")

  best_norm = float("-inf")
  last_avg = float("nan")

  with open(progress_path, "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
      # 列名需要和 trainer 的输出一致，这里给出一种常见命名：
      # "best_normalized_return/dpm" 或 "best_normalized_return"
      # 以及 "average_normalizd_return/dpm" 等。
      # 可以根据实际列名适当调整解析逻辑。
      for key, val in row.items():
        if not val:
          continue
        if "best_normalized_return" in key:
          best_norm = max(best_norm, float(val))
        if "average_normalizd_return" in key:
          last_avg = float(val)

  return RunResult(
    use_pred_astart=cfg.use_pred_astart,
    seed=cfg.seed,
    train_time_sec=train_time,
    best_normalized_return=best_norm,
    avg_normalized_return=last_avg,
  )


def main():
  seeds = [0, 1, 2]
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
```

> 说明：
> - 这个脚本**不修改**项目中的任何现有代码，只是借助已有的 `diffusion.trainer` 作为黑盒；
> - 唯一需要对齐的是一些命令行参数名（如 `--algo_cfg.use_pred_astart`、`--trainer.n_epochs` 等）和日志列名（`best_normalized_return`、`average_normalizd_return`），可以根据实际代码稍作调整；
> - 一旦能跑通，就可以比较系统地回答：“action approximation 在这个框架下，确实带来了 X 倍的训练加速，同时在若干任务上的 normalized return 没有显著退化”。

---

### 五、小结：如何在复试中讲这个困难

如果从「我是这个项目的研发者」的角度，在复试里可以这样讲：

- **难点**：
  - 我们提出了 action approximation，希望在训练期只用单步 \(\hat{x}_0\) 就完成 RL 更新，极大降低计算量；
  - 真正的挑战是要**证明**：在各种 D4RL 任务上，这个近似不会把策略性能明显搞坏。

- **解决思路**：
  - 不直接改核心代码，而是额外写了一个 ablation 脚本，系统对比 `use_pred_astart=True / False` 在训练时间和 normalized return 上的差异；
  - 用多 seed、多个环境的结果支撑“既快、性能又不差”的结论。

- **价值**：
  - 这样一来，action approximation 就不再只是一个“看起来很合理的想法”，而是有完整实验闭环的设计：**算法动机 + 理论合理性 + 成本–效果对比实验**。

