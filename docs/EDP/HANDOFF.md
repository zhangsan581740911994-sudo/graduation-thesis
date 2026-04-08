# 毕设接续说明（给新对话 / 自己看）

> 项目根目录：`edp`（本仓库）  
> 主线：**多只股票组合 + 日频 + EDP 离线预训练 → 在线微调与对比实验**（见 `毕业设计项目文档.md`）

---

## 当前进度（已实现）

- **单标的 ETF 冒烟管线**：`trading_env/env.py`、`behavior_policy.py`、`gym_env.py`；`utilities/etf_dataset.py`；`Trainer` 中 `dataset=etf` + `_setup_etf`；依赖见 `requirements-thesis.txt`（Windows 已固定 JAX/Flax/orbax 版本）。
- **毕设总文档**：`docs/毕业设计项目文档.md`（含 §5.3.4 成果比对与可引用文献）。
- **数据**：`data_thesis/` 下曾为单只 510300 ETF 的 train/val/test；**多标的主实验需新数据与新股池**。

---

## 接下来要做（按顺序）

1. **定股票池**：约 20～50 只、筛选规则（如指数成分 + 流动性）、论文里写清楚。
2. **定时间划分**：train / val / test（或 walk-forward），全篇统一。
3. **面板数据**：多标的日频 OHLCV（或收益矩阵），**对齐交易日**、处理缺失/停牌。
4. **多标的环境**：组合权重 \(\mathbf{w}_t\)、组合奖励 \(r_t^{(port)}\)、`gym` 空间维数 = \(N\) 或与参数化一致（文档附录提到的 `portfolio_env` 类待实现）。
5. **行为策略**：输出 \(N\) 维权重（约束与论文一致）。
6. **离线数据集 + Trainer**：在单标 `get_etf_dataset` 思路上扩展维度；评估加 **沪深300 / 等权** 等基准。
7. **baseline**：纯离线 EDP、纯在线 RL、与你的方法对比（见总文档 §5.2、§5.4）。

---

## 新聊天窗口怎么用

1. 打开新对话，第一条消息写：**「请阅读 `@docs/HANDOFF.md` 和 `@docs/毕业设计项目文档.md`，继续多标的组合实验。」**
2. 需要改代码时再 `@` 具体文件（如 `trading_env/`、`diffusion/trainer.py`）。

---

## 常用路径

| 内容 | 路径 |
|------|------|
| 毕设全文与实验计划 | `docs/毕业设计项目文档.md` |
| 本接续条 | `docs/HANDOFF.md` |
| 单标环境 | `trading_env/env.py` |
| 毕设依赖 | `requirements-thesis.txt` |
| ETF 训练命令（单标） | `python -m diffusion.trainer --dataset=etf --env=etf-hs300 ...`；需 `PYTHONPATH=项目根`，`WANDB_DISABLED=1` 可避免 Windows 上 wandb 卡住 |

---

## 注意

- **单只 ETF 代码可保留**作对照或冒烟；**论文主结论**以多标的实验为准。
- 不存在全国统一「Sharpe 达标线」；写作见总文档 **§5.3.4**（相对 baseline + 基准 + 文献数量级讨论）。

---

**最后更新**：2026-04-07
