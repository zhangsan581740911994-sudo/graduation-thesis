# Day 6：评估与 EAS / 多种采样方式 + 整体串联与复试讲解准备

> **目标**：① 理解「评估时多种 act_method」与论文结论的关系，能说清 EAS、ensemble、DPM vs DDPM 在评估里的角色；② 能从「问题 → 方法 → 代码 → 实验」一口气讲下来，并在 3–5 分钟内完整介绍整个项目。  
> **对应理论**：评估协议、EAS（多采动作用 Q 选）、多种采样方式对比（DDPM/DPM/ensemble）；整体串联对应复试梳理一、二、三、四、五。  
> **说明**：本日整合原计划中的 Day 5.5（评估与 EAS）与 Day 6（整体串联 + 讲稿），先搞清「评估在做什么」，再整理成属于你自己的复试讲稿。

---

### Day 6 提问自测（先尝试自己回答）

> 建议：真正答题时只看这部分，不要往下翻到「参考答案」。

**Part A：评估与 EAS**

1. 训练阶段和评估阶段分别在哪里「生成动作」？训练时有没有跑完整反向链？
2. 「多种采样方式」在代码里是怎么实现的？是多种方式组合成一次评估，还是每种方式分别跑轨迹、分别记一列指标？
3. EAS（Energy-based Action Selection）是什么？论文为什么写 “All results will be reported based on EAS”？
4. `SamplerPolicy` 里 `ddpm_act`、`dpm_act`、`ensemble_act`、`ddpmensemble_act`、`dpmensemble_act` 各做什么？哪几个是「单采样」、哪几个是「多采 + Q 选」？
5. DPM 和 DDPM 在评估里的区别是什么？EDP 的「生成效率」改进体现在哪里？
6. `act_methods`、`post`、`recent_returns` 在 `trainer.py` 的评估循环里分别表示什么？

**Part B：整体串联与讲稿**

7. 用 1 分钟说明：EDP 要解决什么问题、三方面改进（训练效率、生成效率、算法兼容性）分别是什么？
8. 从 `python -m diffusion.trainer ...` 到输出 normalized return，你会按什么顺序讲（至少 5 个关键节点）？
9. 讲「方法核心」时，你会怎么概括：扩散策略、双损失、TD3/CRR/IQL 引导、EAS 评估？
10. 最终成果是通过什么展示的？训练速度 25× 是「成果」还是「技术贡献」？

---

## 零、理论基础：评估阶段才做「生成」

- **训练阶段**：只做扩散的 **training_losses**（随机 t、加噪、噪声 MSE），以及用 **pred_astart** 算 guide loss；**不跑完整反向链**，因此没有「从 x_T 去噪到 x_0」的生成。
- **评估阶段**：要在环境中滚轨迹，每一步都需要**给定 obs 输出一个动作**；此时才用 **p_sample_loop** 或 **DPM_Solver** 从噪声完整去噪得到动作，即「生成」发生在评估/部署时。
- **多种采样方式**：对每种方式（如 ddpm、dpm、ddpmensemble、dpmensemble）**分别**用当前策略滚若干条轨迹，**分别**记一列指标（如 `average_normalizd_return_dpm`、`average_normalizd_return_dpmensemble`），不是把多种方式组合成一次评估。
- **EAS**：先采 N 个候选动作，再用 Q 网络给每个动作打分，按权重 ∝ e^Q 或选 Q 最大的一个；等价于从改进策略 p(a|s) ∝ e^Q π_θ(a|s) 采样。论文 4.5 节写明所有报告结果基于 EAS。

---

## 一、评估部分：多种 act_method 与 EAS

### 1.1 代码位置与角色

| 内容 | 位置 | 作用 |
|------|------|------|
| **评估循环** | `diffusion/trainer.py`，`train()` 内约 303–340 行 | 按 `eval_period` 触发；遍历 `act_methods`，对每个 method 设 `_sampler_policy.act_method`，调 `_eval_sampler.sample(...)` 滚轨迹，算 return / normalized return，写入 metrics（带 `post = "_" + method` 后缀）。 |
| **SamplerPolicy** | `diffusion/trainer.py`，约 113–239 行 | 评估时「obs → action」的封装；根据 `act_method` 在 `__call__` 里分发到 `ddpm_act` / `dpm_act` / `ddim_act` / `ensemble_act` / `ddpmensemble_act` / `dpmensemble_act`。 |
| **act_method 来源** | `train()` 开头：`act_methods = self._cfgs.act_method.split('-')` | 例如 `act_method='dpm-dpmensemble'` → `['dpm', 'dpmensemble']`，即对 dpm 和 dpmensemble 各跑一列指标。 |

### 1.2 几种 act 的含义

| act_method | 含义 | 是否 EAS |
|------------|------|----------|
| **ddpm_act** | 用 DDPM 从噪声去噪，采 **1 个**动作 | 否 |
| **dpm_act** | 用 DPM-Solver 少步去噪，采 **1 个**动作 | 否 |
| **ddim_act** | 用 DDIM 去噪，采 **1 个**动作 | 否 |
| **ensemble_act** | 采 **num_samples 个**动作，用双 Q 的 min(Q1,Q2) 打分，按 Q 做一次 categorical 抽样选一个 | 是（EAS） |
| **ddpmensemble_act** | 用 DDPM 采多个动作，再用 Q 选一个 | 是（EAS） |
| **dpmensemble_act** | 用 DPM 采多个动作，再用 Q 选一个 | 是（EAS） |

- **EAS** 在代码里对应带 `ensemble` 的几种 act；论文所有报告结果用 EAS，即评估时用「多采 + Q 选」得到更好、更稳的 normalized return。
- **DPM vs DDPM**：同一模型可用 DDPM（满 T 步）或 DPM（少步 ODE）采样；评估时可对比两列指标，体现 EDP 的「生成加速」——少步即可达到相近性能。

### 1.3 评估循环中的变量

- **act_methods**：从配置 `act_method` 按 `-` 拆成的列表，决定对哪几种采样方式分别评估。
- **post**：指标后缀。若只评估一种方式则 `post=""`（如 `average_normalizd_return`）；若多种则 `post="_" + method`（如 `average_normalizd_return_dpm`、`average_normalizd_return_dpmensemble`）。
- **recent_returns**：每个 method 对应一个 deque，记录最近若干次该方式的评估 return，用于日志或 early stop 等。

### 1.4 与复试梳理的对应

- 详细表述见 `docs/EDP_复试梳理与演讲稿.md` 的 **3.5 评估部分：多种采样方式与 EAS**、**常见问题 8（EAS/ensemble）**。
- 论文 4.5 节（Controlled Sampling from Diffusion Policies）：EAS 的定义与 “All results based on EAS”。

---

## 二、整体串联与复试讲稿准备

### 2.1 复盘阅读建议

- 快速再浏览：`docs/EDP_复试梳理与演讲稿.md`（尤其一、二、三、四、五节）。
- 代码：`diffusion/trainer.py`（入口、`_setup`、训练循环、评估循环）、`diffusion/dql.py`（`_train_step_td3` / `_train_step_crr` / `_train_step_iql` 各做啥）。

### 2.2 讲稿建议结构（3–5 分钟）

1. **背景与动机（30–60 秒）**  
   EDP 是 NeurIPS 2023 的离线强化学习工作；用扩散模型表示策略，在 D4RL 上取得 SOTA。基于 Diffusion-QL 做两点改进：**训练效率**（action approximation，约 25× 加速）、**算法兼容性**（ELBO 近似 log π，支持 CRR/IQL）；生成上用 DPM-Solver 少步采样。

2. **方法核心（约 1 分钟）**  
   策略是以状态为条件的扩散模型：给定观测，从噪声去噪得到动作。训练有两个损失：**扩散损失**（去噪 MSE，拟合数据）、**引导损失**（由所选算法决定：TD3 直接最大化 Q，CRR/IQL 用优势加权 log π）。采样用 DDPM/DDIM 或 DPM-Solver；**论文所有结果用 EAS**：采多个动作再用 Q 选一个（代码里 ensemble），评估更稳、分数更好。

3. **代码结构（6–8 句）**  
   入口与流程在 `trainer.py`：`DiffusionTrainer()` → `train()` 里首次 `_setup()` 建数据、策略、Q/V、Agent、SamplerPolicy。算法主体在 `dql.py`：`DiffusionQL` 根据 `loss_type` 走 TD3/CRR/IQL，算 value loss、diff loss、guide loss，更新 Q/V 和 policy。扩散过程在 `diffusion.py`（加噪/去噪、training_losses），策略网络与采样在 `nets.py`（DiffusionPolicy、ddpm_sample/dpm_sample）。数据从 D4RL 经 `replay_buffer.get_d4rl_dataset`、`traj_dataset` 的 n-step、`data/dataset.py` 的 `Dataset`，每步 `dataset.sample()` 得到 batch 喂给 `agent.train(batch)`。评估时用 `SamplerPolicy` 的多种 `*_act`（含 EAS），在 `TrajSampler.sample` 里滚轨迹，用 `get_normalized_score` 算 normalized return。

4. **实验与结果**  
   D4RL normalized return 作为主要指标。**三方面改进**：训练加速（action approximation）、生成加速（DPM 少步）、算法兼容（CRR/IQL）；**评估**：论文用 EAS（多采动作再用 Q 选），DPM 少步采样加速生成。

5. **自己的理解/反思（可选）**  
   优势、局限、可改进之处（如：不同域选不同 loss_type、超参与稳定性等）。

### 2.3 当天需要能做到

- **评估与 EAS**：用 2–3 句话说明——论文为什么说「用 Q 选」好？评估里「多种方式」是在做什么？
- **整体串联**：能在 **3–5 分钟**内比较顺畅地完整介绍整个项目（可先对照文档演讲稿练，再变成自己的表达）。
- **代码路径**：能对着文件说清：入口 → `_setup` → 训练循环（sample → agent.train）→ 评估循环（act_methods → SamplerPolicy → TrajSampler.sample → normalized return）。

---

## 三、参考答案（简要）

**Part A**

1. 训练阶段不跑完整反向链，只做加噪 + 噪声 MSE + 用 pred_astart 算 guide loss；评估阶段在 `SamplerPolicy` 调用 ddpm_sample/dpm_sample 等时才从 x_T 去噪到 x_0 生成动作。  
2. 每种采样方式**分别**跑轨迹、**分别**记一列指标（通过 `post = "_" + method` 区分），不是组合成一次评估。  
3. EAS = 采 N 个动作再用 Q 选一个（或按 e^Q 加权）；论文写明所有结果基于 EAS，用 Q 选能降低评估方差、得到更好分数。  
4. ddpm_act / dpm_act / ddim_act：单采样；ensemble_act / ddpmensemble_act / dpmensemble_act：多采 + 双 Q 打分再选一个（EAS）。  
5. DDPM 满 T 步随机去噪，DPM 少步 ODE；EDP 生成效率体现在用 DPM 少步达到相近性能、推理更快。  
6. act_methods：要评估的采样方式列表；post：指标后缀（多方式时为 `_method`）；recent_returns：每个 method 最近若干次 return 的 deque。

**Part B**

7. 见上文 2.2 节「背景与动机」+ 三方面改进。  
8. 入口 main → DiffusionTrainer() → train() → _setup()（数据、策略、Q/V、Agent、SamplerPolicy）→ 训练循环（dataset.sample → agent.train）→ 评估循环（act_methods → SamplerPolicy → TrajSampler.sample → get_normalized_score）。  
9. 见上文 2.2 节「方法核心」。  
10. 最终成果通过 D4RL 的 **normalized return** 展示；训练速度 25× 是**技术贡献**，不是评估策略好坏的指标。

---

## 四、与一周计划、复试梳理的对应

- **一周计划**：原 Day 5.5（评估与 EAS）+ 原 Day 6（整体串联 + 讲稿）已合并到本文档。  
- **复试梳理**：评估细节见 **3.5**；讲稿与常见问答见 **五、六**；公式与概念索引见 **六末尾表格**。  
- **下一步**：Day 7 查漏补缺 + 对照复试梳理「常见面试问题与回答要点」逐条准备，确保能流畅回答并完整介绍项目。
