## EDP 一周学习与复现计划

> 目标：7 天内完成 **能跑通实验 + 能用自己的话清晰讲解方法和代码结构**。  
> 每天任务可根据你实际时间稍作调整。  
>
> **DQL 与 EDP 定位（先建立整体图景）**  
> - **DQL（Diffusion-QL）**：用扩散模型表示策略；训练时**联合训练**噪声网络（diff_loss + guide_loss）和 Q 网络（value loss）。策略侧：guide loss 用的「动作」是当前步前向得到的 **pred_astart**（EDP 的 action approximation），再用 **Q 网络前向**得到的 Q(s, pred_astart) 构造 guide loss（不是 value loss 的输出）。训练循环内**不跑完整生成**；**生成**（从噪声去噪得到动作）仅在**评估/部署**时进行。  
> - **EDP 三方面改进**：① **训练效率**（action approximation：用 pred_astart 当动作，不跑完整反向链）；② **生成效率**（DPM-Solver 少步采样）；③ **算法兼容性**（ELBO 近似 log π，支持 CRR/IQL）。详见 `docs/EDP_复试梳理与演讲稿.md` 的 **1.2 DQL 整体思路**、**1.3 核心问题与动机**。  
>  
> **一周总览：每天在干啥（对应 EDP 三方面改进 + 流程与数据）**  
> - **Day 1：整体结构 + 流程**——先把「从命令行到训练/评估」的完整链路搞清楚：入口 → `DiffusionTrainer.__init__` → `_setup_*` → 训练循环 → 评估循环，不抠公式。  
> - **Day 2：扩散策略本身（DDPM/DDIM/DPM）**——看清 `GaussianDiffusion` + `DiffusionPolicy`，理解扩散前向/后向、ELBO→MSE、三种采样方式；为后续「训练时 diff_loss、评估时生成动作」打基础。  
> - **Day 3：TD3 + 扩散策略（训练效率 + 兼容 TD3）**——在 TD3 分支里看 Q 更新（value loss）、diff loss + guide loss 组合，理解 **action approximation（pred_astart）** 如何加速训练（相对 DQL 若用完整生成链算 guide 的改进）。  
> - **Day 4：CRR / IQL + 扩散策略（算法兼容性）**——看 CRR/IQL 的 value/guide loss，理解为何需要 log π、EDP 如何用 ELBO 近似 log π，让扩散策略也能接 CRR/IQL（DQL 不兼容）。  
> - **Day 5：数据流 + 跑通一个实验**——从 D4RL/RLUP 读离线数据 → `Dataset.sample()` → `agent.train(batch)`，真正跑一次小实验。  
> - **Day 5.5：评估与 EAS / 多种采样方式（生成效率的体现）**——理解评估时「生成」才发生：ddpm/dpm/ensemble/EAS 以及 DPM vs DDPM 的对比（论文 4.5）。  
> - **Day 6–7：串联 + Q&A**——把上面**三方面改进**（训练效率、生成效率、算法兼容性）串成一段从「问题 → 方法 → 代码 → 实验」的复试讲解，并准备常见问答。

---

### 代码与理论对应关系总表（先看这个）

| 文件 / 目录 | 干什么用的 | 对应什么理论知识 | 何时重点看 |
|-------------|------------|------------------|------------|
| **`diffusion/trainer.py`** | 训练与评估的入口和流程：组数据集、建策略/Q/V、建 agent、按 epoch 循环（取 batch → 调 agent.train → 定期评估）。不实现具体算法公式。 | 整体流程（数据 → 训练 → 评估）、评估协议（多种采样方式、EAS）。 | Day 1（结构）、Day 5.5（评估循环） |
| **`diffusion/dql.py`** | 算法核心：实现「扩散损失 + 引导损失 + Q/V 更新」的具体公式（TD3/CRR/IQL 的 value loss、policy loss、target 更新）。 | 扩散训练目标、TD3/CRR/IQL、双 Q、target network、guide loss、action approximation。 | Day 1（骨架）、Day 3（TD3）、Day 4（CRR/IQL） ← **主要代码** |
| **`diffusion/diffusion.py`** | 扩散过程：β/α 调度、前向加噪 q(x_t\|x_0)、后向去噪 p(x_{t-1}\|x_t)、DDPM/DDIM 采样、training_losses（MSE）。 | 扩散模型前向/后向、ELBO 与 MSE、采样（DDPM/DDIM）、生成加速（DPM-Solver 接口）。 | Day 2 |
| **`diffusion/nets.py`** | 网络结构：扩散策略网络（PolicyNet + 时间嵌入）、Critic（Q）、Value（V）、GaussianPolicy；以及 ddpm/ddim/dpm_sample 的调用。 | 条件扩散策略、Q/V 网络、采样方式（DDPM/DDIM/DPM）。 | Day 2 |
| **`data/dataset.py`** + **`utilities/replay_buffer.py`** | 数据：从 D4RL/RLUP 取数据，封装成 batch（obs, action, reward, next_obs, done）。 | 离线数据集、n-step、batch 采样。 | Day 5 |
| **`diffusion/trainer.py` 里的 `SamplerPolicy`** | 评估时怎么选动作：ddpm_act、dpm_act、ddim_act、*_ensemble_act（多采动作用 Q 选）。 | EAS、多种采样方式对比（DPM vs DDPM、ensemble），评估协议。 | Day 5.5 |

> 提示：按「概览 → 细节」来看代码：先用本表知道大致分工，再在下面各 Day 的任务中跟着具体函数名去看。

---

### 项目执行顺序与代码文件对照（按运行顺序看）

从命令行 `python -m diffusion.trainer ...` 开始，实际执行顺序和主要文件如下，便于按「数据 → 训练 → 生成/评估」对着代码看。

| 阶段 | 做什么 | 主要代码文件 |
|------|--------|--------------|
| **0. 入口** | 启动训练 | `diffusion/trainer.py`：`main()` → **`DiffusionTrainer()`** → `trainer.train()`。 |
| **1. Trainer 的 __init__（只做配置）** | 只读 FLAGS、存配置、选激活函数；不建数据、不建网络。 | `diffusion/trainer.py`：`DiffusionTrainer.__init__()`。此时还没有 `_dataset`、`_policy`、`_agent`。 |
| **2. _setup()（真正建数据与网络）** | `train()` 里第一行调用的 `_setup()`，建 logger、数据、策略、Q/V、agent、SamplerPolicy。 | `diffusion/trainer.py`：`train()` 内 `self._setup()`，其内依次 2.1～2.5。 |
| 2.1 数据准备 | 取 D4RL/RLUP 数据 → n-step → 封装成可采样 Dataset | `utilities/replay_buffer.py`：`get_d4rl_dataset`（内部用 `utilities/traj_dataset.py` 的 `get_nstep_dataset`）；`data/dataset.py`：`Dataset` + `RandSampler`；`utilities/sampler.py`：`TrajSampler`（评估时滚轨迹用）。 |
| 2.2 策略与扩散 | 加噪/去噪、训练 loss、采样接口 | `diffusion/diffusion.py`：`GaussianDiffusion`（q_sample、p_mean_variance、training_losses）；`diffusion/nets.py`：`DiffusionPolicy`（PolicyNet、ddpm_sample/dpm_sample）。 |
| 2.3 Q / V | Critic（双 Q）、Value | `diffusion/nets.py`：`Critic`、`Value`。 |
| 2.4 Agent | 算法主体：Q 用 value loss 更新；策略用 diff loss + guide loss 更新（guide 用 Q 网络前向得到的 Q 值） | `diffusion/dql.py`：`DiffusionQL`（`get_value_loss`、`get_diff_loss`、`_train_step_td3/crr/iql`）。 |
| 2.5 评估用 policy | 评估时「obs → action」 | `diffusion/trainer.py`：`SamplerPolicy`（`ddpm_act`、`dpm_act`、`*_ensemble_act`）。 |
| **3. 训练循环** | 每步取 batch、更新参数；**不跑完整生成**，guide loss 用当前步前向的 pred_astart | `data/dataset.py`：`Dataset.sample()` → `diffusion/trainer.py`：`agent.train(batch)` → `diffusion/dql.py`：`train() → _train_step_*`（内部用 `diffusion/diffusion.py` 的 `training_losses` 算扩散 MSE + 在 `dql.py` 里算 guide loss）。 |
| **4. 评估** | **此时才做生成**：滚轨迹时每步从噪声去噪得到动作，算 normalized return | `utilities/sampler.py`：`TrajSampler.sample(sampler_policy, ...)`；每步动作：`SamplerPolicy` → `diffusion/nets.py` 的 `DiffusionPolicy`（ddpm/dpm 采样）→ `diffusion/diffusion.py` 的 `p_sample_loop` 或 `diffusion/dpm_solver.py` 的 `DPM_Solver`；指标在 `diffusion/trainer.py` 里用 `get_normalized_score` 算。 |
| **5. 保存（可选）** | 存 checkpoint | `diffusion/trainer.py`。 |

> 提示：如果想按“执行流程”配合代码看一遍，就对照上表；后面每个 Day 具体要看的函数，会在对应小节再写一遍，不需要记所有行号。

---

## Day 1：整体理解与代码入口

**目标**：知道项目在解决什么问题，整体结构怎样，从哪里跑起来。

**今天看的代码对应什么理论**：整体训练流程（从命令行到数据、到 agent 单步更新、到评估），不涉及具体公式。

- **阅读内容**
  - `docs/EDP_复试梳理与演讲稿.md`  
    - 重点章节：一、二、四、五（背景、理论、完整流程、演讲稿）。  
  - `docs/Day1_整体结构与入口.md`（今天的学习笔记和自测题在这里）。  
  - `README.md` 的 `Usage` 和 `Run Experiments` 部分。

- **代码浏览**
  - `diffusion/trainer.py`（用途：训练与评估的入口和流程；理论：流程控制、数据→训练→评估）  
    - 只看结构，不抠公式：  
    - `DiffusionTrainer.__init__`  
    - `_setup_dataset` / `_setup_policy` / `_setup_qf` / `_setup_vf`  
    - `train()`  
    - 文件末尾 `main` 和 `if __name__ == '__main__':`。  
  - `diffusion/dql.py`（用途：算法核心，算 loss、更新参数；理论：扩散 + TD3/CRR/IQL；← 主要代码）  
    - 只看：  
      - `class DiffusionQL`（整体结构）  
      - `get_default_config`（超参大致分哪几类）  
      - `__init__`（构造了哪些模块、TrainState、target 参数）  
      - `_train_step`（如何根据 `loss_type` 分发到三种算法）  
      - `train`（被 trainer 调用的一步训练接口）  
    - 一句话记：**dql.py：`DiffusionQL` → `get_default_config` → `__init__` → `_train_step_*` → `train`**。

- **当天需要能做到**
  - 用 **1 分钟**中文说明：  
    - EDP 要解决什么问题、为什么用扩散模型做策略、**三方面改进**（训练效率、生成效率、算法兼容性）大致是什么。  
  - 用 **5–8 句**话说明：  
    - 从 `python -m diffusion.trainer ...` 到训练/评估中间发生的关键步骤：入口 → `__init__` → `_setup` → 训练循环 → 评估循环。  
  - 能指出：  
    - 谁是「模型本体」（`DiffusionPolicy`），谁是「算法壳」（`DiffusionQL`），谁负责「流程调度」（`DiffusionTrainer` + `SamplerPolicy`）。

---

## Day 2：扩散策略本身（先不管 RL）

**目标**：弄清楚“扩散策略”如何加噪/去噪，以及如何输出动作。

**今天看的代码对应什么理论**：扩散模型前向 q(x_t\|x_0)、后向 p(x_{t-1}\|x_t)、训练 loss（ELBO → 噪声 MSE）、采样（DDPM/DDIM/DPM），以及和 VAE/ELBO 的关系。对应复试梳理 **2.1、2.4、2.5** 和「常见问题 11（DDPM 与 VAE/ELBO）」。

- **阅读内容**
  - `diffusion/diffusion.py`（用途：扩散过程与 loss；理论：加噪/去噪、ELBO→MSE、采样） ← 今天主要代码  
    - `GaussianDiffusion.__init__`（β、α、ᾱ 等的作用，理解大致含义即可）  
    - `q_sample`（前向加噪）  
    - `q_posterior_mean_variance`（前向后验 q(x_{t-1}\|x_t,x_0)）  
    - `p_mean_variance`（用预测噪声/预测 x₀ 构造 p(x_{t-1}\|x_t) 的均值/方差 + pred_xstart）  
    - `training_losses`（ELBO → 噪声 MSE）  
  - `diffusion/nets.py`（用途：策略网络与 Q/V 网络结构；理论：条件扩散策略、时间嵌入、DDPM/DDIM/DPM 采样接口）  
    - `TimeEmbedding`（时间步嵌入）  
    - `PolicyNet`（state + action + time_embed → 特征）  
    - `DiffusionPolicy` 中：`__call__`、`loss`、`ddpm_sample` / `ddim_sample` / `dpm_sample`。

- **建议操作**
  - 画一条前向/反向小图：  
    - 训练：$x_0 \xrightarrow{q\_sample} x_t \xrightarrow{\epsilon_\theta} \hat{\epsilon}$，loss = $\|\epsilon - \hat\epsilon\|^2$。  
    - 生成：$x_T \xrightarrow{p_\theta} x_{T-1} \to \dots \to x_0$，用 DDPM / DDIM / DPM-Solver 完成。

- **当天需要能做到**
  - 口头说清楚一次扩散训练 step 在做什么：给定 (s, a=x0)，如何随机选 t，加噪得到 x_t，再通过网络预测噪声、计算 MSE。  
  - 懂得：推理时给一个状态 s，如何通过扩散模型（DDPM/DDIM/DPM）从噪声生成一个动作 a。  
  - 能回答「DDPM 和 VAE/ELBO 的关系」的大致思路：DDPM 是一个变分生成模型，ELBO 在固定方差 + 噪声参数化下化为噪声 MSE。

---

## Day 3：离线 RL——先吃掉 TD3 版本（训练加速 + TD3 头）

**目标**：明确当 `loss_type=TD3` 时，一步训练里发生了什么，理解 action approximation 在哪里起作用。

**今天看的代码对应什么理论**：TD3（双 Q、target network、policy 最大化 Q）+ 扩散 loss + guide loss；对应复试梳理 **2.2.2（TD3 对比项）**、**2.3（双损失）**、**3.3（训练一步）**。

- **前置知识：TD3 基础回顾**
  - 双 Q：Q1/Q2 取 min 减少过估计。  
  - target network：soft update 保证目标稳定。  
  - policy 更新：最大化 Q(s, π(s))，即最小化 $-Q$。

- **阅读内容**
  - `diffusion/dql.py`（用途：TD3 的 Q loss 与 policy loss 实现；理论：TD3 双 Q、target、guide loss） ← 今天主要代码  
    - `get_value_loss`（双 Q + target Q 的 MSE）  
    - `get_diff_loss`（扩散 MSE + pred_astart）  
    - `_train_step_td3`（Q loss 和 policy loss 怎么组合；在哪里用到 action approximation）  
  - 可结合 `docs/EDP_复试梳理与演讲稿.md` 的「2.2.2 TD3 对比项」「2.3 双损失」「3.3 训练一步」一起看。

- **关注点**
  - value loss：如何从 batch 的 (s,a,r,s') 构造 target Q、当前 Q1/Q2 和 MSE。  
  - diff loss：`get_diff_loss` 里如何得到 diff_loss 和 `pred_astart`（训练加速）。  
  - guide loss：在 `_train_step_td3` 里，如何用 Q(s, pred_astart) 构造 $-\lambda Q$ 并与 diff_loss 相加。  
  - target 更新：`update_target_network` 在什么时候更新 policy 和 Q 的 target 参数。

- **当天需要能做到**
  - 手写/口述一段伪代码，描述「TD3 + 扩散策略」的一步 `_train_step_td3`。  
  - 能解释：  
    - 与普通 TD3 相比，这里最大的区别在哪里（policy 是扩散模型 + guide loss 用的是 pred_astart 而不是直接 π(s) 的输出）。  
    - action approximation 为何能加速训练（训练时用 pred_astart 当动作，不再跑完整反向链）。

---

## Day 4：CRR / IQL 思路与代码对照（算法兼容性）

**目标**：理解 CRR、IQL 在这个框架里的角色，能说出它们和 TD3 的差异，以及为什么需要 ELBO 近似 log π。

**今天看的代码对应什么理论**：CRR（优势加权 log π）、IQL（expectile V、AWR 加权）；对应复试梳理 **1.3 改进三（算法兼容性 + ELBO 近似 log π）**、**2.2.1（CRR 与 IQL）**。

- **阅读内容**
  - `docs/EDP_复试梳理与演讲稿.md` 中：  
    - **1.2 DQL 整体思路**、**1.3 改进三**（算法兼容性 + ELBO 近似 log π）；  
    - 2.2 离线 RL（TD3/CRR/IQL）；  
    - 2.3 双损失。  
  - `diffusion/dql.py`（用途：CRR/IQL 的 value loss 与 policy guide；理论：优势权重、expectile、AWR） ← 今天主要代码  
    - `_train_step_crr`  
    - `_train_step_iql`

- **关注点**
  - CRR：  
    - 如何用 Q 估计优势 $A = Q - \mathbb{E}Q$；  
    - 权重 $\lambda(A)$ 如何由优势变来（exp / heaviside / softmax）；  
    - guide_loss 如何变成「优势加权的 log π(a|s)」。  
  - IQL：  
    - `value_loss`：expectile 回归（V 对 Q 的分位拟合，V ≈ 某个分位的 Q）；  
    - `critic_loss`：用 V 做 bootstrap 目标更新 Q（TD）；  
    - `policy_loss`：用 $\exp((Q−V)/\tau)$ 权重加权 log π(a|s)（AWR），再加 diff_loss。  
  - ELBO 近似 log π：  
    - 来自 `get_diff_terms` 里通过 `action_dist.log_prob` / `-ts_weights * mse` 把扩散的预测变成一个近似的 log π，用于 CRR/IQL 的 guide loss。

- **当天需要能做到**
  - 一句话概括每个 loss 的核心思想：  
    - TD3：value-maximization（直接最大化 Q）。  
    - CRR：优势加权行为克隆（高优势动作被更大权重放大）。  
    - IQL：通过 V/Q 分解 + expectile + AWR 做隐式策略学习。  
  - 用 3–5 句话回答：  
    - 为何在离线设定下，这三种方法比直接 DDPG/PG 更安全、更鲁棒。  
    - EDP 为何要支持 CRR/IQL（算法兼容性 + 在 Antmaze/Kitchen 等域里的表现；Diffusion-QL 不支持这些算法）。

---

## Day 5：数据流 + 跑通一个小实验

**目标**：搞清数据流向，并在自己机器上真正跑通一轮训练。

**今天看的代码对应什么理论**：离线数据集、n-step、batch 采样；不涉及扩散或 RL 公式。

- **阅读内容**
  - `utilities/replay_buffer.py`（用途：从 D4RL 取数据、n-step；理论：离线数据、n-step return）
    - `get_d4rl_dataset`
  - `utilities/traj_dataset.py`（用途：按轨迹、n-step 处理；理论：轨迹划分、n-step）——只大致看 `split_into_trajectories`、n-step 相关逻辑。
  - `data/dataset.py`（用途：把数据封装成 batch、随机采样；理论：batch 采样）
    - `class Dataset`、`sample()`、`retrieve()`
    - `class RLUPDataset`（略读，知道 RLUP 也是产出同结构 batch 即可）。

- **动手运行**
  - 安装依赖（如果还没做）：`pip install -e .`；按 README 配好 D4RL 和 MuJoCo 或使用 bullet fallback。
  - 运行一个小规模实验（只是测试 pipeline 通不通）：
    ```bash
    python -m diffusion.trainer --env 'hopper-medium-v2' \
      --logging.output_dir './exp_debug' \
      --algo_cfg.loss_type=TD3 \
      --n_epochs=2 \
      --n_train_step_per_epoch=10
    ```
  - 检查：终端是否输出 `agent.*loss`、`average_normalizd_return` 等；`exp_debug` 目录是否生成日志文件。

- **当天需要能做到**
  - 用自己的话说明：D4RL 数据从 env 导出，到被包装成 `Dataset`，再到 `batch = dataset.sample()`，最后喂给 `agent.train(batch)` 的完整路径。

---

## Day 5.5（可选）→ 已并入 Day 6

评估与 EAS / 多种采样方式的内容已整合到 **`docs/Day6_评估EAS与整体串联.md`** 的「一、评估部分」；请直接按 Day 6 文档学习。

---

## Day 6：评估与 EAS + 整体串联与复试讲解准备

**目标**：① 理解评估时多种 act_method、EAS、DPM vs DDPM；② 能从「问题 → 方法 → 代码 → 实验」一口气讲下来，3–5 分钟完整介绍项目。

**详细内容（含提问自测、理论、代码位置、讲稿结构、参考答案）** 见 **`docs/Day6_评估EAS与整体串联.md`**。

- **阅读与复盘**：复试梳理 3.5、论文 4.5；`trainer.py` 评估循环与 `SamplerPolicy`；复试梳理一～五、`dql.py` 三个 `_train_step_*`。
- **当天需要能做到**：说清 EAS 与「多种方式」在做什么；按讲稿结构在 3–5 分钟内完整介绍项目；能对着代码说清入口 → _setup → 训练 → 评估。

---

## Day 7：查漏补缺 + Q&A 准备

**目标**：查缺补漏，准备常见问答，确保自己输出稳定。

- **查漏**
  - 回顾一周内容，问自己：哪几块讲起来还会卡壳？（如 IQL 的 expectile、DPM-Solver、数据预处理细节等）
  - 有疑问的地方，再回到相应文件 + 文档定位：理论去 `docs/EDP_复试梳理与演讲稿.md`，实现去相应 `.py` 文件中搜索对应函数。

- **准备问答**
  - 对照文档中「常见面试问题与回答要点」一节，逐条梳理自己的回答：
    - 为什么用扩散而不是 GAN/VAE？
    - 三种 loss 的区别和优缺点？
    - normalized return 是什么，如何理解？
    - 如果训练不稳定，你会从哪几个方向排查？
    - **EAS/ensemble 是什么？论文为什么说用 Q 选？**（见复试梳理 3.5 与问答 8）
    - **action approximation 是什么？**（见复试梳理 3.5 与问答 9）
    - DDPM 和 VAE/ELBO 的关系？（常见问题 11）

- **最终输出**
  - 能流畅回答上述问题，并能按 Day 6 的讲稿结构完整介绍项目；代码关键路径（入口 → _setup → 训练循环 → 评估）能对着文件说清楚。

