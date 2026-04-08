# Efficient Diffusion Policy (EDP) 复试梳理与演讲稿

> 本文档结合代码与理论，梳理 EDP 项目完整思路，便于复现与面试讲解。后续有疑问可随时追问。

---

## 一、项目与论文背景

### 1.1 一句话概括

**EDP（Efficient Diffusion Policies for Offline RL）**：用**扩散模型**表示策略，在**离线强化学习**设定下训练，实现「训练高效、可插拔多种 RL 算法、在 D4RL 上 SOTA」的决策策略。

### 1.2 DQL（Diffusion-QL）整体思路（EDP 所基于的框架）

**目标**：在离线 RL 里用**扩散模型**表示策略 π(a|s)，在固定数据上训练，得到既拟合数据、又偏向高价值动作的策略。

**策略形式**：策略 = 以 state s 为条件的扩散模型。不是「s → a」的一步映射，而是「给定 s，从噪声 x_T 一步步去噪得到动作 x_0」的生成过程；实现上是一个**噪声预测网络**，输入 (s, x_t, t)，输出预测噪声（或 pred_xstart）。

**训练部分（联合训练）**：  
- **网络**：噪声预测网络（PolicyNet，即策略的可训练部分）、Q 网络（Critic，双 Q + target）；CRR/IQL 还有 policy_dist、IQL 还有 V。  
- **噪声网络**的损失：**diff_loss**（噪声 MSE，拟合数据）+ **guide_loss**（用 **当前 Q 网络**算出的 Q(s, â) 构造，如 −λ Q(s, â)，â 为 pred_astart；梯度**只更新噪声网络**，不更新 Q）。  
- **Q 网络**的损失：**value loss**（TD target 的 MSE）；**只有** value loss 会更新 Q。guide loss 与 value loss 用的是**同一套 Q**（同一动作价值函数），只是 value loss 用来**学 Q**，guide loss 只**用 Q 的取值**给策略梯度。  
- **联合**：每步同时更新噪声网络（diff_loss + guide_loss）和 Q（value loss）；actor 与 critic 共同约束：Q 学出「动作好坏」后，guide loss 才能把策略往高 Q 拉。**注意**：联合训练指「策略与 Q 同时更新」；训练循环内**不跑完整生成**算 guide，故不是「先训练再生成」。  
- **原始 DQL 的代价**：若算 guide loss 时用「完整生成链」得到的动作 â，每步训练都要跑 T 次去噪，极慢。

**生成部分（仅评估/部署时）**：输入 state s；**采样** x_T ~ N(0,I)（随机噪声，非由 s 闭式算出），以 s 为条件用训练好的噪声网络从 t=T 到 t=1 逐步去噪，得到 x_0 作为动作 a。**生成阶段不算任何损失**，只做「给定 s → 输出一个 a」。

**一句话！！！**：DQL = 用扩散当策略，训练时用 diff_loss + guide_loss + value loss 联合训策略和 Q；生成时从噪声去噪得到动作。本质是「扩散生成动作 + Q 引导策略往高回报走」。

---

### 1.3 核心问题与动机（与摘要一致）：EDP 如何改进 DQL

**摘要原文**：Diffusion-QL 用扩散模型表示策略、显著提升离线 RL，但依赖数百步马尔可夫链采样；**“However, Diffusion-QL suffers from two critical limitations.”**（1）训练时在整个马尔可夫链上前向/反向计算，计算效率低；（2）扩散模型的 likelihood 不可算（intractable），与基于最大似然的 RL 算法不兼容。因此提出 **EDP 克服这两个挑战**。  
下面**三方面改进**对应：训练效率、生成效率、算法兼容性（摘要（1）拆为训练+生成，（2）为兼容性）。

- **EDP**：用扩散模型表示策略 π(a|s)，在离线数据上训练，通过**双损失**（扩散 MSE + guide loss）和 Q 网络联合优化。
- **改进一（效率）**：训练时用 **action approximation**（单步 pred_astart 当动作算 guide loss，不跑完整反向链）；训练里**还有一条链**是 value loss 算 $a'=\pi_{\mathrm{target}}(s')$ 时的反向采样，当 `sample_method='dpm'` 时用 **DPM-Solver** 少步替代满 T 步，从而训练也加速。生成/评估时同样用 DPM-Solver 少步采样。
- **改进二（兼容性）**：CRR、IQL 的 guide loss 是**加权 MLE**，需要 **log π(a|s)**；扩散策略只能采样、无闭式 log π。EDP 用 **ELBO 或对角高斯**在 pred_astart 处构造可算的 log π 近似，供 CRR/IQL 使用。

**改进二：生成效率**  
- **问题**：生成时若用完整 DDPM（T 步）去噪，推理慢。  
- **EDP 做法**：**DPM-Solver** 等少步 ODE 采样（如 15 步）代替完整 T 步。  
- **结果**：评估/部署时「给定 s 得到 a」更快。

**改进三：算法兼容性（论文 4.3，摘要中的 incompatible）**  
- **问题**：利用 Q 来更新策略有**两种方式**——  
  - **方式 A**：像 **TD3**，用损失函数**直接最大化 Q**（如 guide loss = −λQ(s,â)），**不需要**策略的 log π(a|s)，扩散策略本来就能接。  
  - **方式 B**：像 **CRR、IQL**，用 **加权 MLE**（用 Q 或优势加权，对 log π(a|s) 做极大似然），**必须能算 log π(a|s)**。扩散策略只能采样、给不出闭式 log π，与方式 B **不兼容**。  
- **EDP 做法**：用 **ELBO（或高斯近似）** 在 pred_astart 等处给出 **log π(a|s) 的可算近似**，使 CRR、IQL 的更新式能用上“log π”这一项。  
- **注意**：DDPM 里 ELBO 是拿来**训练扩散**的（MSE）；EDP 的 ELBO 是拿来**给策略提供 tractable log π**，让下游 CRR/IQL 能用，二者用途不同。

---
**EDP 贡献小结**  
- **训练高效**（action approximation）、**生成高效**（DPM-Solver）、**兼容多种离线 RL 算法**（TD3 / CRR / IQL 任选其一）、D4RL 四类任务 SOTA。  
- **三种 loss**：通过 `loss_type=TD3|CRR|IQL` **选一种**训练，不是同时算三种。**理解论文重点在 CRR 和 IQL**（兼容性的主要贡献）；TD3 是对比项（不需要 log π）。

---
**简短澄清（避免混淆）**  
- **为何支持三种算法**：不同 benchmark 域上表现不同（如 locomotion 常用 TD3，Antmaze 等常用 IQL）；支持三种是为了泛用性与按任务选算法。  
- **Benchmark / 域**：D4RL = 标准任务+数据集+评估方式；Locomotion、Antmaze、Kitchen、Adroit 是**域（环境类型）**，每域下有多条数据集（如 HalfCheetah-v2-medium-replay）。谁决定用哪种算法由实验者根据效果选择。  
- **TD3**：是 actor-critic（策略更新最大化 Q，不用 log π）；「用 TD3 做损失」= 用其 value loss + guide loss（−λQ）当训练目标。

### 1.4 论文信息

- **标题**：Efficient Diffusion Policies for Offline Reinforcement Learning  
- **会议**：NeurIPS 2023  
- **机构**：Sea AI Lab  
- **基于的工作**：论文明确引用 **Diffusion-QL [37]** 作为基础（摘要与 Introduction 均提到），EDP 针对 Diffusion-QL 的两个局限提出改进。  
- **代码**：Jax 实现，结构参考 JaxCQL，扩散部分参考 OpenAI guided-diffusion 与 Diffusion-QL。

---

## 二、理论知识梳理

### 2.1 扩散模型（DDPM）：宏观目标 → 生成需要什么 → 训练学什么 → ELBO 与 MSE

**宏观目标**  
- 要训练一个**生成模型**：能从噪声得到「泛化」的数据（在 EDP 里是**动作** $x_0$）。总体目标是对 $x_0$ 的分布 $p_\theta(x_0)$，即希望**最大化** $\mathbb{E}_{\text{data}}[\log p_\theta(x_0)]$，让模型给真实数据高概率。  
- 该目标**无法直接算**：$\log p_\theta(x_0)$ 需要对整条扩散轨迹 $x_{1:T}$ 积分，没有闭式。  
- 因此把思路拆成两段：**加噪（训练用）** 和 **去噪（生成用）**；并用 **ELBO** 把「max log p(x0)」落到**训练阶段**的一个可算损失上。

**从生成部分入手：去噪需要什么**  
- 生成时我们要得到**预测的** $x_0$。每一步是**从 $x_t$ 采样 $x_{t-1}$**，即用 $p_\theta(x_{t-1}|x_t)$。  
- 该分布取成**高斯**时：**方差**由 schedule 的 $\beta_t$ 等给定（已知），**均值**必须用公式算出来。  
- 问题：均值公式来自**前向过程的后验** $q(x_{t-1}|x_t, x_0)$——给定 $x_t$ **和** $x_0$ 时，后验均值和方差是闭式的。但**生成时没有真实 $x_0$**，不能直接套「后验链」。  
- 做法：**仍用同一套均值闭式**，把其中的 $x_0$ 换成**当前步的预测 $\hat{x}_0$**（或等价的预测噪声 $\epsilon_\theta$）。也就是说：生成时均值的闭式**需要**一个「中间量」——预测的 $x_0$ 或 $\epsilon$，才能算出来。

**训练部分的目标：学出这个「中间量」**  
- 训练阶段的目标就明确了：学一个**噪声预测网络** $\epsilon_\theta(x_t, t \mid \text{obs})$（或等价的 $\hat{x}_0$），使得**生成时**可以把网络输出代入上述均值公式，得到 $p_\theta(x_{t-1}|x_t)$ 的均值，再按固定方差采样。  
- 训练时我们有**真实 $x_0$、采样的噪声、以及加噪得到的 $x_t$**，所以前向和后验 $q(x_{t-1}|x_t, x_0)$ 都是**闭式**的。我们让网络预测的分布 $p_\theta(x_{t-1}|x_t)$ 去**拟合**这个后验（即让 $p_\theta$ 的均值逼近 $q$ 的均值），等价于让网络预测的 $\epsilon$（或 $\hat{x}_0$）逼近真实加噪时用的 $\epsilon$（或真实 $x_0$）。  
- 因此**训练损失**就是对预测噪声的 **MSE**（本代码用 `ModelMeanType.EPSILON`，即预测 $\epsilon$）：
  $$
  \mathcal{L}_{\text{diff}} = \mathbb{E}_{t,x_0,\epsilon}\big[ \| \epsilon - \epsilon_\theta(x_t, t \mid \text{obs}) \|^2 \big]
  $$
  条件 **obs** 即当前观测，**以状态为条件的扩散 = 策略**。

**ELBO 和「加噪 / 去噪」的关系（回答：ELBO 只和训练部分 MSE 有关）**  
- **宏观上**我们想 max $\log p_\theta(x_0)$，不可算，所以用其下界 **ELBO** 作为训练目标；ELBO 展开后包含一系列 KL：让 $p_\theta(x_{t-1}|x_t)$ 逼近 $q(x_{t-1}|x_t,x_0)$。  
- 在 DDPM 的设定下（方差固定由 schedule 定），每个 KL 只与**均值差**有关，最小化 KL = 最小化均差的平方；用预测 $\epsilon$ 的参数化时，**最小化 KL 等价于最小化 $\|\epsilon - \epsilon_\theta\|^2$**，即**训练时的 MSE**。  
- 因此：**ELBO 只在训练阶段起作用**——它**规定**了训练损失取成 MSE；**生成阶段不涉及 ELBO**，只是用训练好的网络算出预测 $\hat{x}_0$（或 $\epsilon_\theta$），代入**同一套均值闭式**得到每步高斯均值，再按固定方差采样。  
- 小结：宏观目标 max log p(x0) → 用 ELBO 代替 → ELBO 细分为「每一步 p 拟合 q 的 KL」→ 在固定方差下变成 **MSE（训练部分）**；生成部分只用「网络输出 + 均值闭式 + 固定方差」采样，不再出现 ELBO。

**前向与反向的公式（便于查）**  
- **前向（加噪）**：$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t)I)$，即 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$。  
- **采样方式**：DDPM 满 T 步；DDIM 确定性、可跳步；DPM-Solver 少步 ODE（代码里约 15 步）。

**代码里 p 和 q 的区别（`diffusion/diffusion.py`）**  
- **q**：**前向过程及其后验**，固定、不学习。`q_sample` = 从 q(x_t|x_0) 采样；`q_posterior_mean_variance(x_start, x_t, t)` = **q(x_{t-1}|x_t,x_0)** 的均值和方差（闭式）。训练时有真实 x_0，用它当**目标**让 p 拟合。  
- **p**：**学到的反向**。`p_mean_variance(model_output, x, t, ...)`：用**网络输出**得到 p(x_{t-1}|x_t) 的均值与 pred_xstart；实现上常用网络得到 pred_xstart，再**用 q 的后验公式**把 pred_xstart 当代入算均值（同一套闭式，只是 x_0 换成预测值）。**生成时只有 p**（没有真实 x_0），用 p 一步步去噪。

### 2.2 离线强化学习与三种算法

- **设定**：只有**固定数据集** $\mathcal{D}$（如 D4RL），不与环境交互；要学策略 $\pi(a|s)$ 尽量利用数据中的好行为，避免**分布偏移**和**过估计**。

- **过估计（简述）**：TD target $y = r + \gamma Q(s',a')$ 里用到了 Q；若 Q 已高估，y 会偏大；value loss 是 $\hat{Q}(s,a)$ 与 y 的 MSE，会把 $\hat{Q}$ 往偏大的 y 拉，下一轮 target 又更高，形成**正反馈**，不断往高估走。双 Q 取 min 压低 target，缓解过估计。

- **三种 loss 类型（任选其一）**：

| 算法 | 策略更新方式 | 是否需要 log π | 论文中的角色 |
|------|----------------|-----------------|----------------|
| **TD3** | 直接最大化 Q，guide loss = −λQ(s,â) | 不需要 | 对比项，扩散策略本来就能接 |
| **CRR** | 优势加权 MLE：权重 ∝ 优势，对 log π(a\|s) 极大似然 | 需要 | 兼容性重点 |
| **IQL** | 学 V、Q，policy 用 AWR 权重 exp((Q−V)/τ) 加权 log π(a\|s) | 需要 | 兼容性重点 |

**理解论文 4.3 时以 CRR、IQL 为重点**；TD3 知道「直接最大化 Q、不需 log π」即可。

#### 2.2.1 CRR 与 IQL（重点：为何需要 log π，EDP 如何兼容）

**为什么这类算法需要 log π**  
- 两类方法都优化同一目标 $J(\pi)=\mathbb{E}[Q(s,a)]$。策略梯度定理给出 $\nabla J \propto \mathbb{E}[Q \cdot \nabla\log\pi]$（随机）或 $\nabla J = \mathbb{E}[\nabla_a Q \cdot \nabla_\theta \pi]$（**DPG**，Deterministic Policy Gradient，确定性策略无需 log π）。
- **方式 A**（TD3）：用 **DPG** 路径，$\nabla Q(s,\pi(s))$ 经 Q 反传，**不需要** log π。  
- **方式 B**（CRR、IQL）：用**加权 MLE** $\mathbb{E}[\omega \cdot \log\pi(a|s)]$，**必须** log π。扩散策略只能采样，给不出闭式 log π，EDP 用 ELBO 近似解决。

**CRR（Critic Regularized Regression）**  
- 权重 $\lambda \propto \exp(A/\beta)$ 或 $\lambda=1[A>0]$，其中 $A = Q(s,a) - \mathbb{E}_\pi Q$（优势）；$\beta$ 为逆温度。guide loss：$-\mathbb{E}[\lambda \cdot \log\pi(a|s)]$。代码：`_train_step_crr` 用 `action_dist.log_prob(actions)` 与权重。

**IQL（Implicit Q-Learning）**  
- 权重 $\omega \propto \exp((Q-V)/\tau)$，$\tau$ 为逆温度；V 来自 **expectile 回归**，更保守。guide loss：$-\mathbb{E}[\omega \cdot \log\pi(a|s)]$。  
- **β 与 τ**：形式相同（$\exp(\cdot/\text{温度})$），都是逆温度；CRR 的 A 用 Q 减策略下 Q 的期望，IQL 的 Q−V 中 V 用 expectile，不取 max，更防过估计。

**EDP 如何提供 log π**  
- 在扩散的某步（如 pred_astart 处）用 **ELBO 或固定方差高斯** 构造 tractable 的 log π 近似，供 CRR/IQL 的 guide loss 使用。代码里通过 `policy_dist`、`action_dist.log_prob` 等接入（见 `dql.py` 的 CRR/IQL 分支）。

**action_dist 与 policy_dist（代码对应）**  
- **action_dist**：以 pred_astart 为均值、**std 来自 policy_dist 可学习参数 log_stds** 的对角高斯（`nets.py` 的 `GaussianPolicy`），与 DDPM 反向链的固定方差无关，是辅助近似 log π 的分布。CRR/IQL 会更新 policy_dist；TD3 不更新。  
- **为何有「log π」的公式？** 对角高斯的 **log 概率是概率论里的标准公式**（各维 log 密度之和），不是为 RL 推的。我们只是**用这个高斯分布去近似**策略 $\pi(a|s)$，把该高斯的 log 概率**当作** $\log \pi(a|s)$ 的近似，故「log π」= 该对角的 log 密度。公式：
  $$\log p(\mathbf{a}) = \sum_{d=1}^{D} \left[ -\log\sigma_d - \frac{1}{2}\log(2\pi) - \frac{(a_d-\mu_d)^2}{2\sigma_d^2} \right]$$
  （$\boldsymbol{\mu}=\texttt{pred\_astart}$，$\sigma_d$ 来自 log_stds）。详见 Day4 **七、7.0**。
- **向量与 state 的对应**：公式里的向量就是**动作向量**（定义域 = 动作空间）。「给定 state s」体现在均值 $\boldsymbol{\mu}=\mathrm{pred\_astart}(s)$ 依赖 s，故每个 s 对应一个高斯；对数据中的 (s,a)，在该 s 对应的高斯下算 a 的 log 密度，即近似 log π(a|s)。  
- **π(x₀|s) 无闭式**：单步 $p(x_{t-1}|x_t)$ 有闭式，但 π(x₀|s) 是整条链的边际积分，无闭式；与是否 action approximation 无关。  
- **log π 在代码中**：mle 用 `action_dist.log_prob(actions)`（上式在 distrax 中实现）；elbo 用 `-terms['ts_weights']*terms['mse']`（`dql.py` 538–539，mse 为真实噪声与预测噪声的 MSE，见 `diffusion.py` 1140、1199）。

**diff_loss 与 guide_loss（elbo 模式）**  
- 都用同一次前向的噪声 MSE；diff_loss 对全体样本均匀最小化，guide_loss 用**优势权重 λ** 加权后最小化，高优势样本压得更狠。  
- **CRR 用 λ 不用 Q**：离线数据用优势加权只加强「比平均好」的动作，更稳；Q 已通过 λ=f(A) 体现。

**ELBO、噪声 MSE 与 CRR guide loss 的 MSE（澄清）**  
- **训练扩散**：$\log p_\theta(x_0)$ 不可算 → 用 **ELBO** 作为目标；DDPM 固定方差下 ELBO 化简为**噪声 MSE**（真实 ε vs 预测），故训噪声网络的损失即此项，**来自 ELBO**。生成时不算 ELBO，只用前向后验均值公式 + 网络预测 $\hat{x}_0$。  
- **CRR elbo 近似 log π**：用的就是**同一次前向**的 terms['mse'] 与 terms['ts_weights']，即**同一个噪声 MSE**，再按时间步加权：$\log\pi \approx -\mathtt{ts\_weights}\times\mathtt{mse}$。时间步权重 $w_t = \beta_t/(2(1-\bar\alpha_t)\alpha_t)$，见 `diffusion.py` 约 264、267 行。  
- **两种「难算」**：① 训扩散的损失由 ELBO 推出（噪声 MSE），**TD3/CRR/IQL 都一样**。② 只有 CRR/IQL 需要 **log π**，才用 ELBO 或高斯近似；TD3 不用 log π，故不用「为 log π 的 ELBO」。详见 Day4 **七、7.10**。

**V 的估计与 replicated_obs**  
- **CRR** 里需估计 $V(s)=\mathbb{E}_a Q(s,a)$ 作为优势的减数，无闭式，用蒙特卡洛：对每个 $s$ 采多份动作、算 $Q(s,a)$ 再平均。replicated_obs 把每个 $s$ 复制多份，与 vf_actions 配对后一次前向算 Q，再对采样维平均得 $V$。**IQL** 有独立 V 网络，用 expectile 回归学 V，不需要此步骤。

**Target 更新与训练分工**  
- **TD target** $y=r+\gamma\min Q'(s',a')$ 用 **target** $\pi'$ 和 **target** $Q'$ 计算，故 $y$ 随 target 缓慢变化（软更新），不随当前 Q/π 每步乱动，体现**延迟更新**。**Guide loss** 用 **current** Q(s, pred_astart)，不给 Q 反传梯度。  
- **Q target**：每步软更新。**policy target**：仅当 `policy_tgt_update=True` 时软更新（`_total_steps>1000` 且 `_total_steps % policy_tgt_freq==0`，默认每 5 步），以稳定 TD 目标。  
- **训练循环**在 `trainer.py`（epoch/step、取 batch、评估）；**单步更新**在 `dql.py`（算法相关的 loss 与梯度），trainer 只调 `agent.train(batch)`。

**IQL 补充（expectile、V、命名、双 Q）**  
- **Expectile 与 V**：τ 为超参（常取 0.7，>0.5）；V(s) 是 Q(s,a) 在数据分布下的 **τ-expectile**，不是 E_a[Q]。用 expectile 回归学 V 时**不需对动作采样**；不对称平方损失：diff=Q−V，diff>0 权重 τ、diff≤0 权重 1−τ，loss = mean(weight·diff²)。  
- **value loss / critic loss**：TD3、CRR 里更新 Q 的都叫 **value loss**；IQL 里更新 V 的叫 value_loss、更新 Q 的叫 critic_loss。  
- **双 Q 在 IQL**：TD target 用 V(s')；双 Q 用于 min(Q1,Q2) 作为训练 V 的目标和 policy 权重 (Q−V)，减轻过估计。  
- **AWR**：IQL 的 guide 在论文里叫 AWR；TD3 无 log π；CRR 只写 guide loss。详见 Day4 **七、7.8**。

#### 2.2.2 TD3（对比：不需要 log π）

- **TD3**：actor-critic，策略更新**直接最大化 Q**，即 guide loss = $-\lambda Q(s,\hat{a})$，**不用** log π。三块组件：**双 Q**（取 min 减过估计）、**target network**（软更新稳目标）、**policy 最大化 Q**。
- **在 EDP 里**：`get_value_loss` 用双 Q + target + MSE；`_train_step_td3` 里 policy loss = diff_loss + guide_coef × (−λQ(s, pred_astart))。细节见 `dql.py` 的 `_train_step_td3`、`get_value_loss`。

### 2.3 EDP 的训练目标（双损失）

策略由**扩散模型**表示，训练时同时优化：

1. **扩散损失** $\mathcal{L}_{\text{diff}}$：让策略在给定 $s$ 下拟合数据中的动作分布（去噪 MSE）。
2. **引导损失** $\mathcal{L}_{\text{guide}}$：用当前所选算法（TD3/CRR/IQL）的信号，把策略往高 Q/高优势方向拉。

总策略损失：$\mathcal{L}_{\text{policy}} = \mathcal{L}_{\text{diff}} + \lambda_{\text{guide}} \cdot \mathcal{L}_{\text{guide}}$。  
- **TD3**：$\mathcal{L}_{\text{guide}} = -\lambda Q(s, \hat{a})$（$\hat{a}$ 为 pred_astart）。  
- **CRR**：$\mathcal{L}_{\text{guide}} = -\mathbb{E}[\lambda \cdot \log\pi(a|s)]$，$\lambda$ 为优势相关权重。  
- **IQL**：$\mathcal{L}_{\text{guide}}$ 为 AWR 式，权重 $\exp((Q-V)/\tau)$ 加权 $\log\pi(a|s)$。  

**Q 在 value loss 与 guide loss 里是同一个**：都是当前学的那套动作价值函数（双 Q）。value loss **更新** Q（拟合同一 TD target）；guide loss **只用** Q 的取值（如 $Q(s,\hat{a})$）给策略梯度，不更新 Q。确定性/随机策略梯度定理里用的都是这同一个 Q；训练 Q 的目的是学出「动作好坏」信号，guide loss 才能把策略往高回报方向拉（actor 与 critic 共同约束）。  

公式细节见 **2.2.1（CRR/IQL）** 与 **2.2.2（TD3）**。

#### 2.3.1 为什么需要联合训练？为什么可以用"还没训练好"的网络输出？

**核心理解：这是联合训练（Joint Training），不是分阶段训练**

- **联合训练**指：每步同时更新**扩散策略**（用 diff_loss + guide_loss）和 **Q 网络**（用 value loss）；策略与 Q 一起训、互相配合。**不是**「每步先训练再生成」——训练循环内**不跑完整生成**，guide loss 用的动作是当前步前向得到的 pred_astart，不是生成链的终点。  
- 训练是**迭代优化的过程**，不是"先训练好扩散模型，再用它做 RL"，而是"同时训练扩散模型和 Q 网络，让它们互相配合"。

**为什么 RL 需要用到扩散模型的训练结果？**

在 DiffusionQL 中，扩散模型不是用来生成图像的，而是用来**表示策略 π(a|s)**。策略需要和 Q 网络配合，才能做强化学习：

1. **扩散模型训练**（diff_loss）：让网络学会预测噪声，拟合数据中的动作分布。
2. **Q 网络训练**（value_loss）：用 TD 误差更新 Q(s, a)，评估动作价值。
3. **策略更新**（policy_loss = diff_loss + guide_loss）：
   - `diff_loss`：让策略拟合数据中的动作分布。
   - `guide_loss`：让策略偏向高价值动作（用 Q 网络评估 `pred_astart` 的价值）。

三者配合，才能实现离线强化学习。如果分开训练，扩散模型只拟合数据，不知道哪些动作好，可能学到很多"低价值动作"，后续 RL 很难纠正。

**为什么可以用"还没训练好"的网络输出？**

即使网络还没完全训练好，它的输出仍然包含**有效的梯度信息**：

- **梯度仍然有效**：`pred_astart` 是从当前网络参数计算出来的（虽然不完美），但梯度会回传到网络，让网络朝着"预测更准确 + 偏向高价值动作"的方向更新。
- **迭代优化过程**：训练是逐步改进的，每一步都在优化：
  - 第 1 步：网络参数随机初始化（很差）→ `pred_astart` 不准确 → `guide_loss` 可能很大，但梯度方向是对的 → 更新网络。
  - 第 N 步：网络参数训练好了 → `pred_astart` 准确了 → `guide_loss` 小了 → 网络既拟合数据，又偏向高价值动作。

**数学上的理解**

联合优化问题：$\min_\theta \left[ \mathcal{L}_{\text{diff}}(\theta) + \lambda \cdot \mathcal{L}_{\text{guide}}(\theta) \right]$

两个目标同时优化：
- `diff_loss` 的梯度：让网络预测更准确。
- `guide_loss` 的梯度：让网络偏向高价值动作。

即使网络不完美，梯度方向仍然正确，网络会逐渐改进。

**为什么这样设计？**

- **如果分开训练**：先训练扩散模型，再做 RL → 扩散模型只拟合数据，不知道哪些动作好 → 可能学到很多"低价值动作" → 后续 RL 很难纠正。
- **联合训练（DiffusionQL）**：一边拟合数据，一边偏向高价值动作 → 两个目标互相促进 → 最终学到"既符合数据分布，又偏向高价值"的策略。

**实际训练中的表现**

- **训练初期**：`pred_astart` 不准确（网络还没训练好）→ `guide_loss` 可能很大 → 但梯度方向正确，网络会逐渐改进。
- **训练中期**：`pred_astart` 逐渐准确 → `guide_loss` 逐渐减小 → 网络同时学会"预测准确"和"偏向高价值"。
- **训练后期**：`pred_astart` 很准确 → `guide_loss` 很小 → 网络既拟合数据，又偏向高价值动作。

**总结**：训练是迭代过程，每一步都在改进；即使网络不完美，梯度仍然有效；联合优化比分开训练效果更好。这就是为什么 DiffusionQL 要用"还没训练好"的网络输出来计算 guide loss 的原因。

### 2.4 扩散模型：训练阶段 vs 生成阶段（与 2.1 对应）

与 **2.1** 的脉络一致：宏观目标 max log p(x0) 不可算 → 用 ELBO；**ELBO 只体现在训练阶段的损失（MSE）**；生成阶段不涉及 ELBO，只用「网络输出 + 同一套均值闭式 + 固定方差」采样。

---

#### 训练阶段

| 项目 | 说明 |
|------|------|
| **目标** | 宏观上 max $\mathbb{E}[\log p_\theta(x_0)]$；因 $\log p_\theta(x_0)$ 不可直接算，用其下界 **ELBO** 作为训练目标。 |
| **ELBO 的组成** | 重建项 + $\sum_t D_{\mathrm{KL}}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))$。训练时有真实 $x_0$，$q(x_{t-1}|x_t,x_0)$ **闭式**。 |
| **需要学的** | 让 $p_\theta(x_{t-1}|x_t)$ 逼近 $q$，即用网络预测均值（或等价地预测 $\epsilon$ / $\hat{x}_0$）。 |
| **为何是 MSE** | 方差固定时 KL 只与均值差有关；用预测 $\epsilon$ 的参数化 → **最小化 KL = 最小化 $\|\epsilon - \epsilon_\theta\|^2$**，即**训练损失取 MSE**。所以 **ELBO 只和训练部分挂钩**，细分为对噪声预测网络的 MSE。 |
| **本项目的改进** | 见 **1.3 改进一**：action approximation，约 25× 训练加速、支持长步数。 |

---

#### 生成阶段（采样）

| 项目 | 说明 |
|------|------|
| **目标** | 从学好的模型**采样**得到 $x_0$。**不计算** log p(x0)，**也不使用 ELBO**；只用已学到的反向过程逐步采样。 |
| **过程** | 从 $x_T \sim \mathcal{N}(0,I)$ 起，每步用 $p_\theta(x_{t-1}|x_t)$ 采样。该分布为高斯：**方差**由 schedule 固定，**均值**用**同一套** $q(x_{t-1}|x_t,x_0)$ 的均值公式，但把其中的 $x_0$ 换成**网络给出的预测 $\hat{x}_0$**（或由 $\epsilon_\theta$ 反推的 $\hat{x}_0$）。 |
| **闭式** | 给定网络输出的 $\hat{x}_0$（或 $\epsilon_\theta$），均值可闭式算出，再 + 固定方差 → 一次高斯采样，即 $x_{t-1}$。 |
| **本项目的改进** | 见 **1.3 改进二**：DPM-Solver 等少步采样，加快推理。 |

---

#### 小结：ELBO 与 加噪/去噪 的对应

- **ELBO** 只和**训练（加噪侧利用的真实 x0 + 闭式后验）** 相关：它规定了训练损失取 MSE，使网络学会「给定 $x_t,t$ 输出预测 $\epsilon$/$\hat{x}_0$」。  
- **生成（去噪）** 不用 ELBO：用训练好的网络得到预测 $\hat{x}_0$，代入**同一套均值闭式**得到每步高斯均值，再按固定方差采样。训练阶段学的就是这个「能代入均值公式的条件」。

#### 2.4.1 一步 $\hat{x}_0$（action approximation）vs 完整去噪链：为什么在 EDP 里可以「近似」？

- **同一条公式，训练和生成都在用**  
  - 前向加噪：$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$；  
  - 若网络预测噪声 $\epsilon_\theta(x_t,t)$，可用闭式反推「一步预测的 $x_0$」：$\hat{x}_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}$。  
  - 这条「从 $x_t$ 反推出 $\hat{x}_0$」的公式**不是只在生成里才出现**：训练时也在用，只是通常把它藏在 `pred_xstart` 字段里。

- **传统 DDPM/DQL：生成阶段需要「高质量样本」，所以要多步去噪**  
  - 纯生成任务（图像、语音）里，我们关心的是「最终样本 $x_0$ 的质量」；  
  - 一步 $\hat{x}_0$ 在高噪声 $t$ 处误差大，需要 **T 步反向去噪链**把这些局部修正积累起来，才能得到高质量的 $x_0$；  
  - 因此传统 DQL 若在训练中也用「完整生成链取动作再算 guide loss」，每次更新都要跑 T 步，训练极慢。

- **EDP：guide loss 只要「方向对」，不需要每次都生成高保真 $x_0$**  
  - TD3 模式下，guide loss = $-\lambda Q(s,\hat{a})$，$\hat{a}$ 只作为**给 Q 打分的动作**，我们真正关心的是 $\nabla_a Q(s,a)$ 的**方向**；  
  - 只要 $\hat{a}$（即 `pred_astart`）被 diff_loss 拉在「数据动作附近」，且 Q 在这附近大致合理，$\nabla_a Q$ 就能指向「往更高 Q 的动作」；  
  - 因此可以用 **当前步随机 $t$ 上的一步 $\hat{x}_0$** 当动作（action approximation），省掉整条去噪链，只在「局部」求导就足够给策略一个有用的更新方向。

- **精度与效率的边际权衡：为什么不在训练里也跑完整链？**  
  - 用完整 T 步反向链得到更好的 $x_0$，确实能让 guide loss 的梯度更「干净」一点，但：  
    - （1）**每步训练成本 ×T**，整体训练时间几乎不可接受；  
    - （2）随着网络和 Q 变好，一步 $\hat{x}_0$ 的质量本身也在提升，「再多跑 T−1 步」的**边际收益越来越小**。  
  - 在 EDP 的设定下，「一步 $\hat{x}_0$ + 频繁更新」在总时间一定的前提下，往往比「少更新几次、每次都跑完整生成链」更划算：  
    - 更便宜的近似动作 → 可以做**更多步迭代**；  
    - diff_loss 与 value loss 持续把策略和 Q 拉向正确区域 → 一步近似的方向感不断变好。  
  - **说明（论文里没有对应曲线）**：上述「边际收益递减」是**定性推理**，EDP 论文**并未给出**「训练时用 pred_astart vs 用完整链」的收敛曲线对比（即没有「normalized return vs 训练步数」下两条曲线的图）。论文实际提供的是：① **Fig.2** 训练/采样**速度**对比（EDP w/o AP 更慢，但无 return 曲线）；② **Tab.1** 最终 normalized score（EDP vs DQL vs Diffusion-QL），说明用 action approximation 不伤最终性能、且能训更大 K；③ **Fig.5** 是**评估阶段** DPM-Solver 步数（3～30）与性能的关系，展示的是**采样步数**的边际效益递减，与**训练时**是否用完整链无关。若要自己验证「边际效益递减」，可做：固定墙钟时间，对比「use_pred_astart=True 跑满 2000 epoch」与「use_pred_astart=False 只跑约 1/T 的 epoch」的 return 曲线，或随 epoch 画 diff_loss / pred_astart 与完整链动作的 MSE。

- **总结一句话**：  
  - 传统 DDPM/DQL：**生成阶段**为了高质量样本，要用多步去噪把「一步预测 $\hat{x}_0$ 的局部信息」整合起来；  
  - EDP：在 **训练阶段**，guide loss 只需「在数据附近给出高 Q 的梯度方向」，不必每次都生成高保真样本；  
  - 因此可以用 action approximation（一步闭式 $\hat{x}_0$）替代完整去噪链来算 guide loss，显著加速训练，而在联合训练和 diff_loss 的约束下，最终学到的策略质量仍然可以和完整链相媲美。

### 2.5 扩散理论补充：ELBO → MSE（训练阶段），以及和 VAE 的类比

- 宏观目标 max $\log p_\theta(x_0)$ 不可算 → 用 **ELBO** 作为可优化的下界。  
- ELBO 展开后含 $\sum_t D_{\mathrm{KL}}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))$：让 $p_\theta$ 逼近前向后验 $q$（训练时有 $x_0$，$q$ 闭式）。  
- 在 DDPM 固定方差下，KL ∝ 均值差平方；用预测 $\epsilon$ 的参数化 → **最大化 ELBO = 最小化 $\|\epsilon - \epsilon_\theta\|^2$**，即**训练时的 MSE**。  
- 因此：**ELBO 仅在训练阶段体现为损失（MSE）**；生成阶段只是「网络输出 + 均值闭式 + 固定方差」采样，不再出现 ELBO。

**和 VAE 的关系（回答你的直觉）**  
- 可以把 DDPM 看成一种**特殊的变分生成模型**，和 VAE 很像：  
  - VAE 里有 $p_\theta(z)$、$p_\theta(x|z)$ 和变分分布 $q_\phi(z|x)$，用 ELBO 来替代难算的 $\log p_\theta(x)$。  
  - DDPM 里「潜变量」是一整条马尔可夫噪声链 $x_{1:T}$，前向链 $q(x_{1:T}|x_0)$ 是**固定的变分分布**，反向链 $p_\theta(x_{0:T})$ 是要学的生成模型，同样通过 ELBO 近似最大化 $\log p_\theta(x_0)$。  
- 只不过在 DDPM 里，由于前/后向都设成高斯、方差固定，再加上噪声参数化，**ELBO 很优雅地化成了逐步拟合噪声的 MSE**，所以工程实现上我们只看到「随机选 t、加噪、用 MSE 训噪声网络」这一条链。  
- 换句话说：**DDPM 的训练“看起来”只是 MSE，但本质上是在最大化一个 ELBO；ELBO 是这条逻辑可行的理论根基**，而不是“完全无关，只是顺带得到 MSE”。

### 2.6 DPM‑Solver：从 DDPM 到少步 ODE 采样（面试版思路）

> **两处会用「从噪声到 $x_0$ 的反向链」**：① **训练时** value loss 里算 $a'=\pi_{\mathrm{target}}(s')$ 需要跑一条链（代码里 `policy.apply(tgt_params['policy'], rng, next_observations)` 会调用 ddpm_sample 或 dpm_sample）；② **评估时**从状态生成动作。当 `sample_method='dpm'` 时，这两处都会用 DPM-Solver 少步采样，从而**训练和评估**都加速。DPM **不改** diff loss、**不参与** action approximation（guide loss 用 pred_astart，不跑链）。

#### 2.6.1 为什么要从 DDPM 换成 DPM‑Solver？

- **DDPM 视角**：反向是**随机 Markov 链**，每一步
  \[
  x_{t-1} \sim \mathcal N(\mu_\theta(x_t),\ \sigma_t^2 I),
  \]
  一般要跑 100～1000 步才能从 $x_T$ 去到 $x_0$，推理很慢。
- **ODE 视角**：这条随机链在连续时间极限上对应一条**概率流 ODE**，可以写成
  \[
  \frac{\mathrm d x_t}{\mathrm d t} = f_\theta(x_t,t),
  \]
  起点仍是 $x_T\sim\mathcal N(0,I)$，只是中间不再每步加噪声。  
- **DPM‑Solver 做的事**：针对这条 ODE 设计一个**高阶数值解法**，用很少的时间步（如 15 步）逼近 DDPM 的完整采样链，大幅减少「网络 forward 次数」，从而加速采样。

#### 2.6.2 数学主线（只抓 EDP 需要的那一条）

1. **前向 VP‑SDE（与 2.1 的离散公式一致）**
   \[
   \mathrm d x_t = -\tfrac12\beta(t)\,x_t\,\mathrm dt + \sqrt{\beta(t)}\,\mathrm d w_t,
   \quad
   q(x_t|x_0)=\mathcal N(\alpha_t x_0,\ \sigma_t^2I).
   \]
2. **反向 SDE（随机版反向过程）**
   \[
   \mathrm d x_t
   = \Bigl[-\tfrac12\beta(t)\,x_t
           -\beta(t)\,\nabla_x\log p_t(x_t)\Bigr]\mathrm dt
     +\sqrt{\beta(t)}\,\mathrm d\bar w_t.
   \]
3. **概率流 ODE（确定版，去掉噪声项）**
   \[
   \frac{\mathrm d x_t}{\mathrm d t}
   = -\tfrac12\beta(t)\,x_t
     -\tfrac12\beta(t)\,\nabla_x\log p_t(x_t),
   \]
   在每个时间点的边缘分布 $p_t(x_t)$ 与反向 SDE 完全一致。
4. **用噪声预测 $\epsilon_\theta$ 写出漂移项 $f_\theta$**
   - 训练好的网络是噪声预测：$\epsilon_\theta(x_t,t\mid\text{obs})$。
   - 利用 $x_t=\alpha_t x_0+\sigma_t\epsilon$ 可得到 score–noise 关系：
     \[
     \nabla_x\log p_t(x_t)\ \approx\ -\frac{1}{\sigma_t}\,\epsilon_\theta(x_t,t).
     \]
   - 代回 ODE：
     \[
     \frac{\mathrm d x_t}{\mathrm d t}
     = -\tfrac12\beta(t)x_t
       +\tfrac12\frac{\beta(t)}{\sigma_t}\,\epsilon_\theta(x_t,t)
     \;\triangleq\; f_\theta(x_t,t).
     \]
   - 这就是「用噪声预测写出封闭形式漂移项」的含义：右边只依赖 $\beta(t),\alpha_t,\sigma_t$（由噪声调度给出）和 $\epsilon_\theta$。

在代码里，这个 $f_\theta$ 已被展开进 `dpm_solver.py` 各种 `*_update` 公式中的 $\phi_1,\phi_2,\phi_3$ 系数，你不需要手动再推一遍。

#### 2.6.3 在 EDP 代码里，DPM‑Solver 是怎么接进去的？

1. **准备连续时间的噪声调度：`NoiseScheduleVP`**
   - 输入离散的 `alphas_cumprod`（`GaussianDiffusion.alphas_cumprod`）。  
   - 通过插值提供任意 $t$ 上的 `marginal_alpha(t)`、`marginal_std(t)`、`marginal_lambda(t)`（即 $\alpha_t,\sigma_t,\lambda_t$）。  
   - 目的：让 ODE 在连续时间上随时能查到「噪声有多大」。  

2. **包装噪声网络为连续时间接口：`model_fn(x,t)`**
   - 在 `DiffusionPolicy.dpm_sample` 中：
     ```python
     ns = NoiseScheduleVP(..., alphas_cumprod=self.diffusion.alphas_cumprod)
     dpm_sampler = DPM_Solver(
       model_fn=wrap_model(partial(self.base_net, observations)),
       noise_schedule=ns,
       predict_x0=...
     )
     ```
   - `partial(self.base_net, observations)`：把 state 固定住，DPM‑Solver 只看到 $(x_t,t)$。`wrap_model` 做时间 rescale + 噪声裁剪，得到统一的 `model_fn(x,t)`。  
   - 数学上等价于：把离散的 $\epsilon_\theta(x_t,n)$ 延拓成连续的 $\epsilon_\theta(x_t,t)$，供上面的 $f_\theta$ 使用。

3. **用少量大时间步数值解 ODE：`DPM_Solver.sample`**
   - 起点：一次采样 $x_T\sim\mathcal N(0,I)$。  
   - `sample(x_T, steps=..., order=...)` 内部：
     - 用 `get_time_steps` 把 $[T\to t_0]$ 划成若干大区间；  
     - 每个区间 \([s,t]\) 上，根据阶数 order=1/2/3：
       - 1 阶：在 $s$ 调一次 `model_fn`，类似 DDIM 一步公式；  
       - 2 阶：在 $s,s_1$ 调两次，利用差分构造二阶修正；  
       - 3 阶：在 $s,s_1,s_2$ 调三次，构造三阶修正。  
     - 这些更新式本质上都是在近似积分 $\int f_\theta(x_t,t)dt$。
   - **steps 与 order 的关系**：
     - `steps` 是**总的网络 forward 次数预算**（NFE）；  
     - 每个区间内部调用次数 = 该区间使用的阶数（1/2/3）；  
     - DPM‑Solver 通过安排若干区间和每个区间的阶数，使「所有区间内的模型调用次数之和 = steps」，在给定 steps 下精度尽量高。

4. **为什么是「十几步」，但不是「一步到 $x_0$」？**
   - 数学上有
     \[
     x_0 = x_T + \int_{T}^{t_0} f_\theta(x_t,t)\,\mathrm dt,
     \]
     但积分里的 $x_t$ 本身是 ODE 的解、依赖整个轨迹，**不存在简单的一步闭式公式**。  
   - ODE 虽然是确定的，但仍需要数值积分逐段逼近轨迹；DPM‑Solver 做到的是：用高阶方法把「原来 1000 个小步」压成「十几次大步」，而不是 1 步完成。
   - 这里的「导数」/「斜率」指的就是 ODE 右边的 $f_\theta(x_t,t)$，在实现里由 `model_fn(x,t)` 给出；数值法不需要再对它求梯度，只是多次在区间内不同 $(x,t)$ 位置调用这个黑盒函数，看清这段轨迹的走向。

5. **训练里哪里还有「反向链」？为什么 DPM 也能加速训练？**
   - **扩散损失**（diff loss）每个样本只做：采一个 $t$、前向加噪得 $x_t$、调一次噪声网络算 MSE → **O(1) 步**，没有反向链，用不到 DPM。
   - **Value loss** 算 TD target 时需要 $a'=\pi_{\mathrm{target}}(s')$。扩散策略没有「s→a」的一步映射，必须**跑一条从噪声到 $x_0$ 的反向链**才能得到 $a'$。代码里通过 `policy.apply(tgt_params['policy'], rng, next_observations)` 实现，内部走 `DiffusionPolicy.__call__`，即 **ddpm_sample 或 dpm_sample**（由 `sample_method` 决定）。因此当 `sample_method='dpm'` 时，**训练时**算 next_action 也会用 DPM‑Solver（少步），从而 value loss 这条链被加速；**评估时**同样用 `dpm_sample` / `dpm_act` / `dpmensemble_act`。所以 DPM 既加速评估，也加速训练中「算 a'」的那条链。

---

## 三、代码架构与数据流

### 3.1 数据流（训练）

1. **数据源**：D4RL（或 RLUP）→ `get_d4rl_dataset` / `RLUPDataset` → 得到 `observations, actions, next_observations, rewards, dones`。
2. **封装**：`Dataset(data)` + `RandSampler` → 每步 `dataset.sample()` 得到一个 batch（默认 256）。
3. **Batch 内容**：`observations`（当前状态）、`actions`（数据中的动作，即扩散的 $x_0$）、`next_observations`、`rewards`、`dones`。

### 3.3 训练一步（以 TD3 为例）——`dql._train_step_td3`

**重要**：代码通过 `loss_type` 选择调用 `_train_step_td3` / `_train_step_crr` / `_train_step_iql` 中的**一个**，不是同时算三种。

1. **Value loss（Q 网络）**  
   - 算 TD target $y = r + \gamma \min_i Q'_i(s', a')$ 时：用 **target policy** 在 `next_observations` 上得到 `next_action`（即 $a'=\pi_{\mathrm{target}}(s')$，代码里 `policy.apply(tgt_params['policy'], ...)` 会跑 **ddpm_sample 或 dpm_sample**，即一条反向去噪链）；用 **target Q'** 算 $Q'(s', a')$。  
   - 用**当前** Q1/Q2 在 `(observations, actions)` 上算 current Q，拟合同一 $y$；  
   - 双 Q 的 MSE loss → **只更新** `qf1`、`qf2`；算完后再软更新 target。**延迟更新**：$y$ 只依赖 target，target 软更新故 $y$ 变化慢，训练更稳。

2. **Policy loss（扩散策略 = 噪声网络）**  
   - **Diffusion loss**：随机时间步 $t$，对 `actions` 加噪得 $x_t$，用 `policy.base_net(obs, x_t, t)` 预测 $\epsilon$，MSE；  
   - **Guide loss**：用 **当前 Q 网络**（不更新 Q）算 $Q(s, \hat{a})$，$\hat{a}=\text{pred_xstart}$，$\mathcal{L}_{\text{guide}} = -\lambda Q$；梯度只反传到 **噪声网络**（policy），不反传到 Q。  
   - 总 policy loss = diff_loss + guide_coef * guide_loss → 只更新 **policy**（即噪声预测网络 PolicyNet 的参数）。

3. **Target 更新**：Q 的 target 每步软更新；policy target 按 `policy_tgt_freq` 软更新。

### 3.4 推理时（采样动作）

- **输入**：当前观测 `obs`（可先做 obs_norm）。  
- **扩散采样**：从 $x_T\sim\mathcal{N}(0,I)$ 起，用 `policy.base_net(obs, x_t, t)` 逐步去噪得到 $x_0$，即动作。  
- **采样方式**：`ddpm_sample` / `ddim_sample` / `dpm_sample`（少步）；还可 `*ensemble`：采多个动作，用 Q 选最大 Q 的动作。  
- **输出**：动作 clip 到 env 的 `[-max_action, max_action]`。

对应代码在 `trainer.SamplerPolicy`：`ddpm_act`、`dpm_act`、`ddim_act`、`*ensemble_act` 等。

### 3.5 评估部分：多种采样方式与 EAS（论文 4.5 节）

- **多种方式**：评估时可以对「如何从扩散模型拿动作」试多种方式，**每种方式单独跑轨迹、单独记一列指标**（不是把多种方式组合成一次评估）。方式包括：  
  - **DDPM**：完整 T 步去噪；**DPM**：少步 ODE（EDP 默认，更快）；**DDIM**：少步确定性。  
  - **ensemble**（或 `ddpmensemble` / `dpmensemble`）：用同一种去噪方式**采多个动作**，再用 **Q 网络**给每个动作打分，按 Q 选一个（或按 $e^{Q}$ 加权抽样）。  
- **论文结论**：论文 4.5 节提出 **EAS（Energy-based Action Selection）** = 先采 N 个动作，再用权重 $\propto e^{Q(s,a)}$ 选一个，等价于从改进策略 $p(a|s)\propto e^{Q}\pi_\theta(a|s)$ 采样。论文写明 **“All results will be reported based on EAS”**，即**所有报告的结果都是用「用 Q 选动作」这种方式评估的**。因此：**“用 Q 值的好”** = 评估时用 EAS/ensemble 更好；代码里 `ensemble_act`、`ddpmensemble_act`、`dpmensemble_act` 即 EAS 的实现。  
- **与 DPM vs DDPM**：评估部分也是**展示 EDP 生成优化（DPM-Solver）相对 DDPM** 的对比：同一模型可分别用 DDPM、DPM 采样，看 normalized return；论文 Fig.2 表明 DPM 更快且性能相当。  
- **action approximation**：训练时算 guide loss 需要的「动作」用 **pred_astart**（当前步预测的 $x_0$）近似，而不是跑完整反向链采一次动作，以加速训练；对应配置 `use_pred_astart=True`（默认）。ensemble 不是本项目发明，是已有做法，本项目只是支持并默认用 EAS 报结果。

### 3.6 关键超参（便于口述）

- **扩散**：`num_timesteps=100`（可 1000）、`schedule_name='linear'`、`ModelMeanType.EPSILON`、`ModelVarType.FIXED_SMALL`。  
- **算法**：`loss_type='TD3'|'CRR'|'IQL'`、`diff_coef=1.0`、`guide_coef=1.0`、TD3 的 `alpha=2`。  
- **训练**：`n_epochs=2000`、`n_train_step_per_epoch=1000`、`batch_size=256`；各 env 的 `gn`（max_grad_norm）在 `hps.py`。

---

## 四、完整流程（训练 → 评估）

### 4.1 环境与依赖

- 安装：`pip install -e .`；配置 MuJoCo 与 D4RL（见 README）。
- 可选：W&B 做日志与可视化。

### 4.2 训练命令示例

```bash
# TD3 + DDPM，默认 100 步
python -m diffusion.trainer --env 'walker2d-medium-v2' --logging.output_dir './experiment_output' --algo_cfg.loss_type=TD3

# IQL + reward 归一化
python -m diffusion.trainer --env 'walker2d-medium-v2' --logging.output_dir './experiment_output' --algo_cfg.loss_type=IQL --norm_reward=True

# 使用 DPM 采样：少步、可配 1000 步扩散
python -m diffusion.trainer --env 'walker2d-medium-v2' --sample_method=dpm --algo_cfg.num_timesteps=1000
```

### 4.3 流程概览（按执行顺序 + 代码文件）

下面按**项目实际执行顺序**梳理，并标出主要涉及的**代码文件**，便于对着代码看。

| 阶段 | 做什么 | 主要代码文件 / 调用链 |
|------|--------|------------------------|
| **0. 程序入口** | 命令行启动 | `diffusion/trainer.py`：`if __name__ == '__main__'` → `main()` → **`DiffusionTrainer()`** → `trainer.train()`。 |
| **1. Trainer 的 __init__（只做配置）** | **只读 FLAGS、存配置、选激活函数、记 env**；**不建数据、不建网络**。 | **`diffusion/trainer.py`**：**`DiffusionTrainer.__init__()`**（约 231–261 行）。此时还没有 `_dataset`、`_policy`、`_agent` 等。 |
| **2. 初始化 _setup()（真正建数据与网络）** | **第一次**进入 `train()` 时调用，一次性建好 logger、数据、网络、agent、评估用 policy。 | **`diffusion/trainer.py`**：**`train()`** 内第一行 **`self._setup()`**（约 266 行），其内依次调用下面 2.1～2.6。 |
| 2.1 日志 | 建 logger | `_setup_logger()` |
| 2.2 数据准备 | 取离线数据 → 封装成可采样的 Dataset | **D4RL**：`_setup_d4rl()` 内先调 `utilities/replay_buffer.py` 的 **`get_d4rl_dataset()`**（内部用 `utilities/traj_dataset.py` 的 `get_nstep_dataset` 做 n-step 与排序），得到 dict；再在 trainer 里做 reward scale/clip；最后用 **`data/dataset.py`** 的 **`Dataset(dataset)`** + **`RandSampler`**，`dataset.set_sampler(sampler)`，得到 `self._dataset` 和 `self._eval_sampler`（**`utilities/sampler.py`** 的 **`TrajSampler`**）。 |
| 2.3 策略网络 | 建扩散策略（加噪/去噪 + 采样接口） | **`diffusion/diffusion.py`**：**`GaussianDiffusion`**（β/α 调度、q_sample、p_mean_variance、training_losses）；**`diffusion/nets.py`**：**`DiffusionPolicy`**（PolicyNet + ddpm_sample/ddim_sample/dpm_sample）。 |
| 2.4 Q / V 网络 | 建 Critic（双 Q）、Value（IQL 用） | **`diffusion/nets.py`**：**`Critic`**、**`Value`**。 |
| 2.5 Agent | 把 policy、qf、vf 等组装成算法类 | **`diffusion/dql.py`**：**`DiffusionQL`**（含 get_value_loss、get_diff_loss、_train_step_td3/_train_step_crr/_train_step_iql）。 |
| 2.6 评估用 Policy | 评估时「给定 obs 输出 action」的封装 | **`diffusion/trainer.py`**：**`SamplerPolicy`**（ddpm_act、dpm_act、ddim_act、*ensemble_act，内部调 `policy.apply(..., method=policy.ddpm_sample/dpm_sample)`）。 |
| **3. 训练循环（每 epoch）** | 多次取 batch、更新参数 | |
| 3.1 取 batch | 从数据集随机抽一个 batch | **`data/dataset.py`**：**`Dataset.sample()`**（内部用 RandSampler 取 indices，再 `retrieve(indices)`）→ 得到 observations, actions, next_observations, rewards, dones。 |
| 3.2 单步训练 | 算 loss、更新 Q/V 和 policy | **`diffusion/trainer.py`**：`batch_to_jax(batch)` → **`self._agent.train(batch)`**；**`diffusion/dql.py`**：**`DiffusionQL.train(batch)`** → **`_train_step`**（按 loss_type 分支到 `_train_step_td3` / `_train_step_crr` / `_train_step_iql`）。各分支内：**`get_value_loss`**（Q/V 更新）、**`get_diff_loss`**（扩散 MSE）、guide loss，policy loss = diff_loss + guide_coef × guide_loss；其中 **`policy.loss`** 会进 **`diffusion/nets.py`** 的 DiffusionPolicy.loss → **`diffusion/diffusion.py`** 的 **`training_losses`**（q_sample、噪声 MSE）。更新 train_states、tgt_params 在 **`diffusion/dql.py`**。 |
| **4. 评估（按 eval_period）** | 在环境中滚轨迹、算指标 | |
| 4.1 滚轨迹 | 用当前参数在 eval env 里采样多条轨迹 | **`diffusion/trainer.py`**：对每个 act_method 设 `sampler_policy.act_method`，调 **`utilities/sampler.py`** 的 **`TrajSampler.sample(sampler_policy.update_params(agent.train_params), eval_n_trajs, ...)`**；采样时每一步 **`SamplerPolicy(obs)`** → 对应 `*_act`（如 dpm_act / dpmensemble_act）→ **`diffusion/nets.py`** 的 **`DiffusionPolicy.apply(..., method=ddpm_sample/dpm_sample)`** → **`diffusion/diffusion.py`** 的 **`p_sample_loop`** 或 **`diffusion/dpm_solver.py`** 的 **DPM_Solver** 得到动作。 |
| 4.2 计算指标 | 对轨迹算 return、normalized return 等 | **`diffusion/trainer.py`**：对每条轨迹 `np.sum(t["rewards"])`，用 **`eval_sampler.env.get_normalized_score(...)`** 得 **normalized return**（D4RL 标准分数）；写入 metrics（average_return、average_normalizd_return、best_normalized_return 等）。 |
| **5. 保存（可选）** | 存 checkpoint | **`diffusion/trainer.py`**：若 `save_model`，保存 agent、variant、epoch。 |

**init 小结**：**初始化分两段**——① **`DiffusionTrainer()`** 时只执行 **`__init__`**（配置），② 真正建数据、建网络是在 **`train()` 里第一次调 `_setup()`** 时（2.1～2.6）。所以看「数据、策略、Agent 在哪建」要看 **`_setup()`** 及其调用的 `_setup_dataset`、`_setup_policy`、`_algo(...)` 等。

**训练 vs 生成（没有合在一起）**：你在扩散理论里细分的「**训练**」（加噪 + MSE 训噪声网络）和「**生成**」（从 x_T 完整去噪得到 x0）在**项目执行流程里是分开的**。  
- **阶段 3（训练循环）**：用 **`training_losses`**（q_sample 加噪 + 噪声 MSE），不跑完整反向链；算 **guide loss** 时用 **pred_astart**（action approximation），不跑链。但算 **value loss** 的 TD target 时需要 $a'=\pi_{\mathrm{target}}(s')$，会跑**一条**反向链（`policy.apply` → ddpm_sample/dpm_sample），当 `sample_method='dpm'` 时用 DPM 少步。  
- **阶段 4（评估）**才做「为环境滚轨迹」的**生成**：每步用 **p_sample_loop** 或 **DPM_Solver** 从噪声去噪得到动作。  
所以：训练阶段内只有「训练用到的链」（value 里算 a' 的一条 + 无链的 guide）；为评估滚轨迹的生成只在阶段 4。

**简要串联**：入口 **trainer.py** → **__init__**（仅配置）→ **train() → _setup()**（数据 **replay_buffer + traj_dataset + data/dataset**、策略 **diffusion.py + nets.py**、agent **dql.py**、SamplerPolicy）→ 每步训练 **dataset.sample() → dql.train(batch)**（内部用 **diffusion.training_losses** 算扩散 MSE）→ 定期评估 **TrajSampler.sample(SamplerPolicy)**（SamplerPolicy 用 **nets** 的 ddpm/dpm 采样，底层 **diffusion.p_sample_loop** 或 **dpm_solver**）→ 指标在 **trainer.py** 里用 **get_normalized_score** 算。

**注意**：训练速度（25× 加速）是技术贡献，但**最终成果展示**是通过评估时的 **normalized return** 体现的。  

**按天拆分的「哪几天重点看哪几段流程」**见 **`docs/EDP_一周学习与复现计划.md`** 中「项目执行顺序与代码文件对照」下的表格「哪几天重点看哪几段流程」。

---

## 五、复试演讲稿（3–5 分钟口述提纲）

下面可直接用作「项目介绍 + 技术要点」的讲话稿，按需删减或展开。

---

**开场（约 30 秒）**  
我复现的是 NeurIPS 2023 的 **Efficient Diffusion Policy（EDP）**，做的是**离线强化学习**：只用事先收集好的数据训练策略，不和环境在线交互。这个工作把**扩散模型**当成策略表示，在 D4RL 上取得了很好的效果，而且训练比原来的扩散策略快很多。

**问题与动机（约 30 秒）**  
论文基于 **Diffusion-QL** 做**两个改进**：一是**训练效率**——Diffusion-QL 训练慢（如 5 天）、步数多难训，EDP 用 action approximation 和 DPM-Solver 大幅加速（约 25×）；二是**算法兼容性**——Diffusion-QL 只支持 TD3 式算法，像 CRR、IQL 这类用加权 MLE 的算法需要 log π(a|s)，扩散策略给不出，EDP 用 ELBO 近似 log π 使它们可用。这样既能长步数稳定训练，又能接多种离线 RL 算法。

**方法（约 1 分钟）**  
策略是**以状态为条件的扩散模型**：给定观测，对噪声反复去噪得到动作。训练有两个损失：**扩散损失**（去噪 MSE，拟合数据动作分布）和**引导损失**（由所选算法决定——TD3 直接最大化 Q，CRR/IQL 用优势加权 log π，所以 EDP 要提供 log π 的近似）。  
采样时用 DDPM、DDIM 或 DPM-Solver；**论文所有结果用 EAS**：采多个动作再用 Q 选一个（代码里 ensemble），评估更稳、分数更好。

**实现与实验（约 30 秒）**  
我用的官方 Jax 实现，数据用 D4RL，支持 TD3、CRR、IQL 三种 loss。流程是：读入离线数据 → 每步采样 batch → 更新 Q（和 V，若是 IQL）与扩散策略（扩散 loss + guide loss）→ 定期在环境里评估。复现时我跑了 walker2d、hopper 等环境，和论文里的设置对齐。

**收尾（约 20 秒）**  
整体上，这个项目把**生成模型**和**离线 RL** 结合得比较紧，既涉及扩散模型的前向/反向过程和采样，又涉及离线 RL 的 value 估计和策略约束，对理解「用生成模型做决策」很有帮助。

---

## 六、常见面试问题与回答要点

1. **为什么用扩散模型而不是 GAN 或 VAE 做策略？**  
   扩散训练稳定、模式覆盖好，不易塌缩；且可以自然做多步去噪，表达多模态动作分布。VAE 容易后验塌缩，GAN 训练不稳定。

2. **EDP 为什么能训 1000 步？Diffusion-QL 为什么难？**  
   EDP 通过更好的方差调度、训练技巧（如梯度裁剪、学习率衰减）和可能的架构设计，使长步数训练稳定；Diffusion-QL 可能对步数敏感、易爆炸或过拟合，且训练时每次都要跑完整反向链，步数一多就极慢。

3. **Guide loss 的作用？**  
   扩散 loss 主要拟合行为克隆；guide loss 用 Q（或优势）把策略往「高回报/高 Q」方向拉，在离线设定下避免纯模仿次优数据。

3.1 **Value loss 里的 Q 和 guide loss 里的 Q 是同一个吗？谁更新 Q？**  
   是**同一个** Q 网络（双 Q）。**Value loss** 用 TD target 拟合并**更新** Q；**guide loss** 只用 current Q(s, pred_astart) 的取值，梯度**只更新策略（噪声网络）**，不更新 Q。TD target 用 **target** π' 和 **target** Q' 算 $y$，体现延迟更新；guide 用 **current** Q。见 **2.3**、**3.3**。

4. **离线 RL 的 distribution shift 问题，EDP 如何缓解？**  
   通过使用保守或约束的离线算法（如 IQL 不直接最大化 Q、CRR 用优势加权），以及用扩散拟合数据分布而非任意外推，减轻对 OOD 动作的依赖。

5. **DDPM、DDIM、DPM 的区别？**  
   DDPM 随机、满 T 步；DDIM 确定性、可跳步；DPM-Solver 是少步 ODE 求解器，用更少步数达到相近质量，推理更快。直观上可以理解为：DDIM 通过把 DDPM 的随机马尔可夫链重参数化为确定性的 ODE 轨迹，在有限步数下“跳步”前进，用更少的去噪步数逼近满步 DDPM 的效果，因此实现生成加速，但在极大步数极限下仍以 DDPM 的完整链为最精细的参考。

6. **代码里 DiffusionQL 和 TD3/CRR/IQL 的关系？**  
   DiffusionQL 是「扩散策略 + 不同 RL 目标」的框架：同一套扩散 policy 和 Q（及可选的 V），通过换 `loss_type` 切换 value loss 和 guide loss 的计算方式（TD3 用 Q 最大化、CRR 用优势加权 MLE、IQL 用 expectile V + AWR）。**注意**：每次训练只选一种 `loss_type`，不是同时算三种。

7. **最终成果是通过什么展示的？训练速度是成果吗？**  
   **最终成果**：通过 D4RL 的 **normalized return**（标准化分数）展示，在论文的表格和图中体现 SOTA 性能。**训练速度提升（25×）**是技术贡献，说明 EDP 比原版 Diffusion-QL 更高效，但不是评估策略好坏的指标。评估时看的是策略在环境中的表现（normalized return），不是训练时间。

8. **EAS / ensemble 是什么？论文为什么说用 Q 选？**  
   **EAS（Energy-based Action Selection）**：评估时从策略先采 N 个动作，再用 Q 网络打分，按权重 $\propto e^{Q(s,a)}$ 选一个（或选 Q 最大的）。论文 4.5 节提出并写明 **“All results will be reported based on EAS”**，所以**用 Q 选**能降低评估方差、得到更好的分数；代码里 `ensemble_act`、`ddpmensemble_act`、`dpmensemble_act` 即 EAS 的实现。ensemble 本身不是 EDP 发明，是已有做法，EDP 采用并据此报结果。

9. **action approximation 是什么？**  
   训练时算 guide loss 需要「策略给出的动作」；若每次都用完整反向链采样，太慢。**action approximation** = 用当前扩散步的**预测 $x_0$（pred_astart）** 当作该动作，只过一遍噪声预测网络即可，从而加速训练。对应代码里 `use_pred_astart=True`（默认）。

10. **概率论里 p、f、F 的关系？**  
    在连续型里，**p 就是概率密度函数 f**（只是记号不同）；**分布**既可用分布函数 F 描述，也可用密度 p 描述。期望里的 p 指密度；对密度积分得到概率。

11. **DDPM 和 VAE / ELBO 有什么关系？训练时的 MSE 和 ELBO 有关吗？**  
    可以把 DDPM 看成一种**特殊的变分生成模型**，和 VAE 很像：它同样想最大化 $\log p_\theta(x_0)$，只是把潜变量换成一条 Markov 噪声链 $x_{1:T}$，前向 $q(x_{1:T}|x_0)$ 是固定的「变分分布」，反向 $p_\theta(x_{0:T})$ 要学。由于 $\log p_\theta(x_0)$ 难算，用 ELBO 做下界；在高斯 + 固定方差 + 噪声参数化的设定下，ELBO 的 KL 项刚好化成逐步拟合噪声的 MSE，所以**训练看起来只是 MSE，本质是在最大化 ELBO**。生成阶段不用 ELBO，只是用学好的反向分布逐步采样。详见 **2.1、2.4、2.5**。

12. **既然 CRR/IQL 需要 log π，为啥不直接用 TD3？为什么需要三种？**  
    EDP 支持三种算法**任选其一**训练，不是同时算三种。不同域上表现不同（如 Antmaze 常用 IQL），支持三种是为了泛用性与按任务选算法；可只用 TD3（`loss_type=TD3`）。见 **1.3 简短澄清**、**2.2 三种 loss 类型**。

13. **TD3 不是 policy-based 吗？「用 TD3 做损失」什么意思？**  
    TD3 是 **actor-critic**（策略更新最大化 Q，不用 log π）。「用 TD3 做损失」= 用其 value loss（双 Q + MSE）+ guide loss（−λQ）当训练目标。见 **1.3 简短澄清**、**2.2.2 TD3**。

14. **确定性 vs 随机策略、DPG 是什么？为何 TD3 不需要 log π？**  
    **确定性策略**：$a=\pi(s)$，网络直接输出动作均值；**随机策略**：$\pi(a|s)$ 输出分布参数，再采样。TD3 用确定性策略，走 **DPG（Deterministic Policy Gradient）** 路径：$\nabla J = \mathbb{E}[\nabla_a Q \cdot \nabla_\theta \pi]$，只需 $\nabla Q$ 和 $\nabla\pi$，**不需要** log π。CRR/IQL 用随机策略的加权 MLE，必须 log π。见 **2.2.1**、**Day4 零、理论基础**。

15. **CRR 和 IQL 的权重 β 与 τ 一样吗？**  
    形式相同：$\exp(\cdot/\text{温度})$，都是逆温度。区别：CRR 用 $A/\beta$，A 为 Q 减策略下 Q 的期望；IQL 用 $(Q-V)/\tau$，V 来自 expectile 回归。β 与 τ 越小越尖锐（更保守）、越大越平滑。见 **2.2.1**。

16. **action_dist 的 std 为什么可学？DDPM 反向方差不是固定的吗？**  
    DDPM 反向链每步的**方差**由 schedule 固定。**action_dist** 是**单独建的对角高斯**（均值=pred_astart），用于近似 log π，与扩散链无关；其 **std** 是「近似分布的宽度」，可设成固定（如 `fixed_std`）或由 **GaussianPolicy 的 log_stds** 学习。见 **2.2.1 action_dist 与 policy_dist**。

17. **policy_tgt_update 怎么定？Q target 和 policy target 更新频率为何不同？**  
    `policy_tgt_update = (_total_steps>1000) and (_total_steps % policy_tgt_freq==0)`（默认每 5 步）。Q target 每步软更新（算 value loss 要跟上）；policy target 每 N 步更新以降低 TD 目标方差、稳定训练。见 **2.2.1 Target 更新**。

18. **训练循环在 trainer 还是 dql？**  
    循环（epoch/step、取 batch、评估、日志）在 **trainer.py**；**dql.py** 只实现**单步**逻辑（给定 batch 算 loss、梯度、更新），算法相关故放在 agent 中。见 **2.2.1 训练分工**。

---

### 公式与概念所在位置索引（你要的“都在哪些文档/代码里”）

| 概念 / 公式 | 在文档里的位置 | 在代码里的位置 |
|-------------|----------------|----------------|
| **EDP 三方面改进**（训练效率 + 生成效率 + 兼容性 / ELBO 近似 log π） | 本文档 **1.2 DQL 整体思路**、**1.3 核心问题与动机** | — |
| **前向加噪** $q(x_t\|x_0)$、$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | 本文档 **2.1**；**2.4 训练阶段**表格 | `diffusion/diffusion.py`：`q_sample`、`q_mean_variance` |
| **p 与 q 的区别**（q=前向+后验闭式，p=学到的反向） | 本文档 **2.1** 末尾「代码里 p 和 q 的区别」 | `diffusion/diffusion.py`：`q_sample`/`q_posterior_mean_variance` vs `p_mean_variance` |
| **扩散训练目标** $\mathcal{L}_{\text{diff}} = \mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$（MSE） | 本文档 **2.1**；**2.4、2.5** | `diffusion/diffusion.py`：`training_losses`；`dql.py`：`get_diff_loss`、`policy.loss` |
| **ELBO** 与「KL → MSE」的关系 | 本文档 **2.4 训练阶段**表格（目标、替代目标、ELBO 组成、为何变成 MSE）；**2.5** | `diffusion/diffusion.py`：`training_losses`（MSE 即 KL 在固定方差下的形式） |
| **Action approximation**（用 pred_astart 当动作、不跑完整链） | 本文档 **3.5 评估部分**最后一段；**常见问题 9** | `diffusion/dql.py`：`use_pred_astart`、`get_diff_terms` 里 `pred_astart`、`p_mean_variance(..., pred_xstart)` |
| **DPM-Solver**（生成加速；训练时 value loss 算 $a'$ 也可用 DPM） | 本文档 **2.1** 采样方式；**2.6**、**3.3**、**3.4、3.5**；一周计划总表 | `diffusion/nets.py`：`dpm_sample`；`diffusion/dql.py`：`get_value_loss` 内 `policy.apply`；`diffusion/dpm_solver.py` |
| **双损失** $\mathcal{L}_{\text{policy}} = \mathcal{L}_{\text{diff}} + \lambda_{\text{guide}} \mathcal{L}_{\text{guide}}$ | 本文档 **2.3**；**3.3** | `diffusion/dql.py`：`_train_step_td3` 里 `policy_loss = diff_loss + guide_coef * guide_loss` |
| **联合训练**（为什么 RL 要用扩散模型的训练结果、为什么可以用"还没训练好"的网络输出） | 本文档 **2.3.1** | `diffusion/dql.py`：`_train_step_td3` 里 `policy_loss_fn`（diff_loss 和 guide_loss 同时优化） |
| **Guide loss（TD3）** $-\lambda Q(s,\hat{a})$ | 本文档 **2.2.2**、**2.3**、**3.3** | `diffusion/dql.py`：`_train_step_td3` 里 `policy_loss_fn`、`policy_loss = -lmbda * q.mean()` |
| **Q 网络 / 双 Q / target**（TD3） | 本文档 **2.2.2**、**3.3** | `diffusion/dql.py`：`get_value_loss`、`update_target_network` |
| **CRR / IQL**（加权 MLE、需 log π；权重 λ/ω、β 与 τ；IQL 的 expectile、AWR） | 本文档 **2.2.1**、**常见问题 15** | `diffusion/dql.py`：`_train_step_crr`、`_train_step_iql`、`get_value_loss`（IQL 的 V） |
| **DPG**（Deterministic Policy Gradient）、确定性 vs 随机策略 | 本文档 **2.2.1**、**常见问题 14**；Day4 **零、理论基础** | — |
| **ELBO 近似 log π**（兼容 CRR/IQL） | 本文档 **1.2 改进二**、**2.2.1** | `diffusion/dql.py`：CRR/IQL 里 `policy_dist`、`action_dist.log_prob` 等 |
| **action_dist / policy_dist**（高斯近似、log_stds、与 DDPM 方差区别） | 本文档 **2.2.1**、**常见问题 16**；Day4 **七、补充** | `diffusion/nets.py`：`GaussianPolicy`；`dql.py`：`get_diff_terms`、CRR/IQL 更新 |
| **policy_tgt_update、Q/policy target 更新频率** | 本文档 **2.2.1**、**常见问题 17** | `diffusion/dql.py`：`_train_step_*` 末尾、调用处 `policy_tgt_update` |
| **训练循环 vs 单步更新**（trainer / dql 分工） | 本文档 **2.2.1**、**常见问题 18** | `diffusion/trainer.py`：循环；`diffusion/dql.py`：`train`、`_train_step_*` |

- **文档**：若无特别说明，“本文档”指 `docs/EDP_复试梳理与演讲稿.md`；“一周计划”指 `docs/EDP_一周学习与复现计划.md`。  
- **代码**：均为项目根目录下的相对路径（如 `diffusion/dql.py`）。

---

## 八、实验数据结论速查（论文里「用数据说了什么」）

> 学到现在容易只记得理论，这里把论文**用实验/表格/图得出的结论**单独拎出来，便于回答「这篇论文设计实验数据的结论都是哪些」。

### 8.1 用了 AP 快多少（Fig. 2，有数字）

- **设定**：在 walker2d-medium-expert-v2 上跑 10k 步训练、10k 步采样，算 IPS（iterations per second）和 SPS（steps per second）。以 **DQL (JAX)** 为 baseline。
- **论文给的数值**（原文 Fig. 2 caption）：
  - 训练 IPS：DQL 4.66，DQL(JAX) 22.30，EDP w/o DPM 50.94，**EDP w/o AP 38.4**，**EDP 116.21**。
  - 采样 SPS：18.67，123.70，123.06，411.0，411.79（同上顺序）。
- **结论（论文正文）**：
  - **Action approximation**：相对 DQL (JAX)，带来 **2.3× 训练加速**。论文正文还写了「3.3× 采样加速」——见下方 **8.1.1 澄清**。
  - **DPM-Solver**：在已有 AP 的基础上，再带来 **约 2.3× 训练加速**；**采样加速来自 DPM**（EDP w/o DPM 用 DDPM 约 123 SPS，EDP / EDP w/o AP 用 DPM 约 411 SPS，即约 3.3×）。
  - 整体：EDP 相对 DQL (JAX) 约 **5× 训练加速**（116/22 ≈ 5.3）；相对原版 Diffusion-QL（PyTorch）是 **5 天 → 5 小时**（约 25×，含 JAX 实现与 AP+DPM）。
- **注意**：Fig. 2 只做了**速度**对比，没有「用 AP vs 不用 AP」的 **normalized return** 对比；「without performance loss」在正文没有对应 ablation 表。

**8.1.1 为什么「采样」也会加速？AP 会加速采样吗？**  
- **AP 只改训练阶段**：用 pred_astart 当动作算 guide loss，**不**改「训练结束后评估时怎么从策略里采动作」。评估时还是：给定 s，用 DDPM 或 DPM-Solver 从噪声去噪得到 a，和是否用 AP 无关。  
- **Fig. 2 的采样 SPS**：顺序为 DQL / DQL(JAX) / **EDP w/o DPM** / **EDP w/o AP** / **EDP**，对应 18.67 / 123.70 / **123.06** / **411.0** / **411.79**。  
  - EDP w/o DPM（有 AP、用 DDPM 采样）≈ 123；EDP w/o AP（无 AP、用 DPM 采样）≈ 411；EDP（有 AP + DPM）≈ 411。  
  - 所以 **3.3× 采样加速（123→411）来自 DPM-Solver**（少步 ODE 替代满 T 步 DDPM），**不是**来自 AP。论文把「2.3× 训练、3.3× 采样」都写进同一句容易误解；按数据：**AP 只加速训练，采样加速是 DPM 的贡献**。

### 8.2 论文用实验数据得出的主要结论（按章节）

下面都是**有表格/图/实验设置**支撑的结论，不是纯理论陈述。

| 结论 | 依据 | 内容摘要 |
|------|------|----------|
| **效率：EDP 训练/采样远快于 Diffusion-QL** | Fig. 1 左、Fig. 2、正文 5.1 | 5 天→5 小时（locomotion）；Fig. 2 给出 IPS/SPS 与 AP、DPM 的消融（见 8.1）。 |
| **效率：AP 与 DPM 各自贡献** | Fig. 2 | AP 约 2.3× 训练、3.3× 采样；DPM 再约 2.3× 训练；DQL(JAX) 比原版 DQL 约 5×。 |
| **性能：EDP 不比 DQL/Diffusion-QL 差，且能用大 K** | Tab. 1 | 同域统一超参下，EDP（K=1000）与 DQL(JAX)、Diffusion-QL 比：locomotion 89.4/89.4/**90.3**；antmaze 75.4/78.6/**77.9**；adroit 68.3/66.5/**89.1**；kitchen 71.6/83.0/**80.5**。说明大 K 可训且能提升或持平。 |
| **泛化：EDP 可插拔 TD3/CRR/IQL，且优于 FF 策略** | Tab. 2 | 四域（locomotion, kitchen, adroit, antmaze）上 EDP+TD3 / EDP+CRR / EDP+IQL 与同算法 FF 对比；EDP 列平均分均不低于 FF，多任务明显更高（如 locomotion 平均 85.5，antmaze 73.4）。 |
| **评估指标：RAT 比 OMS 更稳** | Fig. 3、5.3 | walker2d/hopper 上 OMS 与 RAT 接近；antmaze 上训练可先好后崩，OMS 会虚高，RAT 更可靠，故主表用 RAT。 |
| **EAS：扩散策略需要，FF 不需要** | 5.3 + Appendix | 对 TD3+BC 用 EAS 无提升；对扩散策略用 EAS 有提升；EAS 动作数 1→200 在 9 个 locomotion 里 8 个单调升，主实验取 10 做折中（Fig. 4）。 |
| **DPM 步数：约 15 步后收益变平** | Fig. 5 | DPM-Solver 的 model call 数 3～30 在 locomotion 上试，性能随步数增后约 15 步后趋于平稳，主实验用 15。 |
| **大 K 更好** | Tab. 1 及 4.4 | EDP 用 K=1000；Diffusion-QL 用 5～100；Tab. 1 中 EDP 多域优于或持平 DQL，说明「大 K 可训」带来性能收益。 |

### 8.3 和「理论」的区分（面试可答）

- **理论**：action approximation 的公式、为什么梯度方向对、ELBO 近似 log π、DPM-Solver 的 ODE 形式、联合训练等——这些是**方法设计与推导**。
- **实验结论**：上面 8.1、8.2 都是**用 D4RL 的跑数、表格、图**得出的——例如「快多少」看 Fig. 2，「好不好」看 Tab. 1/Tab. 2，「EAS/DPM 步数怎么选」看 Fig. 4/5。
- 论文**没有**在正文里做的实验：**「用 AP vs 不用 AP」的 normalized return 对照**（只有速度对照）；**边际效益递减**的收敛曲线。

---

## 九、何恺明团队扩散/去噪相关工作与本项目对比

> 何恺明团队 2024–2025 年在扩散与去噪生成模型上有若干工作，和本项目的「扩散策略 + 少步采样加速」有可比性，便于回答「有没有相关前沿」「和 EDP 的区别」类问题。

### 9.1 何恺明团队相关论文简要总结

| 工作 | 会议/年份 | 核心思想 | 与「去噪链/加速」的关系 |
|------|------------|----------|--------------------------|
| **Mean Flows for One-step Generative Modeling** | NeurIPS 2025 (Oral) | 用**平均速度**刻画流场，替代 Flow Matching 的瞬时速度；**单步生成**（1-NFE），无需蒸馏/课程。ImageNet 256×256 上 FID 3.43。 | **一步生成**，彻底去掉多步去噪链；与 EDP 的「少步 DPM」形成鲜明对比。 |
| **pMF (Pixel Mean Flow)** | 2026.3 等 | *One-step Image Generation in Pixel Space*：**纯像素空间**一步生成，无 VAE/潜空间，去掉多步采样。ImageNet 256×256 一步 FID **2.22**，可视为 Mean Flows 在像素空间的强化版。 | 与 Mean Flows 同属「一步、无多步链」；强调像素空间、非潜空间，算力低于 GAN。 |
| **Drifting Models（漂移模型）** | 2026.2 等 | *Generative Modeling via Drifting*：把**分布演化锁在训练阶段**，推理仅**单次前向**；**潜空间**一步生成，FID **1.54**，显著超越此前一步方法。 | 另一条一步生成路线：训练时学漂移，推理一步；与 pMF 互补（潜空间 vs 像素空间）。 |
| **Is Noise Conditioning Necessary for Denoising Generative Models?** | ICML 2025 | 质疑**噪声水平条件**（如 σₜ / timestep）是否必要；无噪声条件的 uEDM 在 CIFAR-10 上 FID 2.23，接近有条件 EDM。 | 从「是否必须给模型 timestep」角度简化扩散，不改步数；本项目仍显式给 t。 |
| **Back to Basics: Let Denoising Generative Models Denoise** (JiT) | — | 基于流形假设：**直接预测洁净数据**比预测噪声/速度更高效；Just image Transformers，无 VAE/Tokenizer。 | 改预测目标（x₀ 而非 ε），与单步/少步可结合；本项目仍是预测 ε + 多步/少步采样。 |

- **Mean Flows** 与「一步加速去噪链」最直接相关：把多步去噪**压缩成一步**，用平均速度的数学恒等式指导训练，从零训练即可 1-NFE 生成。**pMF** 在其基础上做到纯像素空间、更高指标（2.22）；**Drifting** 则走潜空间、把复杂度压到训练阶段，推理一步（1.54）。
- **Is Noise Conditioning Necessary** 关注**条件输入**的简化；**JiT** 关注**预测目标**的简化（预测干净数据）。二者都可与「少步/一步」结合，但论文重点不在步数。

### 9.2 与本项目（EDP）的对比

| 维度 | 何恺明团队（以 Mean Flows 为主） | 本项目（EDP） |
|------|----------------------------------|----------------|
| **应用场景** | 图像生成（ImageNet 等） | 离线强化学习（扩散策略，D4RL） |
| **训练目标** | 平均速度 / 单步流匹配，为 1-NFE 设计 | DDPM 式噪声预测 MSE + TD3/CRR/IQL 引导；训练目标与多步扩散一致 |
| **推理步数** | **1 步**（单次前向） | **多步**：DDPM 满 T 步 / DDIM 跳步 / **DPM-Solver 约 15 步** |
| **加速方式** | 从算法/目标上设计成**一步生成**，无需多步链 | 保留多步去噪链，用 **DPM-Solver（高阶 ODE 求解器）** 在 15 步内近似解出，少步 ≈ 原多步质量 |
| **时间/噪声条件** | Mean Flows 不依赖 timestep 条件；Noise Conditioning 论文则质疑 σₜ 必要性 | **显式 timestep**：训练与推理都喂 t（或离散步下标），DPM 用连续 t 插值 |
| **预测目标** | Mean Flows：平均速度；JiT：直接预测 x₀ | **预测 ε（噪声）**，用 `p_mean_variance` 等可推出 pred_x0，但主输出仍是 ε |

**一句话区别**：  
何恺明团队（尤其 Mean Flows）是**「重设计」**——为单步生成重新设计训练目标与流场，从而**取消去噪链**；EDP 是**「链不变、少步近似」**——训练仍是标准扩散（ε 预测 + timestep），推理时用 DPM-Solver 等**在少步内近似整条去噪链**，兼顾与现有扩散策略、CRR/IQL 的兼容性。

### 9.3 面试可答要点

- **有相关**：何恺明团队 Mean Flows（NeurIPS 2025）做**单步生成**，可视为「扩散模型一步加速去噪链」的代表工作；后续 **pMF、Drifting**（2026）在同一方向上继续推进（像素空间一步 2.22、潜空间一步 1.54）；Noise Conditioning、JiT 则从条件与预测目标上简化扩散。
- **和 EDP 的区别**：EDP 不改训练目标、不改成单步，而是用 **DPM-Solver 少步 ODE** 在 15 步左右完成从 x_T 到 x₀，属于**采样器加速**；Mean Flows / pMF / Drifting 是**模型与目标**为 1-NFE 设计，属于**架构/目标层面的加速**。二者可互补：未来若有「单步扩散策略」类工作，可类比 Mean Flows 系列；若继续用多步扩散策略，则 EDP 的 DPM 少步方案仍适用。

### 9.4 本节总结：与本项目的关系与区别

- **时间线**：2025 年 Mean Flows（单步 FID 3.43）、Is Noise Conditioning Necessary、JiT 等；2026 年 pMF（像素空间一步 2.22）、Drifting（潜空间一步 1.54），整体是「扩散→一步生成」的持续演进。
- **和本项目的关系**：同属「减少扩散推理成本」的大方向——何恺明团队从**图像生成**侧做一步/少步生成；本项目（EDP）从**离线 RL 扩散策略**侧用 DPM-Solver 做少步采样，都是对多步去噪链的加速或替代。
- **和本项目的区别**：何恺明系列是**重设计**——为 1-NFE 重新设计训练目标与流场（平均速度、漂移等），**取消**多步去噪链，场景是图像生成；EDP 是**链保留、少步近似**——训练仍是标准扩散（预测 ε + timestep），推理用 DPM 约 15 步近似整条链，场景是强化学习策略生成，并保持与 TD3/CRR/IQL 的兼容。面试时可概括为：**同一大方向（加速扩散），不同场景（图像 vs RL）与不同手段（一步重设计 vs 少步采样器）。**

---

## 三、复现与实验结果（面试可直接引用）

### 3.1 复现实验设置

- **平台与环境**：AutoDL 云实例 + 1×RTX 4090，CUDA 12.1 镜像，在 conda 环境中通过 `jax[cuda12]` 安装 GPU 版 JAX，`jax.devices()` 显示 `CudaDevice(0)`。  
- **物理引擎与数据集**：MuJoCo 2.1.0 + `mujoco_py`，配置 `LD_LIBRARY_PATH` 与 `MUJOCO_PY_MUJOCO_PATH`；安装 D4RL + `mjrl`，成功注册 `walker2d-medium-replay-v2` 等 MuJoCo 环境，并从 Berkeley 官方地址下载 D4RL HDF5 数据集。  
- **训练配置**：严格沿用原仓库默认超参，在 `walker2d-medium-replay-v2` 上使用 **TD3 损失（`loss_type=TD3`）**，运行 **2000 epoch × 1000 step/epoch** 的离线训练；评估采用 D4RL 的 normalized return。

### 3.2 关键结果与论文对比

- **复现结果**：在 Walker2d-medium-replay-v2 上得到  
  - `best_normalized_return ≈ 0.934`  
  - `average_normalizd_return ≈ 0.93`。  
- **与论文/官方结果对齐情况**：EDP/Diffusion-QL 在该任务上公开的 normalized return 约 **0.93 左右**（不同 seed/实现会有轻微波动），本次复现结果与之高度一致，说明：  
  - 扩散策略 + TD3 引导的实现**功能正确、数值稳定**；  
  - 「D4RL 数据 → n-step 预处理 → DiffusionQL 训练 → D4RL normalized return」整条 pipeline 在 GPU 环境下**可以端到端复现论文级别表现**。

### 3.3 工程排坑与收获（可以在问答中展开）

- **JAX + CUDA 版本问题**：一开始在 CUDA 11.8 镜像下，`jax/jaxlib` 与 `jax_cuda_releases` 的 CUDA wheel 版本不匹配，GPU 加速始终回退到 CPU；最终通过切换到 CUDA 12 镜像 + `pip install "jax[cuda12]"`，避免了旧 wheel 覆盖问题，使训练全程跑在 GPU 上。  
- **MuJoCo / D4RL 依赖链**：依次解决了缺少 MuJoCo 库、OSMesa 头文件（`GL/osmesa.h`）、`mjrl` 缺失导致 D4RL MuJoCo 环境未注册等问题，完成从系统依赖到 Python 包的闭环配置。  
- **端到端验证思路**：先在本机和 AutoDL 上用 2 epoch × 100 step 的小实验验证数据流与 JAX/GPU，再跑 2000 epoch 正式实验，并对比 normalized return 与论文结果；这一套流程本身就体现了「先理解代码结构，再用小实验验证，再做完整复现」的工程思路，可在复试中重点强调。

---

## 七、后续可追问的方向

- 某一段代码的具体含义（如 `get_diff_terms`、`p_mean_variance`、expectile 实现）；  
- 如何改超参或换 env 做小实验；  
- 如何画曲线、对比 TD3/CRR/IQL 或 DDPM/DPM；  
- 扩散的数学推导（前向/反向、$\bar\alpha_t$ 的用法）；  
- 离线 RL 的 IQL、CQL、CRR 的区别与适用场景；  
- 评估时 `act_method`、`post`、`recent_returns` 等含义（见 3.5 节）。

你可以随时说「我想问 EDP 的 xxx」或「解释一下 dql 里 CRR 的 guide loss」，我会按代码和理论一起答。
