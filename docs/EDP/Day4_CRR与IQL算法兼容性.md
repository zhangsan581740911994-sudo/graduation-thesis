# Day 4：CRR / IQL + 扩散策略（算法兼容性）

> **目标**：理解 CRR、IQL 在 EDP 框架里的角色，能说出它们与 TD3 的差异，以及为什么需要 ELBO 近似 log π。  
> **对应理论**：CRR（优势加权 log π）、IQL（expectile V、AWR 加权）；EDP 改进三「算法兼容性」+ ELBO 近似 log π。  
> **说明**：Day 3 的 TD3 不需要策略的 log π(a|s)；CRR 和 IQL 的 guide loss 是**加权 MLE**，必须能算 log π。扩散策略只能采样、给不出闭式 log π，EDP 用 **action_dist / ELBO 近似** 提供 tractable 的 log π，使扩散策略能接 CRR/IQL。

---

### Day 4 提问自测（先尝试自己回答）

> 建议：真正答题时只看这部分，不要往下翻到「参考答案」。

1. 用 Q 更新策略有哪两种方式？哪种需要 log π(a|s)，哪种不需要？
2. 为什么扩散策略「原生」不兼容 CRR/IQL？EDP 如何解决？
3. CRR 的 guide loss 核心思想是什么？权重 λ 从哪里来？
4. IQL 和 TD3 在「如何用 Q 更新策略」上有什么本质区别？
5. IQL 里 V、Q、policy 分别怎么更新？expectile 回归在干什么？
6. `get_diff_terms` 里的 `action_dist`、`log_prob` 在 CRR/IQL 里用在哪儿？
7. CRR 的 `crr_weight_mode`（elbo / mle / 其他）分别对应哪种 log π 近似？
8. 一句话概括：TD3、CRR、IQL 各自的策略更新思想。
9. 为何离线设定下 CRR/IQL 比直接 DDPG/PG 更稳？EDP 为何要支持多种算法？
10. IQL 里 V(s) 的定义是什么？和「Q 对 a 的期望」有何不同？为什么学 V 时不需要对动作采样？
11. TD3/CRR 里更新 Q 的损失在代码里叫什么？IQL 里更新 V、更新 Q 的损失分别叫什么？为什么要区分名字？
12. IQL 的 TD target 用 V(s') 还是 min Q(s',a')？双 Q 在 IQL 里还用在什么地方？
13. CRR 里 replicated_obs 用来做什么？IQL 需要 replicated_obs 吗？为什么？
14. action_dist 和 policy_dist 分别是什么？在 TD3、CRR、IQL 中哪些网络/分布会被梯度更新？
15. CRR 的 guide loss 形式是什么？crr_weight_mode 的 mle 和 elbo 分别如何得到 log π？为什么说「分叉的是 log π 的来源」？
16. policy_dist 在 CRR 的 MLE 模式下学的是什么？实际被哪个损失更新？MLE 模式和 ELBO 模式分别更新几套网络（双 Q、扩散、policy_dist）？
17. 训练扩散时的「噪声 MSE」和 ELBO 是什么关系？CRR 用 elbo 近似 log π 时的 MSE 和训练噪声网络时的 MSE 是同一个吗？TD3 不用 log π，为什么说「所有 DDPM 训练都经历 ELBO」？

---

## 零、理论基础：策略梯度与两种实现路径

> 以下内容为 Day4 核心概念的理论铺垫，便于理解「为什么有两种用 Q 更新策略的方式」。

### 0.1 共同的根：同一目标 J(π)

两类方法都在优化同一量：**策略的期望回报** $J(\pi) = \mathbb{E}_{s,a\sim\pi}[Q(s,a)]$。  
策略梯度定理给出其梯度：$\nabla_\theta J = \mathbb{E}[Q(s,a) \cdot \nabla_\theta \log \pi(a|s)]$（随机策略），或 $\nabla_\theta J = \mathbb{E}[\nabla_a Q \cdot \nabla_\theta \pi]$（确定性策略，DPG）。

### 0.2 策略梯度定理的两种等价写法

- **R/G 形式**：$\nabla J \propto \mathbb{E}[\nabla\log\pi \cdot G_t]$，其中 $G_t$ 为回报，REINFORCE 常用，方差大。  
- **Q 形式**：$\nabla J \propto \mathbb{E}[\nabla\log\pi \cdot Q(s,a)]$，因 $Q(s,a)=\mathbb{E}[G_t|s,a]$，二者在期望下等价；Actor-Critic 用 Q 降低方差。

### 0.3 梯度公式的两因子与两种「攻击」路径

梯度可视为两因子乘积：① **Q**（或由 Q 得到的权重）、② **∇log π**（或与 log π 相关的项）。

| 路径 | 代表算法 | 利用的因子 | 计算方式 |
|------|----------|------------|----------|
| **从 Q 侧** | TD3 | ① Q | $\nabla Q(s,\pi(s))$，梯度经 Q 传到策略，**不需要** log π |
| **从 log π 侧** | CRR/IQL | ② log π | $\omega \cdot \nabla\log\pi(a|s)$，ω 由 Q/优势得到，**必须** log π |

### 0.4 为何 TD3 可以「只考虑 Q」？DPG 与确定性策略

- **随机策略梯度**：$\nabla J = \mathbb{E}[Q \cdot \nabla\log\pi]$，需要 log π。  
- **确定性策略梯度（DPG）**：对 $a=\pi(s)$，$\nabla J = \mathbb{E}[\nabla_a Q|_{a=\pi(s)} \cdot \nabla_\theta \pi]$，**不需要** log π（确定性策略无分布）。  
- TD3 用确定性策略，走 DPG 路径，故不涉及 log π。

### 0.5 确定性 vs 随机策略（简要）

|  | 确定性策略 | 随机策略 |
|--|------------|----------|
| **定义** | $a=\pi(s)$，一一映射 | $\pi(a|s)$，输出分布 |
| **实现** | 网络直接输出动作均值 | 输出分布参数（如 $\mu$, $\sigma$），再采样 |
| **典型场景** | 连续动作（DPG/TD3/DDPG） | 离散动作、需随机探索 |
| **DPG** | Deterministic Policy Gradient，Silver et al. 2014 |

### 0.6 权重 ω 的计算与 CRR / IQL 的 β、τ

| 算法 | 权重公式 | 说明 |
|------|----------|------|
| **CRR** | $\lambda \propto \exp(A/\beta)$ 或 $\lambda=1[A>0]$ | $A = Q(s,a) - \mathbb{E}_\pi Q$（优势） |
| **IQL** | $\omega \propto \exp((Q-V)/\tau)$ | $Q-V$ 也是「优势」，但 V 来自 expectile，更保守 |

**β 与 τ**：形式类似，都是**逆温度**——越小越尖锐（更保守）、越大越平滑。区别在于优势中「减数」的来源：CRR 的减数为当前策略下 Q 的期望（常由蒙特卡洛估计）；IQL 的减数为 V，来自 expectile 回归，不取 max，更防过估计。

---

## 一、核心概念速览

### 1.1 两种用 Q 更新策略的方式

| 方式 | 代表算法 | 策略更新形式 | 是否需要 log π |
|------|----------|--------------|----------------|
| **A：直接最大化 Q** | TD3 | $\mathrm{loss} = -\lambda Q(s, \hat{a})$，梯度通过 Q 反传到策略 | **不需要** log $\pi$ |
| **B：加权 MLE** | CRR、IQL | $\mathrm{loss} = -\mathbb{E}[\omega \cdot \log \pi(a|s)]$，权重 $\omega$ 由 Q/优势得到 | **需要** $\log \pi(a|s)$ |

- 扩散策略**只能采样**（从 $x_T$ 去噪得 $a$），给不出闭式 **$\log \pi(a|s)$**，故与方式 B 不兼容。
- EDP 的**算法兼容性**：在 pred_astart 等处用 **ELBO 或高斯近似** 构造可算的 log π，供 CRR/IQL 使用。

### 1.2 CRR（Critic Regularized Regression）

- **思想**：用**优势** $A(s,a) = Q(s,a) - \mathbb{E}_\pi Q(s,a')$ 得到权重，对 **$\log \pi(a|s)$** 做**加权极大似然**，让策略更倾向高优势动作。
- **权重**：$\lambda \propto \exp(A/\beta)$ 或 $\lambda = \mathbf{1}[A>0]$（heaviside）等，再 stop_gradient，只让策略侧吃梯度。
- **Guide loss**：$-\mathbb{E}[\lambda \cdot \log\pi(a|s)]$。需要 **log $\pi$** 来自某处；EDP 用 action_dist.log_prob 或 $-\mathtt{ts\_weights}\times\mathtt{mse}$（ELBO 近似）。

### 1.3 IQL（Implicit Q-Learning）

- **思想**：不直接最大化 Q，而是学 **V**（状态价值）、**Q**，再用 **AWR**（Advantage-Weighted Regression）更新策略。
- **V**：用 **expectile 回归** 拟合「Q 的某个分位」，使 V 不偏向 max，更保守。
- **Q**：用 **TD target**，$\mathrm{target} = r + \gamma V(s')$（用 V 而不是 $Q(s',a')$），进一步减过估计。
- **Policy**：权重 $\omega \propto \exp((Q-V)/\tau)$，对 $\log \pi(a|s)$ 加权 MLE。同样需要 **log $\pi$**；EDP 用 action_dist.log_prob。

### 1.4 EDP 如何提供 log π

- **来源**：`get_diff_terms` 里在得到 pred_astart 后，用 **policy_dist** 在 pred_astart 上构造 **action_dist**（如高斯），再算 **log_prob = action_dist.log_prob(sample)**，写入 `terms['log_p']`。
- **用途**：CRR/IQL 的 guide loss 用 `action_dist.log_prob(actions)` 或与 terms 里 mse/ts_weights 组合的 ELBO 式 log π。
- **注意**：TD3 不用 action_dist / log π，只用到 terms 里的 diff_loss 和 pred_astart；Day 3 已学。CRR/IQL 才依赖这段（见 `dql.py` 的 `get_diff_terms`，约 284–332 行，action_dist/log_p 在约 317–330 行）。

---

## 二、代码结构：get_diff_terms 中的 action_dist 与 log π

### 2.1 位置与作用

**位置**：`diffusion/dql.py` 的 `get_diff_terms`（约 284–332 行；action_dist、log_p 在约 317–330 行）。

**作用**：用 pred_astart 构造**可算 log π 的动作分布**（高斯等），供 CRR/IQL 的 guide loss 使用；TD3 不依赖。

**关键代码逻辑**：

```python
# 用 pred_astart 构造动作分布（policy_dist 是高斯等）
action_dist = self.policy_dist.apply(params['policy_dist'], pred_astart)
sample = pred_astart  # 或从 action_dist 采样（若 sample_logp）
log_prob = action_dist.log_prob(sample)
terms['sample'] = sample
terms['action_dist'] = action_dist
terms['log_p'] = log_prob
```

- **action_dist**：以 pred_astart 为均值（或参数）的分布，可调用 `.log_prob(a)` 得到 log π(a|s) 的近似。
- **crr_weight_mode** 为 `mle` 时，直接用 `action_dist.log_prob(actions)`；为 `elbo` 时用 `-terms['ts_weights'] * terms['mse']` 作为 log π 的 ELBO 近似。

---

## 三、代码结构：_train_step_crr

### 3.1 整体流程

1. **Value loss**：与 TD3 相同，`get_value_loss(batch)`，双 Q + target，更新 qf1/qf2。
2. **Policy loss**：
   - 调 `diff_loss_fn` 得到 diff_loss、terms（含 action_dist、log_p 等）。
   - 构造**优势**：对 batch 中 $(s,a)$ 算 $Q(s,a)$；再采样一批 vf_actions，算 $V = \mathbb{E}[\min(Q_1,Q_2)]$ 的估计，$\mathrm{adv} = Q(s,a) - V$。
   - 把优势变成**权重** $\lambda$：$\exp(\mathrm{adv}/\beta)$ 或 heaviside(adv)，再 stop_gradient。
   - **Guide loss** $= -\mathrm{mean}(\lambda \cdot \log \pi(a|s))$，其中 $\log \pi$ 来自 terms（mle 用 action_dist.log_prob(actions)，elbo 用 $-\mathtt{ts\_weights}\times\mathtt{mse}$）。
   - **Policy loss** = diff_coef×diff_loss + guide_coef×guide_loss；对 policy 和 policy_dist 都回传梯度（value_and_multi_grad(..., 2)）。

### 3.2 关键代码位置

| 内容 | 位置（约） |
|------|------------|
| value_loss_fn、diff_loss_fn | 481–482 |
| 优势 adv = q_pred − avg(v) | 509–517 |
| 权重 λ（exp/heaviside） | 519–529 |
| log π（elbo/mle/其他） | 531–547 |
| guide_loss = −mean(λ·log_prob) | 549 |
| policy_loss、梯度更新、target 更新 | 551–594 |

---

## 四、代码结构：_train_step_iql

### 4.1 三步拆解

IQL 一步训练拆成**三个 loss**，依次更新不同参数：

| 步骤 | Loss | 更新谁 | 含义 |
|------|------|--------|------|
| 1 | **value_loss**（expectile） | **V** | V 拟合 Q 的某个分位（expectile 回归），不取 max，更保守 |
| 2 | **critic_loss**（TD） | **Q1, Q2** | TD target = r + γ·V(s')，用 V 做 bootstrap，更新 Q |
| 3 | **policy_loss**（AWR + diff） | **policy, policy_dist** | 权重 $\omega = \exp((Q-V)/\tau)$，$\mathrm{guide\_loss} = -\mathrm{mean}(\omega \cdot \log \pi(a|s))$，再加 diff_loss |

### 4.2 Expectile 回归（V 的更新）

- **目标**：让 $V(s)$ 逼近「$Q(s,a)$ 的 $\tau$-expectile」（不是 max，也不是 mean），$\tau$ 为超参（如 0.7）。
- **形式**：$\mathrm{diff} = Q(s,a) - V(s)$；若 $\mathrm{diff}>0$ 权重大、$\mathrm{diff}<0$ 权重小，用 expectile_weight 加权 $(\mathrm{diff})^2$ 的 mean。
- **代码**：`value_loss` 函数（约 634–651 行），更新 `train_states['vf']`。V 的定义、不对称损失公式及与采样的关系见 **7.8**。

### 4.3 Critic loss（Q 的更新，就是CRR和TD3里的value_loss）

- **Target**：$\mathrm{td\_target} = r + (1-\mathrm{done})\cdot\gamma\cdot V(s')$（注意是 **V** 不是 $Q(s',a')$），减轻过估计。
- **Loss**：$\mathrm{MSE}(q_1^{\mathrm{pred}}, \mathrm{td\_target}) + \mathrm{MSE}(q_2^{\mathrm{pred}}, \mathrm{td\_target})$。
- **代码**：`critic_loss` 函数（约 653–672 行），更新 qf1、qf2。

### 4.4 Policy loss（AWR + diff）

- **权重**：$\mathrm{exp\_a} = \exp((Q(s,a) - V(s)) / \tau)$，可 clip 或 softmax 归一化。
- **Guide loss**：$\mathrm{awr\_loss} = -\mathrm{mean}(\mathrm{exp\_a} \cdot \log \pi(a|s))$，$\log \pi$ 来自 action_dist.log_prob(actions)。
- **Policy loss** $= \mathtt{diff\_coef}\times\mathtt{diff\_loss} + \mathtt{guide\_coef}\times\mathtt{guide\_loss}$；对 policy 和 policy_dist 都更新。
- **代码**：`policy_loss` 函数（约 674–700 行），更新 policy、policy_dist。

### 4.5 执行顺序（代码中）

先更新 V → 再更新 policy（依赖 V、Q、terms）→ 再更新 Q（critic_loss）。target 的软更新在最后（与 TD3 类似）。

---

## 五、三种算法对比与 EDP 兼容性小结

| 项目 | TD3 | CRR | IQL |
|------|-----|-----|-----|
| **策略更新方式** | 直接最大化 Q，$-\lambda Q(s,\hat{a})$ | 优势加权 MLE，$-\mathbb{E}[\lambda\cdot\log \pi]$ | AWR 加权 MLE，$-\mathbb{E}[\omega\cdot\log \pi]$，$\omega\propto\exp((Q-V)/\tau)$ |
| **是否需要 log π** | 否 | 是 | 是 |
| **Guide 权重来源** | λ 为标量缩放（与 Q 尺度有关） | λ 来自优势 A=Q−E[Q]，E[Q] 常蒙特卡洛估计 | ω 来自 Q−V；**V 由 expectile 回归训出**（expectile 只训 V，不直接算 ω） |
| **Value 部分** | 双 Q + target，TD target 用 Q' | 同 TD3 | V 用 expectile；Q 用 TD target = r+γV(s') |
| **log π 来源（EDP）** | 不用 | mle：action_dist.log_prob；elbo：−ts_weights×mse（同一次前向的噪声 MSE 乘时间步权重） | action_dist.log_prob（本实现用 mle） |
| **ELBO 使用** | 仅训扩散时（噪声 MSE 来自 ELBO） | 训扩散一次；若 crr_weight_mode=elbo 则 log π 近似再用一次 | 仅训扩散时；log π 用高斯闭式，不再用 ELBO |
| **代码入口** | _train_step_td3 | _train_step_crr | _train_step_iql |

- **CRR 与 IQL 的 guide loss**：**形式相同**（都是 −𝔼[权重·log π]），区别在**权重的来源**。CRR 的 λ 来自优势 A=Q−E[Q]；IQL 的 ω∝exp((Q−V)/τ)，其中 **V** 由 **expectile 回归**单独训出（value_loss），再用于 TD target 和 guide 权重 (Q−V)，即 expectile 不直接算 ω，而是先得到 V，再用 (Q−V) 算 ω。
- **非确定性策略 DQL（CRR/IQL）**：若在 guide 里**不用** ELBO 近似 log π，则用 **MLE**（对角高斯闭式 action_dist.log_prob）；若用 ELBO，则 log π≈−ts_weights×mse（即同一次前向的噪声 MSE 乘时间步权重）。

### 5.1 各损失函数对照（区别一览）

| 损失名称 | 含义 | 更新谁 | 使用算法 | 公式/来源简要 |
|----------|------|--------|----------|----------------|
| **diff_loss** | 扩散训练损失 | **policy**（噪声网络） | TD3、CRR、IQL 共用 | 噪声 MSE：$\|\varepsilon - \text{model\_output}\|^2$，来自 ELBO 化简；同一次前向得到 terms['loss']/mse。 |
| **guide_loss** | 用 Q/优势引导策略 | **policy**（+ **policy_dist** 仅 MLE 模式） | 三种都有，形式不同 | **TD3**：$-\lambda Q(s,\hat{a})$，无 log π。**CRR**：$-\mathbb{E}[\lambda\cdot\log\pi]$，λ=A 相关，log π 来自 mle 或 elbo。**IQL**：$-\mathbb{E}[\omega\cdot\log\pi]$，ω∝exp((Q−V)/τ)，log π 来自 action_dist。 |
| **policy_loss** | 策略侧总损失 | policy、policy_dist（CRR/IQL 且 MLE 时） | 三种都有 | $\mathtt{diff\_coef}\times\mathtt{diff\_loss} + \mathtt{guide\_coef}\times\mathtt{guide\_loss}$。对 policy_dist 只有 guide_loss 有梯度。 |
| **value_loss**（TD3/CRR） | Q 网络的 TD 损失 | **qf1、qf2** | TD3、CRR | `get_value_loss`：MSE(Q(s,a), r+γ·min(Q'(s',a')))，双 Q + target。 |
| **value_loss**（IQL） | V 网络的 expectile 损失 | **vf** | 仅 IQL | 不对称平方损失：diff=Q−V，权重 τ/(1−τ)，最小化 $\mathbb{E}[\mathrm{weight}\cdot(\mathrm{diff})^2]$，只更新 V。 |
| **critic_loss**（IQL） | Q 网络的 TD 损失 | **qf1、qf2** | 仅 IQL | MSE(Q(s,a), r+γ·V(s'))，target 用 V 不用 min Q'。 |

- **区别小结**：**diff_loss** 三种算法相同（噪声 MSE）。**guide_loss** 三种形式不同：TD3 不涉及 log π；CRR/IQL 都是「权重×log π」，权重与 log π 来源见上表。**value_loss** 在 TD3/CRR 里指更新 Q 的损失，在 IQL 里指更新 **V** 的损失；IQL 另用 **critic_loss** 更新 Q。

**EDP 兼容性**：通过 `get_diff_terms` 统一提供 **action_dist** 与 **log π 近似**，TD3 只用 diff_loss+pred_astart；CRR/IQL 额外用 action_dist / log_p（或 ELBO 式），实现「同一套扩散策略 + 三种算法任选其一」。

---

## 六、当天需要能做到

- 用 **1–2 句话**说明：为何 CRR/IQL 需要 log π，EDP 如何提供（action_dist、ELBO 近似）。
- 用 **3–5 句话**概括：TD3、CRR、IQL 各自的策略更新思想及在离线设定下为何更稳。
- 能指出：`get_diff_terms` 里 action_dist / log_p 在 CRR/IQL 中的使用位置；`_train_step_crr` 与 `_train_step_iql` 的大致流程（CRR：value + policy；IQL：value(V) + critic(Q) + policy）。

---

### Day 4 自测参考答案（折叠，答题时不要展开）

<details>
<summary>点击展开参考答案</summary>

1. **两种用 Q 更新策略的方式**  
   - **方式 A**：直接最大化 Q，如 TD3 的 −λQ(s,â)，**不需要** log π。  
   - **方式 B**：加权 MLE，目标形如 −𝔼[ω·log π(a|s)]，权重 ω 由 Q/优势得到，**必须能算** log π(a|s)。

2. **扩散为何不兼容 CRR/IQL；EDP 如何解决**  
   - 扩散策略只能采样、给不出闭式 log π(a|s)，而 CRR/IQL 的 guide 是加权 MLE，需要 log π。  
   - EDP 在 `get_diff_terms` 里用 **policy_dist** 在 pred_astart 上构造 **action_dist**（如高斯），得到 **log_prob**，或用 **−ts_weights×mse** 作 ELBO 近似，供 CRR/IQL 使用。

3. **CRR 的 guide loss 与权重**  
   - 核心：**优势加权行为克隆**，−𝔼[λ·log π(a|s)]。  
   - 权重 λ：由优势 A = Q(s,a) − 𝔼_π Q 得到，如 λ ∝ exp(A/β) 或 heaviside(A)；λ 做 stop_gradient，只对策略反传。

4. **IQL 与 TD3 在用 Q 更新策略上的区别**  
   - TD3：直接用 −λQ(s,â) 更新策略，不用 log π。  
   - IQL：不直接最大化 Q；先学 V（expectile）、Q（TD with V）；策略用 **AWR**：ω ∝ exp((Q−V)/τ)，对 log π(a|s) 加权 MLE。

5. **IQL 里 V、Q、policy 的更新**  
   - **V**：expectile 回归，让 V(s) 拟合 Q(s,a) 的 τ-分位（不是 max）。  
   - **Q**：TD，target = r + γ·V(s')，MSE 更新 Q1、Q2。  
   - **Policy**：AWR，guide_loss = −mean(exp((Q−V)/τ) · log π(a|s))，再加 diff_loss。

6. **action_dist / log_prob 在 CRR/IQL 里的用法**  
   - 在 `get_diff_terms` 里得到 `terms['action_dist']`、`terms['log_p']`。  
   - CRR：guide_loss 用 λ·log π，log π 取 action_dist.log_prob(actions) 或 ELBO 式（−ts_weights×mse）。  
   - IQL：guide_loss 用 exp((Q−V)/τ)·log π，log π = action_dist.log_prob(actions)。

7. **crr_weight_mode**  
   - **elbo**：log π 近似为 −terms['ts_weights']×terms['mse']（扩散 ELBO 形式）。  
   - **mle**：log π = action_dist.log_prob(actions)（高斯等 tractable 分布）。  
   - 其他：可用多采样 MSE 等形式近似（见代码分支）。

8. **一句话概括三种算法的策略更新**  
   - **TD3**：策略更新 = 最大化 Q(s, â)。  
   - **CRR**：策略更新 = 优势加权 log π(a|s)（高优势动作权重大）。  
   - **IQL**：策略更新 = (Q−V)/τ 加权 log π(a|s)（AWR），更保守。

9. **离线设定下为何更稳；EDP 为何支持多种算法**  
   - 离线数据固定，直接 DDPG/PG 易过估计、分布偏移；CRR/IQL 用加权 MLE 或 expectile+V 更保守，减轻过估计、贴近数据。  
   - EDP 支持多种算法：不同域（如 Antmaze、Kitchen）上表现不同，可依任务选 TD3/CRR/IQL；**算法兼容性**是论文贡献之一，Diffusion-QL 不接 CRR/IQL，EDP 用 log π 近似接上。

10. **IQL 里 V(s) 的定义与为何不需采样**  
   - V(s) 是「Q(s,a) 在 (s,a) 数据分布下的 **τ-expectile**」，**不是** $E_a[Q(s,a)]$（不是对 a 的期望）。  
   - 实现上：V 网络通过最小化 expectile 损失（value_loss）训练，用 batch 里已有的 (s,a) 和 target Q 即可，不需要再采样动作。V(s) 只依赖 s，是 V 网络的输出。

11. **value loss / critic loss 命名**  
   - TD3、CRR 里更新 Q 的损失都叫 **value loss**（`get_value_loss`）。  
   - IQL 里更新 **V** 的叫 **value_loss**，更新 **Q** 的叫 **critic_loss**。因为 IQL 多了一个 V 网络，需要区分「训 V 的损失」和「训 Q 的损失」。

12. **IQL 的 TD target 与双 Q 的用法**  
   - TD target 用 **V(s')**，不用 min Q(s',a')。  
   - 双 Q 仍用于：① 训练 V 时，expectile 的目标 Q(s,a) 取 **min(Q1,Q2)**（target Q）；② 策略权重 (Q−V) 里的 Q 也取 **min(Q1,Q2)**，减轻过估计。

13. **replicated_obs 与 IQL**  
   - CRR 里 **replicated_obs** 用来估计 $V(s)=\mathbb{E}_a Q(s,a)$：对每个 s 复制多份，与采样的 vf_actions 配对算 Q，再对采样维平均得 V，用于优势 A = Q−V。  
   - **IQL 不需要**：IQL 有独立 V 网络，用 expectile 回归学 V，V(s) 直接由网络输出，不需要对动作采样再平均。

14. **action_dist、policy_dist 与谁被更新**  
   - **action_dist**：以 pred_astart 为均值的对角高斯（std 来自 policy_dist 的 log_stds），用于算 log π 近似；是临时构造的分布，不单独「更新」。  
   - **policy_dist**：即 GaussianPolicy，输出该高斯的参数；**CRR/IQL 会更新** policy_dist（因 guide loss 依赖 log_stds），**TD3 不更新**。  
   - 此外：TD3 更新 qf1、qf2、policy；CRR 多更新 policy_dist；IQL 更新 vf、qf1、qf2、policy、policy_dist。

15. **CRR guide loss 与 crr_weight_mode**  
   - Guide loss **形式**始终是 $-\mathbb{E}[\lambda\cdot\log\pi(a|s)]$（加权 MLE）。分叉的是**如何得到 log π 的数值**：**mle** = 用对角高斯闭式 action_dist.log_prob(actions)；**elbo** = 用扩散 ELBO 项 $-\mathtt{ts\_weights}\times\mathtt{mse}$。两种模式都是同一 guide loss 形式，只是 log π 的**近似来源**不同（高斯 vs ELBO），不是「policy 是不是 MLE」的分支。  
16. **policy_dist 学什么、被谁更新；MLE vs ELBO 网络数**  
   - **MLE 模式**：policy_dist 提供**方差**（log_stds），均值 = pred_astart（扩散给）。名义上与 policy 共用 policy_loss = diff_loss + guide_loss，但 **diff_loss 与 policy_dist 无关**，故 policy_dist **实际只被 guide_loss 更新**。  
   - **网络数**：**MLE** = 双 Q + 扩散（diff_loss+guide_loss）+ **policy_dist**（只 guide_loss）= 三套；**ELBO** = 双 Q + 扩散（diff_loss+guide_loss，log π 用 −ts_weights×mse，不经过 policy_dist）= 两套。

17. **ELBO、噪声 MSE、CRR guide loss 的 MSE**  
   - **训练扩散**：要最大化 $\log p_\theta(x_0)$ 不可算（整条链边际）→ 用 **ELBO** 作为可优化目标；在 DDPM 固定方差下 ELBO 化简为**噪声 MSE**（target=真实噪声 ε，model_output=网络预测），故**训噪声网络的损失就是来自 ELBO 的 MSE**。**生成时**不算 ELBO，只用前向后验 $q(x_{t-1}|x_t,x_0)$ 的均值公式，把 $x_0$ 换成网络预测的 $\hat{x}_0$。  
   - **CRR elbo 近似 log π**：用的就是**同一次前向**里的 terms['mse'] 和 terms['ts_weights']，即 **同一个噪声 MSE**（真实 ε vs 预测），再按时间步加权：$\log\pi \approx -\mathtt{ts\_weights}\times\mathtt{mse}$。时间步权重 $w_t = \beta_t/(2(1-\bar\alpha_t)\alpha_t)$，再归一化（见 `diffusion.py` 约 264、267 行）。  
   - **两种「难算」**：① **训扩散**：$\log p_\theta(x_0)$ 难算 → 用 ELBO → MSE，**TD3/CRR/IQL 都一样**，训噪声网络都用这项。② **log π(a|s)**：只有 CRR/IQL 的 guide 需要，才用 ELBO 或高斯近似；**TD3 不需要 log π**，故不会用到「为 log π 的 ELBO」。所以「所有 DDPM 训练都经历 ELBO」指**扩散的训练目标**由 ELBO 推导而来；TD3 只是不在 **guide loss** 里用 log π 的 ELBO 近似。

</details>

---

## 七、补充：代码与概念澄清

> 以下为对话中反复出现的概念与代码对应关系，便于和 Day4 正文对照。

### 7.0 对角高斯的 log 概率公式：是数学公式，不是为 RL 专门推的

- **对角高斯**是**概率论里的标准分布**：各维独立的多维高斯，均值向量 $\boldsymbol{\mu}$、每维标准差 $\sigma_d$，密度可写成各维一维高斯的乘积。
- **它的 log 概率**是**纯数学结论**，和强化学习无关：对向量 $\mathbf{a} = (a_1,\ldots,a_d)$，
  $$
  \log p(\mathbf{a}) = \sum_{d=1}^{D} \left[ -\log \sigma_d - \frac{1}{2}\log(2\pi) - \frac{(a_d - \mu_d)^2}{2\sigma_d^2} \right]
  $$
  这就是「对角高斯分布」的 log 密度公式，任何教材里都有，**不是为 RL 推导的**。
- **为什么可以有一个「log π」的公式？**  
  - 在 CRR/IQL 里我们需要的是**策略** $\pi(a|s)$ 的 **log π(a|s)**。  
  - 扩散策略本身**给不出**闭式 log π，所以我们**用一个对角高斯分布去近似**「给定 s 时动作的分布」：令均值 $\boldsymbol{\mu} = \text{pred\_astart}$，方差由 policy_dist 的 log_stds 得到，把这个分布记作 **action_dist**。  
  - 于是我们**约定**：用这个高斯的 log 概率**当作** log π(a|s) 的近似，即「log π」≈ action_dist 的 log 概率。  
  - 所以：**log π 的公式 = 对角高斯的 log 概率公式**，后者是数学自带的；我们只是**选了这个数学对象来近似策略的 log 概率**，没有为 RL 单独推新公式。
- **「向量」与「给定 state」如何对应？** 对角高斯里的向量就是**动作空间**里的向量（数学上的向量 = 动作 a）。「在某个 state 下」体现在：高斯的**均值**取为 pred_astart(s)，依赖 s，所以每个 s 对应一个不同的高斯；对 batch 里的 (s,a)，我们在「s 对应的高斯」下算该分布对向量 a 的 log 密度，并把它当作 log π(a|s) 的近似。即：同一套数学公式，高斯的定义域 = 动作空间，参数依赖 s。

### 7.1 action_dist 与 policy_dist

- **action_dist**：单独构造的一个**对角高斯**，均值 = pred_astart，标准差 std 来自 **policy_dist** 的可学习参数 **log_stds**（`nets.py` 的 `GaussianPolicy`）。与 DDPM 反向链的方差（schedule 固定）无关，这里是**辅助近似 log π** 的分布。
- **policy_dist**：即 `GaussianPolicy`，输入 pred_astart 输出 `MultivariateNormalDiag(mean, std)`；std = exp(log_stds)。CRR/IQL 要对 policy_dist 做梯度更新（TD3 不更新 policy_dist）。

### 7.2 为何 π(x₀|s) 无闭式

- 扩散**单步** $p(x_{t-1}|x_t)$ 有闭式（高斯，均值用预测 x₀ 代入）。
- **$\pi(x_0|s)$** 是整条链的**边际**，需对 $x_1, \ldots, x_T$ 积分，无闭式；与是否做 action approximation 无关。CRR/IQL 需要 $\log \pi$ 来自「用 pred_astart 建的高斯近似」，不是来自完整链。

### 7.3 log π 的公式在代码哪

- **有 action_dist 时**：log_prob = action_dist.log_prob(·)，公式为高斯密度，在 **distrax** 库；我们只在 `nets.py` 构造分布、`dql.py` 调用。
- **ELBO 近似**：$\mathtt{log\_prob} = -\mathtt{terms['ts\_weights']}\times\mathtt{terms['mse']}$，**dql.py** CRR 分支约 532–533；ts_weights、mse 来自 **diffusion.py** 约 1140、1199 行（真实噪声 vs 预测噪声的 MSE）。

### 7.4 对角高斯、高维、diff_loss vs guide_loss

- **对角高斯**：各维独立，$\mathrm{std} = (\sigma_1,\ldots,\sigma_D)$；**高维** = 动作向量有多个分量（如 6 个关节），不是「同一状态多个动作」。
- **diff_loss**：对噪声 MSE 均匀最小化。**guide_loss（elbo 模式）**：用同一 MSE，但乘**优势权重 λ** 后最小化，高优势样本被压得更狠。

### 7.5 CRR 用 λ 不用 Q、replicated_obs、V 的估计（仅 CRR）

- **用 λ**：离线数据用**优势**加权更稳，只加强「比平均好」的动作；Q 已体现在 λ = f(A) 里。
- **replicated_obs**：CRR 需估计 $V(s)=\mathbb{E}_a Q(s,a)$ 作为优势的减数，无闭式，用蒙特卡洛：对每个 $s$ 采多份动作、算 $Q(s,a)$ 再平均。复制状态是为了把 $(s, a_j)$ 排成一批，一次前向算完 Q 再对采样维平均。**IQL** 有独立 V 网络，用 expectile 回归学 V，不需要此步骤。

### 7.6 梯度更新与 target 更新

- **TD3**：更新 qf1、qf2、policy（3 个）；**CRR**：多更新 policy_dist（4 个），因 guide loss 依赖 action_dist 的 log_stds。
- **Q target**：每步软更新，用于 Q 网络的 TD target（TD3/CRR 为 min Q′，IQL 为 V(s′)），需跟上当前 Q。
- **policy target**：仅当 `policy_tgt_update=True` 时软更新；`policy_tgt_update = (_total_steps>1000) and (_total_steps % policy_tgt_freq==0)`（默认每 5 步），以降低 TD 目标方差、稳定训练。

### 7.7 训练循环与 dql 分工

- **训练循环**（epoch/step 迭代、取 batch、评估、日志）在 **trainer.py**。
- **dql.py** 里是**单步更新**逻辑：给定 batch，算 loss、梯度、更新参数；算法相关（TD3/CRR/IQL）故放在 agent 中，trainer 只调 `agent.train(batch)`。

### 7.8 IQL 补充：expectile、V 定义、命名与双 Q

- **Expectile 回归与 V**：用**不对称平方损失**拟合 V(s)：$\mathrm{diff}=Q(s,a)-V(s)$，权重为 $\tau$（diff>0）或 $1-\tau$（diff≤0），$\mathcal{L}=\mathbb{E}[\mathrm{weight}\cdot(\mathrm{diff})^2]$。最小化后 V(s) 逼近 Q(s,a) 在数据分布下的 **τ-expectile**（τ 为超参，一般 >0.5，如 0.7）。**V 的定义**：IQL 里 V(s) **不是** $E_a[Q(s,a)]$，而是「Q(s,a) 在 (s,a) 数据分布下的 τ-expectile」。实现上：**V 网络**通过最小化上述 expectile 损失（即代码里的 value_loss）来训练；训练好后 V(s) 就是该网络的输出，故 V 只依赖 s，**学 V 时不需要对动作采样**。
- **value loss vs critic loss 命名**：TD3/CRR 里更新 Q 的损失在代码和文档里都叫 **value loss**（`get_value_loss`）；IQL 因多了一个 V 网络，把更新 **V** 的叫 **value_loss**、更新 **Q** 的叫 **critic_loss**，以区分。
- **双 Q 在 IQL 中的作用**：TD target 只用 V(s')，**不**用 min Q(s',a')。双 Q 仍用于：① 训练 V 时，expectile 的目标 Q(s,a) 取 **min(Q1,Q2)**（target Q）；② 策略权重 (Q−V) 里的 Q 也取 **min(Q1,Q2)**，减轻过估计。
- **AWR 与 guide loss 叫法**：IQL 的 guide 在论文里明确叫 **AWR**（Advantage-Weighted Regression），故代码注释写「AWR guide loss」；TD3 的 guide 无 log π，只写 guide loss；CRR 是优势加权 log π，通常也只写 guide loss。
- **value_loss 与 V 网络**：`value_loss(train_params)` 内 `v_pred = self.vf.apply(train_params['vf'], observations)` 即 V 网络前向；返回标量 **expectile_loss**，外层用 `value_and_multi_grad` 对其求梯度得到对 `vf` 的梯度，再 `apply_gradients` 更新 V 网络。
- **IQL vs CRR（权重含义）**：IQL 的 ω∝exp((Q−V)/τ) 对「比 V 好」的动作权大、对「比 V 差」的权小但非零；CRR 多用 A>0 或 λ 锐化，更偏「只加强比均值好的」。参考线：CRR 用均值，IQL 用 V（expectile）。

### 7.9 CRR 补充：crr_weight_mode、policy_dist 只训方差、MLE 与 ELBO 的网络分工

- **Guide loss 与 crr_weight_mode**：Guide loss 的**形式**始终是 $-\mathbb{E}[\lambda\cdot\log\pi(a|s)]$（加权 MLE）。**crr_weight_mode** 决定**如何得到 log π 的数值**：**mle** = 用对角高斯闭式 action_dist.log_prob(actions)（均值=pred_astart，方差=policy_dist 的 log_stds）；**elbo** = 用扩散 ELBO 项 $-\mathtt{ts\_weights}\times\mathtt{mse}$（来自同一次扩散前向）。分叉的是「log π 的近似来源」，不是「guide 是不是 MLE」——两种模式都是同一 guide loss 形式。
- **policy_dist：只训方差、只被 guide_loss 更新**：均值 = pred_astart（扩散给出）；方差由 **policy_dist** 的可学习参数 **log_stds** 提供（std=exp(log_stds)）。名义上 policy 与 policy_dist 共用 policy_loss = diff_loss + guide_loss，但 **diff_loss 与 policy_dist 无关**（只依赖扩散），故对 policy_dist 求梯度时只有 **guide_loss** 非零，即 **policy_dist 实际只被 guide_loss 更新**。方差没有「由均值算出来」的公式，是通过出现在 guide_loss 的计算图中被梯度优化学出来的。
- **MLE 三网络 vs ELBO 两网络**：**MLE 模式**：双 Q、扩散（policy，吃 diff_loss+guide_loss）、**policy_dist**（只吃 guide_loss，用于构造 action_dist 的 log π）。**ELBO 模式**：双 Q、扩散（policy，吃 diff_loss+guide_loss；log π 用 −ts_weights×mse，来自同一次扩散前向，**不经过 policy_dist**）。故 ELBO 下无 policy_dist 参与 guide loss，只有两套在更新的网络。

### 7.10 ELBO、DDPM、噪声 MSE 与 CRR guide loss 的 MSE（澄清）

- **训练扩散时为何和 ELBO 相关**：我们要最大化 $\log p_\theta(x_0)$（数据似然），对整条扩散链边际不可算，所以用 **ELBO** 作为可优化下界；在 DDPM、固定方差下，ELBO 化简为**噪声预测的 MSE**（target = 真实噪声 ε，model_output = 网络预测）。因此**训噪声网络的损失**就是这项 MSE，它**来自 ELBO**；不是「随便用 MLE 或其他损失」——扩散的**训练目标**就是由 ELBO 推导出来的。**生成时**不计算 ELBO，只用前向后验 $q(x_{t-1}|x_t,x_0)$ 的均值公式，把其中的 $x_0$ 换成网络预测的 $\hat{x}_0$。
- **CRR 用 elbo 时的 MSE 与噪声 MSE**：**是同一个量**。CRR 的 log π 近似 $\log\pi \approx -\mathtt{ts\_weights}\times\mathtt{mse}$ 里，`terms['mse']` 和 `terms['ts_weights']` 来自**同一次**扩散前向（`training_losses`），即 **(真实噪声 ε − 预测噪声)²** 的逐样本 MSE。相对「只取 mse」而言，CRR 只是再乘上**时间步权重**（并取负、当作 log π）。时间步权重公式：$w_t = \beta_t/(2(1-\bar\alpha_t)\alpha_t)$，代码中再用 `normalized_ts_weights = ws * num_timesteps / ws.sum()`（`diffusion.py` 约 264、267 行）。
- **两种「后验/难算」别搞混**：① **训练扩散**：$\log p_\theta(x_0)$ 难算 → 用 ELBO → 得到训练损失 = 噪声 MSE。**所有**用 DDPM 的算法（TD3/CRR/IQL）在训噪声网络时都用这项，即都「经历 ELBO」。② **策略的 log π(a|s)**：只有 CRR/IQL 的 guide loss 需要；扩散给不出闭式 → 才需要**用 ELBO 或高斯近似 log π**。TD3 是确定性策略、不用 log π，所以**从不需要**「为 log π 的 ELBO」；但**训扩散**时仍用 ELBO 导出的 MSE。

---

## 📖 参考文档与代码

- **理论**：`docs/EDP_复试梳理与演讲稿.md`  
  - **1.3 改进三（算法兼容性）**、**2.2.1 CRR 与 IQL**、**2.3 双损失**
- **代码**：`diffusion/dql.py`  
  - `get_diff_terms`（约 284–332 行；action_dist、log_p 约 317–330 行）  
  - `_train_step_crr`（约 477–620 行）  
  - `_train_step_iql`（约 622–772 行）

---

## 🚀 下一步（Day 5 预告）

Day 5 将学习：
- 数据流：D4RL/RLUP → Dataset.sample() → agent.train(batch)
- 跑通一个小实验（命令行、n_epochs、检查指标与日志）

**今天把 CRR/IQL 和 log π 接好，明天看数据与跑实验！**

---
