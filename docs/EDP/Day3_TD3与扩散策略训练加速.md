# Day 3：TD3 + 扩散策略（训练加速 + 兼容 TD3）

> **目标**：明确当 `loss_type=TD3` 时，一步训练里发生了什么，理解 action approximation 在哪里起作用。

---

## 📋 任务清单

### ✅ 前置准备
- [ ] 回顾 TD3 基础知识（双 Q、target network、policy 最大化 Q）→ 见下 **「前置：TD3 概念速览」**
- [ ] 理解扩散策略的基本训练流程（Day 2 内容）
- [ ] 准备好 `diffusion/dql.py` 和 `diffusion/diffusion.py` 两个文件
- [ ] 需要时查阅 **「附录：离线 RL 与 RL 概念补充」**（损失与 bootstrap、BC、RL 三分法、为何不用 Q*、马尔可夫性等）

### 📖 理论学习
- [ ] 理解 TD3 在离线 RL 中的角色
- [ ] 理解「扩散损失 + 引导损失」的双损失结构
- [ ] 理解 action approximation（pred_astart）如何加速训练
- [ ] 理解为什么可以用「还没训练好」的网络输出计算 guide loss

### 💻 代码阅读
- [ ] 阅读 `diffusion/dql.py` 的 `get_value_loss`（Q 网络更新）
- [ ] 阅读 `diffusion/dql.py` 的 `get_diff_loss` 和 `get_diff_terms`（扩散损失 + pred_astart）
- [ ] 阅读 `diffusion/dql.py` 的 `_train_step_td3`（完整训练一步）
- [ ] 理解 `update_target_network`（target 参数软更新）

### 🎯 实践验证
- [ ] 能口述/手写 `_train_step_td3` 的伪代码
- [ ] 能解释 action approximation 在哪里起作用
- [ ] 能说明与普通 TD3 相比，这里的最大区别是什么

---

### Day 3 提问自测（先尝试自己回答）

> 建议：真正答题时只看这部分，不要往下翻到「参考答案」。

1. TD3 的三块组件是什么？EDP 训练部分用的是「只拿双 Q 算 value loss」还是「完整的 TD3 AC 流程」？
2. value loss、diff loss、guide loss 分别更新哪个网络？各自的数学/代码位置在哪？
3. 为什么 value loss 里要让 Q1、Q2 **都**拟合同一个 target y，而不是只训练「取到 min 的那个 Q」？
4. value loss 和 guide loss 的作用区别是什么？「更高的 Q」在两者里分别指什么？
5. 什么是 action approximation？pred_astart 从哪里来、在代码里用在哪儿？
6. 当前网络和 target 网络各有几份参数？为什么软更新要调用三次 `update_target_network`？
7. `stop_gradient(tgt_q)` 和软更新分别做什么？为什么有了软更新还要 stop_gradient？
8. `update_target_network(main_params, target_params, tau)` 里，函数中的 x、y 分别对应 main 还是 target？返回值是哪套参数？
9. 软更新和硬更新的区别是什么？为什么这里用软更新？
10. 从「构造 loss」到「更新参数」再到「更新 target」，`_train_step_td3` 的完整顺序是什么？
11. 与普通 TD3 相比，EDP（TD3 + 扩散）的最大区别是什么？
12. TD3 原论文里有「guide loss」这个叫法吗？和 EDP 里的 guide loss 是什么关系？
13. 什么是「过估计」的正反馈？双 Q 取 min 如何缓解？
14. 为什么可以用「还没训练好」的 pred_astart 算 guide loss？（联合训练）

---

## 📌 前置：TD3 概念速览

（以下对应「任务清单」里的「回顾 TD3 基础知识」，先过一遍再看代码。）

### TD3 是什么、名字从哪来

- **TD3 = Twin Delayed DDPG**：**T**win（双 Q）、**D**elayed（策略延迟更新）。这里的 **TD 不是 Temporal Difference**，只是缩写；但 Critic 的更新公式**就是**时序差分（target = r + γ Q(s', a')，让 Q̂ 拟合 target）。
- **三块组件**：双 Q（取 min 减过估计）、target network（软更新稳目标）、policy 最大化 Q（guide loss = −λQ）。

### 当前网络 vs Target 网络

| 概念 | 含义 |
|------|------|
| **当前网络（current）** | 每步用梯度更新的 Q1/Q2 和 π |
| **Target 网络（target）** | 与当前网络结构相同、**单独存一份参数**，更新很慢，**只用来算目标**，不直接吃梯度 |

- **Target policy**：算 Q 的 target 时，用 **π_target(s')** 得到 next_action，再用 **Q_target(s', next_action)**。这样目标不会随当前 π/Q 每步乱动。
- 标准 AC（最简形式）可以没有 target；**DQN** 引入 target Q（硬更新：每 C 步复制）；**TD3** 用 target Q1/Q2 + target π，且常用**软更新**。

### 双 Q：为什么是「同一个 a'」、取 min？

- **动作只选一次**：next_action = π_target(s')，只有一个 a'。
- **两个 Q 对同一 a' 打分**：Q1_target(s', a')、Q2_target(s', a')，再 **target = r + γ · min(两者)**。
- **目的**：减轻**过估计**（两个估计取 min 更保守）。**不是**「选两个不同动作防 OOD」；防 OOD 靠 BC/约束策略等。

### 软更新 vs 硬更新、τ 是什么

- **硬更新（DQN 常用）**：每 C 步做 θ_tgt = θ_cur（直接复制），target 会突变。
- **软更新（TD3 常用）**：每步做 θ_tgt = τ·θ_cur + (1−τ)·θ_tgt。**τ 是超参**（如 0.005）；等价于 target 每步向 current 靠近 τ 比例的差距，**τ 小则目标更稳、跟得慢**。
- **为何用软更新**：target 平滑变化，TD 目标更稳定，训练更稳；TD3 论文与实现通常用软更新。
- Target 在**算 target 时**被用；在**更新完 current 后**再按上述方式更新 target。所谓「延迟更新 target」= **延迟/保守地更新 target 那套参数**（不跟 current 一起用梯度更新）。

### AC 流程一句话

**Actor 选动作（含用 target policy 算 target）→ Critic 用 target 算目标并更新 Q → Actor 用当前 Q 更新策略（最大化 Q）**。TD3 的「选动作」在算 target 那一步用的是 target policy，在更新 Actor 时用的是当前 policy 给出的动作（EDP 里是 pred_astart）。

### 符号：Q̂、Q* 与误差

- **Q***：真实动作价值（未知，无法算出）。
- **Q̂**：网络学出的 Q（如 Q1、Q2），是 Q* 的估计。
- **误差** = Q̂ − Q*；>0 为高估（overestimation），<0 为低估。误差来自有限采样（目标带方差）、网络未收敛/近似能力、以及 bootstrap（目标里含 Q̂，误差会传播）；在 Q-learning 里因用 **max**，高估更容易被选进目标，所以**系统性偏正**。

### 过估计（overestimation）与正反馈

- **正反馈链**：target $y = r + \gamma Q(s',a')$ 里若 Q 已高估 → y 偏大；value loss 是 $\hat{Q}(s,a)$ 与 y 的 MSE，会把 $\hat{Q}$ 往偏大的 y 拉 → 下一轮 target 更高 → 不断往高估方向走。
- **简单例子**：s' 有三个动作，真实 Q*(a1)=5，Q*(a2)=4，Q*(a3)=3；估计 Q̂(a1)=7，Q̂(a2)=3，Q̂(a3)=4。用 **max** 构造 target：y = r + γ·**7**，选到被高估的 a1，目标偏大即过估计。
- **双 Q**：对**同一** a' 用两个 Q 估计后取 **min**，压低 target，打断上述正反馈；**不是**选两个不同动作防 OOD。

### TD3 里的「guide」与 Actor 更新方式

- **TD3 没有「guide loss」这个名字**，但 Actor 的更新就是「最大化 Q(s, π(s))」，和 EDP 的 guide loss 同一思想；EDP 把这项单独叫 guide loss，以便和 diff loss 区分。
- **TD3 的 Actor 更新**不是用策略梯度定理（∇log π · A），而是**确定性策略梯度**：loss = −Q(s, π(s))，梯度为 −∇_a Q · ∇_θ π(s)，即 Q 对动作的梯度再通过 π 反传。

### 与标准 AC、DQN 的对比

| 项目 | 标准 AC（最简） | DQN | TD3 |
|------|-----------------|-----|-----|
| Target 网络 | 可没有 | 有 target Q（常**硬更新**） | 有 target Q1/Q2 + target π（常**软更新**） |
| 双 Q | 无 | 无（Double DQN 另有） | 有，对同一 a' 取 min |
| 算 target 时用的动作 | π(s') | argmax_a' Q_target(s',a') | π_target(s') |

---

## 🗺️ 学习思路

### 第一步：理解整体框架（10分钟）

**核心问题**：TD3 + 扩散策略的一步训练包含哪些部分？

**答案**：
1. **Value Loss（Q 网络更新）**：用双 Q + target network 更新 **Q1/Q2**（只更新 Q 网络参数）
2. **Diffusion Loss（扩散损失）**：让**策略（噪声网络）**拟合数据中的动作分布
3. **Guide Loss（引导损失）**：用 **Q 网络前向**得到的 **Q(s, pred_astart)** 构造（最大化 Q），引导策略偏向高价值动作；**不是** value loss 的输出
4. **Target 更新**：软更新 Q 和 policy 的 target 参数

**两个网络**：**噪声网络（策略）** 用 diff_loss + guide_loss 更新；**Q 网络（Critic）** 用 value loss 更新。guide loss 用的是 **Q 网络前向**的 Q 值，不是 value loss 这个损失本身。

**TD3 的完整 AC 流程是如何嵌入 EDP 的？（这一步要搞清的就是这个）**

- **TD3 的 AC 流程**：Critic 用 value loss + 双 Q + target 更新 Q；Actor 用 policy loss = −λQ(s, π(s)) 更新策略，即「Critic 评判动作 → Actor 朝高 Q 更新」。
- **嵌入方式**：
  - **Critic 部分**：**原样保留**。EDP 里仍然用 `get_value_loss` 算双 Q + target 的 TD 目标，更新 Q1/Q2，再软更新 target Q 和 target policy。和 TD3 完全一致。
  - **Actor 部分**：**策略换成扩散模型**，但「用 Q 引导 Actor」的方式仍是 TD3 的：用 **Q(s, 动作)** 构造 guide loss（最大化 Q）。区别只是「动作」从 **π(s)** 换成 **pred_astart**（扩散的单步预测），即 guide loss = −λQ(s, pred_astart)；**diff_loss** 是扩散策略自带的（拟合数据），而 TD3 里没有这一项。
- **对应关系**：  
  | TD3（AC） | EDP 里对应 |
  |-----------|-------------|
  | Critic 更新（value loss + 双 Q + target） | 同左，`get_value_loss` + 更新 qf1/qf2 + `update_target_network` |
  | Actor 更新：loss = −λQ(s, π(s)) | policy_loss = **diff_loss** + guide_coef × **(−λQ(s, pred_astart))** |
  | 动作来源：a = π(s) | 动作来源：a = pred_astart（action approximation） |

所以：**TD3 的 AC 骨架完整地插在 EDP 里**——Critic 那一套不变；Actor 从普通网络换成扩散策略，并在原有「最大化 Q」的基础上多了一项 diff_loss，让策略同时拟合数据。

**对应代码位置**：`diffusion/dql.py` 的 `_train_step_td3`（约 371-475 行）

### 第二步：深入理解 Value Loss（20分钟）

**核心问题**：Q 网络如何更新？双 Q 和 target network 如何工作？

**阅读顺序**：
1. 先看 `get_value_loss`（220-282 行）
   - 理解如何构造 target Q：`tgt_q = r + (1 - done) * discount * min(Q1_target, Q2_target)`
   - 理解如何用当前 Q1/Q2 估计当前 Q 值
   - 理解双 Q 取 min 的作用（减少过估计）

2. 在 `_train_step_td3` 中看如何使用 value_loss_fn（415-432 行）
   - 理解如何用 value_and_multi_grad 求梯度并 apply_gradients 更新 Q1/Q2

**关键点**：
- **双 Q**：用 Q1 和 Q2 的最小值作为 target，减少过估计；是对**同一个** next_action 用两个 Q 估计再取 min，不是选两个不同动作。**为什么两个 Q 都拟合同一 y**：min 只用来构造 target y；构造出 y 后，让 Q1、Q2 **都**学会预测这个 y，下次取 min 时两边都准。
- **Target Network**：在**算 target** 时用 `tgt_params`（π_target、Q1_target、Q2_target），保证目标稳定；算完 loss **先更新当前网络**（apply_gradients），再**软更新**三份 target。**两套参数**：当前网络（train_states）用梯度更新；target 网络（tgt_params）三份，只做软更新、不接梯度；引入 target 后**仍然要更新原本的当前网络**。
- **损失在干什么**：让 **Q̂(s,a)** 逼近 **target**（不是逼近 Q*）。target 里已经含 Q̂，是「用当前估计构造目标、再让估计去拟合」= **bootstrap**；Q* 不可得，所以必须这样设计。
- **TD 更新**：`tgt_q = rewards + (1 - dones) * discount * tgt_q`；`dones` 来自 batch（是否回合结束），(1-done) 在终止步不加上下一 Q。对 `tgt_q` 做 **stop_gradient**，梯度不反传到 target 参数。
- **τ**：软更新里的混合系数，**超参**，可自己设（默认 0.005）；τ 小则 target 跟得慢、目标更稳。

**对应理论**：见 `docs/EDP_复试梳理与演讲稿.md` 的 **2.2.2 TD3** 和 **3.3 训练一步**

### 第三步：深入理解 Diffusion Loss（30分钟）

**核心问题**：扩散损失如何计算？pred_astart 是什么？action approximation 在哪里？

**阅读顺序**：
1. 先看 `get_diff_terms`（284-333 行）
   - 理解如何随机采样时间步 `t`
   - 理解如何调用 `policy.loss` 得到扩散损失（terms 含 loss、model_output、x_t 等）
   - **重点**：理解 `pred_astart` 的计算
     - 若 `use_pred_astart=True`：用 `p_mean_variance(terms["model_output"], terms["x_t"], ts)["pred_xstart"]`（**即 action approximation**）
     - 若 `use_pred_astart=False`：需跑完整反向链采样动作（慢）

2. 再看 `get_diff_loss`（336-359 行）
   - 调 `get_diff_terms` 得到 `diff_loss = terms["loss"].mean()` 和 `pred_astart`
   - 返回值供 `policy_loss_fn` 使用

3. 在 `_train_step_td3` 的 `policy_loss_fn` 中看（398-399 行）：`diff_loss, _, _, pred_astart = diff_loss_fn(params, split_rng)`

**关键点**：
- **Action Approximation**：训练时用 `pred_astart`（单步预测的 x0）代替完整反向链采样，**大幅加速训练**
- **Diffusion Loss**：就是 Day 2 学的噪声 MSE，让策略学会去噪
- **pred_astart 的作用**：既用于计算 diff_loss，也用于后续的 guide loss

**对应理论**：见 `docs/EDP_复试梳理与演讲稿.md` 的 **1.3 改进一（训练效率）**

### 第四步：深入理解 Guide Loss（20分钟）

**核心问题**：如何用 Q 值引导策略？为什么可以用 pred_astart？

**阅读顺序**：
1. 在 `_train_step_td3` 的 `policy_loss_fn` 中看 guide loss（388-413 行）
   - 用 `pred_astart` 算 Q：`q = self.qf.apply(params[key], observations, pred_astart)`
   - 自适应系数：`lmbda = alpha / stop_gradient(|q|.mean())`
   - guide loss：`guide_loss = -lmbda * q.mean()`（最大化 Q）

2. 总 policy loss：`policy_loss = diff_loss + guide_coef * guide_loss`（diff 拟合数据，guide 偏向高 Q）

**关键点**：
- **Value loss 与 Guide loss 的区别**：**Value loss** 让 Q **预测准**（拟合 TD target），更新 Q 网络；**Guide loss** 让**策略选到的动作**在现有 Q 下得分高（−λ·mean(Q)），更新策略，不更新 Q。「更高的 Q」在 guide 里指「策略输出的动作对应更高的 Q(s,a)」，不是用 value loss 去把 Q 值调高。
- **Guide Loss 的作用**：用 **Q 网络前向**得到的 Q(s, pred_astart) 把策略往「高 Q」方向拉（不是 value loss 的输出）。**mean(Q)** 对 batch 求平均；**λ = alpha / mean(|Q|)** 随 Q 尺度自适应；alpha 为超参。
- **为什么可以用 pred_astart**：这是**联合训练**（策略与 Q 同时更新；**不是**「每步先训练再生成」），梯度方向正确，会逐步改进（见 **2.3.1 为什么需要联合训练**）。
- **与普通 TD3 的区别**：普通 TD3 直接用 policy 输出动作；这里用扩散模型的 `pred_astart` 当动作。

**对应理论**：见 `docs/EDP_复试梳理与演讲稿.md` 的 **2.3 EDP 的训练目标（双损失）** 和 **2.3.1 为什么需要联合训练**

### 第五步：理解 Target 更新（10分钟）

**核心问题**：什么时候更新 target？如何更新？

**阅读顺序**：
1. 看 `update_target_network`（36-39 行）
   - 软更新公式：新 target = `tau * main_params + (1 - tau) * target_params`；lambda 中 x=main（当前），y=target（旧），返回值写回 tgt_params

2. 在 `_train_step_td3` 中看何时更新 target（439-449 行）
   - Policy 的 target 按 `policy_tgt_update` 条件更新（非每步）
   - Q1、Q2 的 target 每步软更新

**关键点**：
- **软更新**：用 `tau`（默认 0.005）混合主网络和 target 网络参数，保证目标稳定
- **更新频率**：Q 每步更新，policy 按频率更新（默认每 5 步）

---

## 📚 代码阅读顺序

### 推荐阅读路径

> 以下行号以当前 `diffusion/dql.py` 为准，便于查找；若代码有增删，行号可能偏移。

```
1. diffusion/dql.py: _train_step_td3（371-475 行）
   ↓ 先看整体结构，理解三个主要部分
   
2. diffusion/dql.py: get_value_loss（220-282 行）
   ↓ 理解 Q 网络如何更新
   
3. diffusion/dql.py: get_diff_loss（336-359 行）
   ↓ 理解扩散损失如何计算
   
4. diffusion/dql.py: get_diff_terms（284-333 行）
   ↓ 深入理解 pred_astart 和 action approximation
   
5. _train_step_td3 内的 policy_loss_fn（388-413 行）
   ↓ 理解 guide loss 如何计算
   
6. diffusion/dql.py: update_target_network（36-39 行）
   ↓ 理解 target 更新机制
```

### 关键函数索引

| 函数名 | 行号（dql.py） | 作用 | 重点理解 |
|--------|----------------|------|----------|
| `_train_step_td3` | 371-475 | TD3 的一步训练 | **整体流程**：value loss → diff loss → guide loss → 更新参数 → 软更新 target |
| `get_value_loss` | 220-282 | 构造 Q 的 TD 损失函数 | **双 Q**：`min(Q1_target, Q2_target)`；**TD**：`r + γ * tgt_q`；**stop_gradient(tgt_q)** |
| `get_diff_loss` | 336-359 | 构造扩散损失函数 | 调 `get_diff_terms` 得到 `diff_loss` 和 `pred_astart`；噪声 MSE 在 `diffusion.py` 的 `training_losses` |
| `get_diff_terms` | 284-333 | 扩散步中间量 | **Action Approximation**：305-308 行 `pred_astart = p_mean_variance(...)["pred_xstart"]`；供 CRR/IQL 的 action_dist / log π 在 Day4 |
| `policy_loss_fn`（在 _train_step_td3 内） | 388-413 | TD3 的策略损失 | **Guide Loss**：400-412 行；**总损失**：`diff_loss + guide_coef * guide_loss` |
| `update_target_network` | 36-39 | 软更新 target 参数 | **公式**：`tau * main + (1-tau) * target`；对 policy、qf1、qf2 各调用一次（439-449 行） |

---

## 🔍 关键概念理解

### 1. Action Approximation（动作近似）

**问题**：训练时为什么用 `pred_astart` 而不是完整采样？

**答案**：
- **完整采样**：需要从 `x_T` 到 `x_0` 跑完整反向链（如 100 步），每步都要过网络，**极慢**
- **Action Approximation**：用当前步预测的 `pred_xstart`（单步预测的 x0）当动作，**只过一遍网络**，**约 25× 加速**

**代码位置**：`get_diff_terms` 内（305-308 行）
```python
if self.config.use_pred_astart:
    pred_astart = self.diffusion.p_mean_variance(
        terms["model_output"], terms["x_t"], ts
    )["pred_xstart"]  # ← 这就是 action approximation
else:
    # 需要跑完整反向链（慢）
    pred_astart = self.policy.apply(params['policy'], split_rng, observations)
```

**pred_astart 与 DDPM 的关系**：pred_astart 就是 DDPM 里「去噪网络在某一时间步 t 预测出的 x0」（同一套 `p_mean_variance` / pred_xstart），**不是新公式**；**新的是用法**：训练时用单步预测当动作近似，省掉完整 T 步采样，加速训练。

**为什么单步预测「方向」仍然对**：guide loss = −λQ(s, pred_astart) 对策略参数求梯度，梯度指向「让 pred_astart 对应的 Q 变大」，即策略改进方向正确；和「用梯度下降」无关，是 loss 设计决定了方向。

**单步 pred 不稳 vs 总迭代**：单步 pred 确实比完整 T 步采样更抖，但每步成本从 T 次前向降到 1 次；总迭代数即使略增，也远不到 T 倍，所以总训练时间仍大幅下降（论文约 25× 加速）。

**对应理论**：见 `docs/EDP_复试梳理与演讲稿.md` 的 **1.3 改进一（训练效率）**

### 2. 双损失结构

**问题**：为什么需要 `diff_loss + guide_loss`？

**答案**：
- **diff_loss**：让策略拟合数据中的动作分布（行为克隆）
- **guide_loss**：用 Q 值引导策略偏向高价值动作（强化学习）

两者结合：既拟合数据，又偏向高价值，实现离线 RL。

**代码位置**：`_train_step_td3` 的 `policy_loss_fn` 内（412 行），`policy_loss = diff_loss + self.config.guide_coef * guide_loss`

**对应理论**：见 `docs/EDP_复试梳理与演讲稿.md` 的 **2.3 EDP 的训练目标（双损失）**

### 3. 联合训练

**问题**：为什么可以用「还没训练好」的网络输出（pred_astart）计算 guide loss？

**答案**：
- **联合训练**指：每步同时更新**策略**（diff_loss + guide_loss）和 **Q**（value loss）；不是分阶段训练，也**不是**「每步先训练再生成」——训练循环内**不跑完整生成**，guide 用的动作是当前步前向的 pred_astart
- **梯度方向**：guide loss = −λQ(s, pred_astart)，对策略参数求导后，梯度指向「让 Q(s, pred_astart) 变大」，即策略改进方向正确；所以即使用不完美的 pred_astart，方向仍然对
- 训练是迭代过程：第 1 步网络差 → pred_astart 不准 → guide_loss 大但梯度对 → 更新网络 → ... → 第 N 步网络好 → pred_astart 准 → guide_loss 小

**对应理论**：见 `docs/EDP_复试梳理与演讲稿.md` 的 **2.3.1 为什么需要联合训练**

### 4. 与普通 TD3 的区别

**问题**：与普通 TD3 相比，这里最大的区别是什么？

**答案**：
- **普通 TD3**：policy 直接输出动作 `a = π(s)`，guide loss = `-λ * Q(s, a)`
- **EDP（TD3 + 扩散）**：policy 是扩散模型，用 `pred_astart`（预测的 x0）当动作，guide loss = `-λ * Q(s, pred_astart)`
- **关键区别**：策略表示方式不同（扩散模型 vs 普通神经网络），但 guide loss 的形式相同（都是最大化 Q）

---

### Day 3 自测参考答案（折叠，答题时不要展开）

<details>
<summary>点击展开参考答案</summary>

1. **TD3 三块组件与 EDP**  
   - 三块：双 Q（取 min 减过估计）、target network（软更新）、policy 最大化 Q。  
   - EDP 用的是**完整 TD3 AC 流程**：value loss（双 Q + target）更新 Q，guide loss（−λQ(s, pred_astart)）更新策略，再软更新三份 target；不是「只拿双 Q 算 value loss」。

2. **三个 loss 更新谁、公式在哪**  
   - **value loss**：更新 Q 网络（Critic）；公式在 `get_value_loss` 内部（y = r + (1−done)·γ·min(Q1′, Q2′)，L = MSE(Q1,y) + MSE(Q2,y)）。  
   - **diff loss**：更新扩散策略；噪声 MSE 的公式在 `diffusion/diffusion.py` 的 `training_losses`（‖ε − model_output‖²），dql 里通过 `get_diff_loss` → `get_diff_terms` → `policy.loss` 调用。  
   - **guide loss**：更新扩散策略；公式在 `_train_step_td3` 的 `policy_loss_fn` 里，−λ·mean(Q(s, pred_astart))。

3. **为什么两个 Q 都拟合同一 y**  
   - min 只用来**构造** target y（压保守）；构造出 y 后，希望 Q1、Q2 **都**学会预测这个 y，下次算 target 时两边都准。若只训「取到 min 的那一个」，另一个 Q 不更新，双 Q 效果会变差。

4. **value loss 和 guide loss 的区别**  
   - **value loss**：让 Q **预测准**（拟合 TD target），更新 Q；不追求「Q 值更高」。  
   - **guide loss**：让**策略选出的动作**在现有 Q 下得分高（−λ·mean(Q)），更新策略，不更新 Q。「更高的 Q」在 guide 里指策略输出的动作对应更高的 Q(s,a)。

5. **action approximation**  
   - 用**单步预测的 x0（pred_astart）**当动作，不跑完整去噪链，加速训练。  
   - 来源：`get_diff_terms` 里 `policy.loss` 得到 terms 后，用 `p_mean_variance(..., ts)["pred_xstart"]`；用在 `get_diff_loss` 的返回值里，再在 `policy_loss_fn` 里喂给 Q 算 guide loss。

6. **当前网络 vs target、为何三次软更新**  
   - 当前网络三份（policy、qf1、qf2），target 三份（tgt_params['policy']、'qf1'、'qf2'）；共 3×2 套参数。  
   - 软更新是「每个 target 单独向对应 current 靠拢」，所以要对三份 target 各调用一次 `update_target_network`。

7. **stop_gradient 与软更新**  
   - **stop_gradient(tgt_q)**：算 value loss 时，梯度**不**反传到算 tgt_q 用到的 target 参数，只更新当前 Q。  
   - **软更新**：在**更新完当前网络之后**，用公式 θ_tgt = τ·θ_cur + (1−τ)·θ_tgt 更新 target。  
   - 两者分工：stop_gradient 禁止用 loss 梯度改 target；软更新是**唯一**我们想要的 target 更新方式。

8. **update_target_network 里 x、y 对应谁**  
   - 调用为 `update_target_network(main_params, target_params, tau)`，即 main = 当前参数，target = target 参数。  
   - lambda 里 x 对应 **main_params**（当前），y 对应 **target_params**（target）。  
   - 返回值 = τ·x + (1−τ)·y = **新的 target 参数**（写回 tgt_params）。

9. **软更新 vs 硬更新、为何用软更新**  
   - **硬更新**：每 C 步 θ_tgt = θ_cur（直接复制），target 会突变。  
   - **软更新**：每步 θ_tgt = τ·θ_cur + (1−τ)·θ_tgt，target 平滑跟随。  
   - 软更新让 TD 目标变化更平滑，训练更稳；TD3 常用软更新。

10. **\_train_step_td3 的完整顺序**  
    - 构造 value_loss_fn、diff_loss_fn → 用 value_and_multi_grad 对 value_loss 求梯度 → apply_gradients 更新 qf1、qf2 → 用 value_and_multi_grad 对 policy_loss（含 diff_loss + guide_loss）求梯度 → apply_gradients 更新 policy → 软更新三份 target（policy 按 policy_tgt_update 条件）。

11. **与普通 TD3 的最大区别**  
    - 策略从普通网络换成**扩散模型**；动作从 π(s) 换成 **pred_astart**（单步预测 x0）；policy loss 多了一项 **diff_loss**（拟合数据）。AC 骨架一致：Critic 用 value loss + 双 Q + target，Actor 用 guide loss 最大化 Q。

12. **TD3 里有没有 guide loss 这个名字**  
    - TD3 原论文**没有**「guide loss」这个术语，但 Actor 更新就是最大化 Q(s, π(s))，和 EDP 的 guide loss 同一思想。EDP 把这项叫 guide loss，是为了和 diff loss 区分开。

13. **过估计的正反馈与双 Q**  
    - **正反馈**：target y = r + γQ(s',a') 若 Q 已高估 → y 偏大 → value loss 把 Q̂ 往 y 拉 → 下一轮 target 更高 → 持续高估。  
    - **双 Q 取 min**：用 min(Q1′, Q2′) 构造 y，压低 target，打断上述正反馈；两个 Q 都拟合同一 y，下次两边都更准。

14. **为何用「没训练好」的 pred_astart 算 guide loss**  
    - **联合训练**：每步同时更新 Q 和策略，不跑完整去噪；guide 用的是当前步前向得到的 pred_astart。  
    - **梯度方向对**：guide loss = −λQ(s, pred_astart)，梯度指向「让 Q(s, pred_astart) 变大」，即策略改进方向正确；pred_astart 不完美时方向仍对，随训练会变好。

</details>

---

## 📎 附录：离线 RL 与 RL 概念补充

（TD3 / Q̂·Q* / 过估计等见 **「前置：TD3 概念速览」**；此处只补：损失与 bootstrap、BC、RL 分类、为何不用 Q*、马尔可夫性。）

### 损失在干什么、为何不能用真实 Q*

- **损失**：让 **Q̂(s,a)** 逼近 **target**（不是逼近 Q*）。target 里已含 Q̂（如 r + γ·min Q̂_target），即「用当前估计构造目标、再让估计去拟合」= **bootstrap**。
- **为何不能设成「真实 Q* 与 target 的 MSE」**：Q* 不可得（需从 (s,a) 出发按最优策略的无穷采样），离线/在线都算不出；所以必须 bootstrap。本质依赖**马尔可夫性**：目标可写成 r + γ × 下一状态价值，才能用 TD target 逐步更新。

### 误差从哪来、为何「经常为正」

- 误差 = Q̂ − Q*。来源：(1) **有限采样**：目标带方差，Q̂ 拟合带噪目标就有误差；(2) **未收敛/近似能力**：当前 Q̂ 离最优拟合还差一截；(3) **bootstrap**：目标里含 Q̂，误差会传播。在 Q-learning 里因用 **max**，高估的动作更易被选进目标，误差分布**偏向正**（系统性高估）。

### BC（Behavior Cloning）简介

- **定义**：用监督学习模仿数据里的 (s, a)，不做价值估计、不做 RL；损失常为 MSE 或 −log π(a|s)。
- **作用**：策略贴近数据动作分布，**减轻 OOD**；不解决高估。
- **在离线 RL 里**：常与 value/AC 结合，如 TD3+BC；EDP 里的 **diff_loss** 也是拟合数据分布，可视为 BC 成分。

### 离线 RL 里的算法类型（简要）

- **TD3+BC**：TD3 防高估，BC 防 OOD。
- **IQL / CQL**：用 expectile、保守项等同时压高估并限制策略不离数据太远。
- **Policy-based 为何常要结合 value/AC**：纯策略梯度依赖 on-policy 或高方差重要性采样，离线数据固定，直接套用易不稳；与 value/AC 结合后用 Q/advantage 加权或约束才更稳。

### RL 三分法与 TD target

- **Value-based**：学 Q/V，用 **TD target** 更新（如 DQN、CQL）。
- **Policy-based**：学 π，用策略梯度定理，**没有**「对价值网络拟合 TD target」；常用 return/advantage 当权重。
- **Actor-Critic**：Critic 用 TD target；Actor 用策略梯度（如 −Q(s,π(s))）。**TD target** = loss 里要拟合的目标值，形式多为 r + γ × 下一价值。

### TD3 名字里的「TD」vs 公式里的 TD

- **TD3 = Twin Delayed**：名字里的 TD 不是 Temporal Difference（见前置「TD3 是什么、名字从哪来」）。
- **Critic 更新公式**：target = r + γ·Q(s',a')、loss = (Q−target)²，**就是**时序差分（bootstrap）。

### RL 与马尔可夫性

- 当前步回报只依赖当前状态与动作，目标才能写成 r + γ × 下一价值，才能用 TD target 逐步更新；离线 RL 仍用这套，仅数据固定、不交互。

### 小结

- **过估计**：Q̂ > Q*，双 Q 取 min 缓解；**OOD**：动作偏离数据，BC/约束策略缓解。
- **TD target**：loss 中要拟合的目标（r + γ × 下一价值）；**RL 更新**依赖马尔可夫性。

---

## 📝 自测题（补充练习）

> 可与上方「Day 3 提问自测」配合使用：提问自测为 12 道带参考答案的题目；本节为按能力维度的练习要求。

### 基础理解

1. **口述 `_train_step_td3` 的完整流程**
   - 要求：能说出 value loss、diff loss、guide loss 如何计算，参数如何更新

2. **解释 action approximation 在哪里起作用**
   - 要求：能指出代码位置，说明为什么能加速

3. **说明与普通 TD3 的区别**
   - 要求：能说出策略表示方式、动作获取方式的区别

### 深入理解

4. **为什么可以用 pred_astart 计算 guide loss？**
   - 要求：理解联合训练的概念，说明梯度仍然有效

5. **双 Q 的作用是什么？**
   - 要求：理解如何减少过估计，代码中如何实现

6. **Target network 的作用是什么？**
   - 要求：理解为什么需要 target，软更新如何保证目标稳定

### 代码实现

7. **手写 `_train_step_td3` 的伪代码**
   - 要求：包含 value loss、diff loss、guide loss、参数更新、target 更新

---

## 🎓 学习检查清单

完成以下任务后，Day 3 的学习目标就达到了：

- [ ] ✅ 能口述/手写 `_train_step_td3` 的完整流程
- [ ] ✅ 能指出 action approximation 的代码位置并解释其作用
- [ ] ✅ 能说明与普通 TD3 的区别
- [ ] ✅ 能解释为什么可以用「还没训练好」的网络输出
- [ ] ✅ 能理解双 Q 和 target network 的作用
- [ ] ✅ 能理解双损失结构（diff_loss + guide_loss）的必要性

---

## 📖 参考文档

- **理论部分**：`docs/EDP_复试梳理与演讲稿.md`
  - **1.2 DQL 整体思路**、**1.3 核心问题与动机（EDP 三方面改进）**：DQL 框架与 EDP 改进
  - **2.2.2 TD3**：TD3 基础知识
  - **2.3 EDP 的训练目标（双损失）**：双损失结构
  - **2.3.1 为什么需要联合训练**：联合训练的理解（含「不是先训练再生成」）
  - **3.3 训练一步（以 TD3 为例）**：训练流程概述

- **代码部分**：`diffusion/dql.py`（行号以当前文件为准，便于查找）
  - `_train_step_td3`（371-475 行）：主要函数
  - `get_value_loss`（220-282 行）：Q 网络更新
  - `get_diff_loss`（336-359 行）：扩散损失与 pred_astart
  - `get_diff_terms`（284-333 行）：action approximation 与 pred_astart 计算（305-308 行）
  - `update_target_network`（36-39 行）：软更新

- **扩散部分**：`diffusion/diffusion.py`
  - `p_mean_variance`（约 399 行起）：计算 pred_xstart（action approximation 用到）
  - `training_losses`（约 1081-1212 行）：噪声 MSE 公式，被 `nets.DiffusionPolicy.loss` 调用

---

## 💡 学习建议

1. **先看整体，再看细节**：先理解 `_train_step_td3` 的整体结构，再深入每个函数
2. **对照理论看代码**：结合 `EDP_复试梳理与演讲稿.md` 的理论部分理解代码
3. **画流程图**：可以画一个流程图，标注 value loss、diff loss、guide loss 的计算和参数更新
4. **动手调试**：如果有条件，可以在关键位置加 print，观察 `pred_astart`、`diff_loss`、`guide_loss` 的值
5. **对比理解**：对比普通 TD3 和 EDP（TD3 + 扩散）的区别，加深理解

---

## 🚀 下一步

完成 Day 3 后，可以进入 **Day 4：CRR / IQL + 扩散策略（算法兼容性）**，学习：
- CRR 和 IQL 如何需要 log π
- EDP 如何用 ELBO 近似 log π
- `_train_step_crr` 和 `_train_step_iql` 的实现
