# EDP 数学公式超参数与符号表（代码对齐版）

目的：把项目里**所有出现在数学公式/推导中的关键超参数与系数**做成一份“符号-含义-默认值-代码位置”清单，便于复试口述与对照实现。

说明：本表默认以本项目 `diffusion/` 与 `docs/EDP_复试梳理与演讲稿.md` 中使用的记号为主，并严格对齐代码实现（特别是 `diffusion/dql.py` 与 `diffusion/diffusion.py`）。

---

## 0. 统一符号与对象（复习用）

- $s$：环境观测（state / observation）
- $a$：动作（action）。在扩散里 $a$ 也可视为扩散变量（文档中常用 $x$ 表示，但代码里就是 `actions` / `x_t`）
- $a_0$：干净动作（扩散训练里的真实动作，数据集提供的 `batch['actions']`）
- $a_t$：第 $t$ 个噪声步的动作样本（加噪后的变量）
- $\varepsilon$：标准高斯噪声，$\varepsilon\sim\mathcal N(0,I)$
- $\varepsilon_\theta$：噪声网络在时刻 $t$ 的预测（本项目默认输出噪声）
- $\pi_\theta(a\mid s)$：策略（扩散策略）在状态 $s$ 下对动作的隐式分布
- $Q(s,a)$：Q 网络（Critic）。本项目实现为双 Q：$Q_1, Q_2$，通常取 $\min(Q_1,Q_2)$ 以抑制过估计
- $V(s)$：IQL 的 Value 网络（仅 IQL 用）
- $T$：扩散离散步数（`num_timesteps`）

---

## 1. 扩散模型（DDPM 训练 + DDPM/DPM 采样）相关超参数

### 1.1 调度与模型输出形式

| 符号/字段 | 含义 | 默认值（本项目代码） | 代码位置 |
|---|---|---:|---|
| `T` / `num_timesteps` | 扩散离散步数（离散时间索引 $t\in[0,T)$） | 100（`DiffusionQL.get_default_config`） | `diffusion/dql.py` 默认配置；`diffusion/trainer.py` 里传给 `GaussianDiffusion` |
| `schedule_name` | β 调度类型，用于构造 $\beta_t,\alpha_t,\bar\alpha_t$ | `'linear'` | `diffusion/dql.py` 默认配置；`diffusion/diffusion.py` 的 `get_named_beta_schedule` |
| `model_mean_type` | 网络输出的是 $\varepsilon$ / $a_0$ / $a_{t-1}$ 的哪一种 | `ModelMeanType.EPSILON` | `diffusion/trainer.py`：`model_mean_type=EPSILON` |
| `model_var_type` | 反向分布方差是否固定/学习 | `ModelVarType.FIXED_SMALL` | `diffusion/trainer.py`：`model_var_type=FIXED_SMALL` |
| `loss_type` | 扩散训练损失的类型 | `LossType.MSE` | `diffusion/trainer.py` |
| `min_value / max_value` | 反推得到的 $\hat a_0$（或 `pred_xstart`）是否裁剪到动作范围 | 由 `max_action` 决定：`[-max_action, +max_action]` | `diffusion/trainer.py` |
| `time_embed_size` | 时间步嵌入维度 | 16 | `diffusion/dql.py` 默认配置；`diffusion/trainer.py` 传给 `DiffusionPolicy` |

---

### 1.2 前向加噪公式的隐含系数（由 schedule 决定）

给定 $\bar\alpha_t$，前向加噪闭式为：

$\;\;\;a_t=\sqrt{\bar\alpha_t}\,a_0+\sqrt{1-\bar\alpha_t}\,\varepsilon.$

这些量由 β 调度构造：

- $\beta_t$：噪声强度
- $\alpha_t = 1-\beta_t$
- $\bar\alpha_t = \prod_{i=0}^{t}\alpha_i$

下面给出 **线性 schedule（代码默认）** 下 β 的“端点数值”和通式，便于你在复试时口述噪声强度的具体范围：

#### 1.2.1 线性 schedule（`schedule_name='linear'`）

代码在 `diffusion/diffusion.py:get_named_beta_schedule` 里定义：

- `scale = 1000 / T`
- `beta_start = scale * 0.0001`
- `beta_end   = scale * 0.02`
- `beta_t` 在区间 `[beta_start, beta_end]` 上**线性均匀插值**，长度为 `T`。

因此：

- $\beta_{\mathrm{start}}=\dfrac{1000}{T}\cdot 0.0001=\dfrac{0.1}{T}$
- $\beta_{\mathrm{end}}=\dfrac{1000}{T}\cdot 0.02=\dfrac{20}{T}$
- $\beta_t=\beta_{\mathrm{start}}+\dfrac{t}{T-1}\left(\beta_{\mathrm{end}}-\beta_{\mathrm{start}}\right),\quad t=0,\dots,T-1.$

在本项目默认 `T=num_timesteps=100` 时：

- $\beta_{\mathrm{start}}=0.001$
- $\beta_{\mathrm{end}}=0.2$

对应地：

- $\alpha_t=1-\beta_t\in[0.8,0.999]$
- $\bar\alpha_t=\prod_{i=0}^{t}\alpha_i$ 随 $t$ 递减。

代码中主要预计算并缓存的对象：

- `alphas_cumprod` = $\bar\alpha_t$
- `sqrt_alphas_cumprod` = $\sqrt{\bar\alpha_t}$
- `sqrt_one_minus_alphas_cumprod` = $\sqrt{1-\bar\alpha_t}$

---

### 1.3 反向一步（DDPM）中用到的后验系数

在生成（或训练 value loss 的 TD target 里）计算一步去噪的高斯均值时，用的是前向后验：

$q(a_{t-1}\mid a_t,a_0)=\mathcal N(\mu_{\mathrm{post}}(a_t,a_0,t),\;\sigma^2_{\mathrm{post}}(t)I)$

代码给出的关键系数：

- 后验方差：
  - $\sigma^2_{\mathrm{post}}(t)=\beta_t\cdot\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}$
  - 实现：`posterior_variance`
- 后验均值线性系数：
  - $\mu_{\mathrm{post}} = c_1(t)\,a_0 + c_2(t)\,a_t$
  - 实现：`posterior_mean_coef1`（$c_1$）与 `posterior_mean_coef2`（$c_2$）
- 数值稳定用的裁剪对数方差：
  - 实现：`posterior_log_variance_clipped`（避免 $\log 0$）

---

### 1.4 ELBO→加权 MSE 中的时间步权重（用于 CRR elbo 模式）

在 `GaussianDiffusion.__init__` 中预计算：

- 时间步权重：
  - $w_t=\frac{\beta_t}{2(1-\bar\alpha_t)\alpha_t}$
  - 实现：`self.ts_weights`
- 归一化权重（保持总权重尺度稳定）：
  - 实现：`self.normalized_ts_weights`

当 CRR 使用 `crr_weight_mode='elbo'` 时，代码用：

`log_prob = -ts_weights * mse`

因此：**`ts_weights` 是“公式中出现的 ELBO 相关系数”**。

---

## 2. 训练（TD3/CRR/IQL）共享的 RL 超参数

来自 `DiffusionQL.get_default_config`（默认值）：

| 字段 | 含义（公式位置） | 默认值 | 代码位置 |
|---|---|---:|---|
| `loss_type` | 选择 TD3 / CRR / IQL（决定 guide loss 形式） | `'TD3'` | `diffusion/dql.py` |
| `nstep` | n-step 预处理后的折扣/bootstrapping 语义 | 1 | `diffusion/dql.py` |
| `discount` | 折扣因子 γ（配合 n-step 数据预处理） | 0.99 | `diffusion/dql.py` |
| `tau` | **target 网络软更新系数**（不是 IQL 的温度） | 0.005 | `diffusion/dql.py`：`update_target_network(..., tau)` |
| `policy_tgt_freq` | actor/policy target 软更新频率（延迟更新） | 5 | `diffusion/dql.py` |
| `diff_coef` | policy loss 中 diff_loss 的系数 | 1.0 | `diffusion/dql.py` |
| `guide_coef` | policy loss 中 guide_loss 的系数 | 1.0 | `diffusion/dql.py` |
| `use_pred_astart` | action approximation：policy 用单步 `pred_astart` 给 Q 打分/给 log π 近似 | True | `diffusion/dql.py`：`pred_astart` 分支 |
| `alpha` | TD3 的 guide loss 基础系数（λ 自适应） | 2.0 | `diffusion/dql.py`：TD3 `lmbda = alpha / mean(abs(q))` |

---

## 3. TD3 相关公式中的超参数

### 3.1 Guide loss 的 λ（alpha 的作用）

TD3 中使用：

- $q = Q(s,\hat a)$（实现里对双 Q 随机取一个 qf 分支）
- 代码里的自适应权重：
  - $\lambda=\dfrac{\alpha}{\mathrm{stop\_grad}(\mathbb E[|q|])}$
  - `guide_loss = -\lambda\cdot \mathbb E[q]`

默认：

- `alpha=2.0`

### 3.2 Policy loss 组合

`policy_loss = diff_coef * diff_loss + guide_coef * guide_loss`

- 默认 `diff_coef=1.0`, `guide_coef=1.0`

---

## 4. CRR 相关公式中的超参数

CRR 对应代码段中核心超参数（默认来自 `get_default_config`）：

| 字段 | 出现在公式里的位置 | 默认值 | 作用 |
|---|---|---:|---|
| `sample_actions` | 估计 $V(s)\approx \mathbb E_{a\sim\pi}[ \min(Q_1,Q_2)]$ 的采样数 | 10 | 每个 s 采多份动作 |
| `crr_beta` | $ \lambda = \exp(A/\beta)$ 中的 β | 1.0 | 控制优势权重形状 |
| `crr_fn` | λ 的函数形式：`exp` 或 `heaviside` | `'exp'` | 决定权重生成规则 |
| `crr_ratio_upper_bound` | λ 的上界裁剪 | 20 | 防止 λ 过大 |
| `adv_norm` | 是否对 $\exp(A/\beta)$ 做 softmax 归一化 | False | 改变 λ 的尺度 |
| `crr_weight_mode` | log π 近似模式：`elbo` / `mle` / 采样 MSE | `'mle'` | 决定 `log_prob` 怎么来 |
| `fixed_std` | 是否把 `action_dist` 强制成固定方差高斯 | True | 使 log π 更稳定（并简化方差来源） |
| `crr_avg_fn` | 用来计算 $V(s)$ 时对采样维的聚合方式 | `'mean'` | 例如 mean/median 等 |
| `crr_multi_sample_mse` | 采样 MSE 近似时是否用多采样维 | False | 决定近似方式 |

### 4.1 CRR 的 λ（对应代码）

优势：

- `adv = q_pred - avg_fn(v, axis=0)`
- λ 当 `crr_fn='exp'`：
  - $\lambda=\min(\mathrm{crr\_ratio\_upper\_bound},\exp(\mathrm{adv}/\mathrm{crr\_beta}))$
  - 若 `adv_norm=True`：用 softmax(adv/crr_beta) 重新归一化

### 4.2 CRR 的 log π 近似（对应代码）

当 `crr_weight_mode='elbo'`：

- `log_prob = -ts_weights * mse`

当 `crr_weight_mode='mle'`：

- `log_prob = action_dist.log_prob(actions)`

其他模式（采样 MSE）：

- `log_prob` 用负的平方误差形式近似（通过 `action_dist.sample` 采样动作）

### 4.3 action_dist 的方差来源（fixed_std）

当 `fixed_std=True`，代码把 `action_dist` 强制为：

- $\mathcal N(\mathrm{pred\_astart}, I)$
- 具体是：`distrax.MultivariateNormalDiag(pred_astart, ones_like(pred_astart))`

当 `fixed_std=False`，则使用 `GaussianPolicy` 对 log π 的默认参数化（方差随 `policy_dist` 学习而变化）。

---

## 5. IQL 相关公式中的超参数

默认配置（来自 `get_default_config`）：

| 字段 | 出现在公式里的位置 | 默认值 | 作用 |
|---|---|---:|---|
| `expectile` | expectile 回归：diff>0 用权重 expectile，否则 1-expectile | 0.7 | 学 V 的非均值回归目标 |
| `awr_temperature` | guide 权重：代码里 `exp((Q-V)*awr_temperature)` | 3.0 | AWR 的逆温度/温度系数（注意本代码是乘法） |
| `adv_norm` | 是否把 exp 权重做 softmax | False | 改变权重尺度 |

### 5.1 expectile 回归损失（value loss）

代码权重：

- `diff = Q_min(s,a) - V(s)`
- `expectile_weight = diff>0 ? expectile : (1-expectile)`
- `expectile_loss = mean(expectile_weight * diff^2)`

### 5.2 AWR 权重（policy guide）

代码中：

- `exp_a = exp((q_pred - v_pred) * awr_temperature)`
- `log_probs = action_dist.log_prob(actions)`
- `awr_loss = -(exp_a * log_probs).mean()`
- `guide_loss = awr_loss`

因此：在本实现里 **`awr_temperature` 是 exp 的乘数**，而不是写成 “除以 τ” 的形式。

---

## 6. DPM-Solver 数值求解超参数（采样加速）

训练不改 diff loss，但在需要“从噪声到动作”的地方（value loss 的 target 动作 `a'`、以及评估生成动作）会走 `dpm_sample`。

本项目在 `DiffusionPolicy.dpm_sample` 中调用：

`dpm_sampler.sample(x, steps=self.dpm_steps, t_end=self.dpm_t_end)`

默认配置：

| 字段 | 含义 | 默认值 | 代码位置 |
|---|---|---:|---|
| `dpm_steps` | DPM-Solver 的 NFE 预算（用多少次 model forward） | 15 | `dql.py` 默认配置；`nets.py` 调用 |
| `dpm_t_end` | ODE 终止时间（常取 1e-3 量级，随 steps 调整） | 0.001 | `dql.py` 默认配置 |

另外，`dpm_solver.py` 的 `sample()` 默认参数也会生效（本项目没显式传 order/method/skip_type 时）：

- `order=3`
- `method='singlestep'`
- `skip_type='time_uniform'`
- `denoise=False`
- `solver_type='dpm_solver'`
- `atol/rtol=0.0078 / 0.05`

如果你后续要在复试中解释“为什么少步”，可以结合这些默认值与 DPM-Solver 的高阶更新策略来讲。

---

## 7. EAS / ensemble（评估动作选择）相关“超参数”

论文的 EAS 在代码里用 ensemble 形式实现，关键数值来自 `diffusion/trainer.py` 的 `SamplerPolicy`：

- `num_samples = 50`：每个 state 采样候选动作数量
- 双 Q 打分使用：`q = min(q1,q2)`
- 选择方式：`jax.random.categorical(rng, q)`（categorical 的 logits 由 `q` 提供，即相当于对 `q` 做 softmax 后采样）

这部分属于评估策略的配置，不直接出现在训练损失公式里，但通常在复试会被问到，所以单列。

---

## 8. 快速索引：哪些超参数影响“公式的哪块”

- 扩散/ELBO 权重：`num_timesteps`、`schedule_name`、以及由 schedule 推出的 `β_t、α_t、\barα_t` 与 `ts_weights`
- DDPM 单步均值方差：后验系数 `posterior_mean_coef1/2`、`posterior_variance`
- 指导策略（actor 更新）：
  - TD3：`alpha`（自适应 λ）、`diff_coef`、`guide_coef`
  - CRR：`sample_actions`、`crr_beta`、`crr_fn`、`crr_ratio_upper_bound`、`crr_weight_mode`、`fixed_std`
  - IQL：`expectile`、`awr_temperature`、`adv_norm`
- 目标网络延迟更新：`tau`、`policy_tgt_freq`
- 采样加速：`dpm_steps`、`dpm_t_end`（以及 DPM-Solver 默认 `order=3` 等）

