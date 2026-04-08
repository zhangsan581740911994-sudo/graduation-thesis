## Day 2：扩散策略核心理解

> **目标**：弄清楚"扩散策略"如何加噪/去噪，以及如何输出动作。  
> **对应理论**：扩散模型前向 q(x_t|x_0)、后向 p(x_{t-1}|x_t)、训练 loss（ELBO → 噪声 MSE）、采样（DDPM/DDIM/DPM-Solver），以及和 VAE/ELBO 的关系。  
> **说明**：Day2 既讲扩散的**训练**（有 loss，即 noise MSE），也讲**生成**（无 loss，只做采样输出动作）；在 DQL/EDP 里，训练循环只做「训练」，生成仅在评估/部署时进行。

---

### Day 2 提问自测（先尝试自己回答）

> 建议：真正答题时只看这部分，不要往下翻到「参考答案」。

1. 用不超过 3 句话，说明扩散策略的本质是什么？训练时和生成时分别做什么？（生成时输入、输出各是什么？）
2. 前向加噪 q(x_t|x_0) 和后向去噪 p(x_{t-1}|x_t) 有什么区别？哪个是固定的，哪个是学习的？
3. 训练时为什么不跑完整反向链？生成时为什么必须跑完整反向链？
4. 训练时 x_0（动作）从哪里来？observations 和 x_0 在代码里分别对应什么（state 还是 action）？
5. `q_sample` 和 `p_mean_variance` 的区别是什么？它们分别在什么时候使用？
6. `training_losses` 的完整流程是什么？target 是什么？为什么训练"看起来只是 MSE，本质是 ELBO"？
7. pred_xstart 是什么？在训练时（如 DQL 的 guide loss）和生成时分别怎么用？
8. DDPM、DDIM、DPM-Solver 三种采样方法有什么区别？各自的步数、方式、特点是什么？
9. 什么是 SDE？什么是 ODE？为什么 DPM-Solver 可以去掉随机项变成 ODE？
10. 为什么高阶数值方法（如 3 阶）可以用更大的步长？用泰勒展开解释。
11. lambda_t（half-logSNR）是什么？为什么 DPM-Solver 要用它？
12. DDPM 和 VAE/ELBO 的关系是什么？为什么说"训练看起来只是 MSE，本质是最大化 ELBO"？
13. 如果复试老师问你：「请你从代码角度，把扩散策略的训练和生成流程用 1–2 分钟讲一遍」，你会按什么顺序讲？列出你会提到的至少 5 个关键函数。

---

## 一、核心概念速览

### 1.1 扩散策略的本质

- **策略**：给定状态 s，输出动作 a
- **扩散策略**：用扩散模型表示策略，即 `π(a|s)` = 以 s 为条件的扩散生成模型
- **训练时**（扩散本身的训练）：在给定 s 下，用**真实动作 x_0** 加噪得 x_t，让网络预测噪声，用 **MSE** 作为损失，从而**学会**从噪声去噪的能力；**不跑**从 x_T 到 x_0 的完整反向链。
- **推理/生成时**：给定 s，从 **x_T ~ N(0,I)** **采样**（随机噪声，非由 s 算出的闭式）；以 s 为条件逐步去噪得到 x_0（动作）。**此时才跑完整反向链**。在 RL 里，生成部分的**输入**只有 state s，**输出**是动作 x_0；**t** 是去噪循环的步数下标（T, T−1, …, 1），每步一个 t，不是「额外输入」。
- **在 DQL/EDP 中**：训练循环里只做「训练」（noise MSE + guide loss，guide 用 **Q 网络前向**得到的 Q(s, pred_astart)，动作用同一次前向的 pred_astart）；**不跑完整生成**。「生成」仅在**评估/部署**时进行，两者在时间上分开。详见 `docs/EDP_复试梳理与演讲稿.md` 的 **1.2 DQL 整体思路**、**1.3 核心问题与动机**。

**重要澄清：x_0 和 observations 分别是什么？**
- **x_0 = action（动作）**，不是 (s, a) 的二元组。扩散过程作用在**动作空间**上：对动作加噪/去噪。
- **observations = state（状态）**，不是 (s, a) 的二元组。状态作为**条件**，在训练和生成时都固定为当前步的观测。
- **训练时**：batch 来自离线数据集（如 D4RL），`batch['observations']` 为状态，`batch['actions']` 为真实动作（即 x_0）。
- **生成时**：输入只有 state s，输出为动作 x_0；没有「来自数据集的 x_0」。

### 1.2 前向（加噪）vs 后向（去噪）

- **前向 q(x_t|x_0)**：固定过程，把干净动作 x_0 加噪成 x_t（训练时用）
- **后向 p(x_{t-1}|x_t)**：学习过程，从 x_t 去噪得到 x_{t-1}（训练和生成都用）

### 1.3 三种采样方法对比

| 方法 | 步数 | 方式 | 网络调用 | 特点 |
|------|------|------|----------|------|
| **DDPM** | 完整 T 步（如 1000 步） | 离散时间采样 | T 次（1次/步） | 随机性，必须一步步来 |
| **DDIM** | 可以跳步（如 100 步） | 离散时间确定性采样 | K 次（1次/步） | 确定性（eta=0），可以跳步 |
| **DPM-Solver** | 少步（如 15 步） | 连续时间 ODE 求解 | steps×3 次（如 45 次） | 用数值方法大步长求解 |

---

## 二、代码结构：diffusion.py 核心函数

### 2.1 GaussianDiffusion.__init__：β、α、ᾱ 的作用

**位置**：`diffusion/diffusion.py` 第 148-209 行

**关键变量**：
```python
betas = get_named_beta_schedule(schedule_name, num_timesteps)  # 噪声调度
alphas = 1.0 - betas  # α_t = 1 - β_t
alphas_cumprod = np.cumprod(alphas, axis=0)  # ᾱ_t = ∏(1-β_i)
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)  # √ᾱ_t
sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)  # √(1-ᾱ_t)
```

**含义**：
- **β_t**：第 t 步的噪声强度（0→1，逐渐增大）
- **α_t = 1-β_t**：第 t 步保留信号的比例
- **ᾱ_t = ∏α_i**：从 x_0 到 x_t 的累积保留比例
- **前向公式**：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

**为什么需要这些**：
- 前向加噪需要：`q_sample` 用 `sqrt_alphas_cumprod` 和 `sqrt_one_minus_alphas_cumprod`
- 后向去噪需要：`p_mean_variance` 用这些系数计算均值
- 后验计算需要：`q_posterior_mean_variance` 用这些系数

### 2.2 q_sample：前向加噪

**位置**：`diffusion/diffusion.py` 第 305-343 行

**函数签名**：
```python
def q_sample(self, x_start, t, noise):
    """
    前向加噪：从 q(x_t | x_0) 采样，对干净数据 x_0 加噪得到 x_t。
    数学公式：x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
    """
```

**实现**：
```python
signal_part = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
noise_part = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
return signal_part + noise_part
```

**对应公式**：
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

**用途**：
- **训练时**：在 `training_losses` 中，对真实动作 x_0 加噪得到 x_t，然后让网络预测噪声

**关键理解**：
- 这是**固定的、不学习**的过程
- 训练时用这个对真实动作加噪

### 2.3 q_posterior_mean_variance：前向后验（闭式）

**位置**：`diffusion/diffusion.py` 第 345-372 行

**函数签名**：
```python
def q_posterior_mean_variance(self, x_start, x_t, t):
    """
    计算前向过程的后验分布 q(x_{t-1} | x_t, x_0) 的均值、方差和 log 方差。
    这是前向过程的解析后验（闭式解，不依赖网络）。
    """
```

**实现**：
```python
posterior_mean = (
    _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
    _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
)
posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
```

**含义**：
- **q(x_{t-1}|x_t, x_0)**：给定 x_t 和真实 x_0，x_{t-1} 的分布（闭式，不学习）
- **训练时**：用这个作为目标，让 p_θ(x_{t-1}|x_t) 去拟合它
- **生成时**：没有真实 x_0，所以不能用这个，只能用 p_θ

**为什么需要**：
- 训练时我们有真实 x_0，可以算闭式后验，作为目标让网络学习
- 这是 ELBO 中 KL 项的基础：让 p_θ 逼近 q

### 2.4 p_mean_variance：学到的反向去噪

**位置**：`diffusion/diffusion.py` 第 398-540 行

**函数签名**：
```python
def p_mean_variance(self, model_output, x, t, clip_denoised=True):
    """
    计算学到的反向分布 p(x_{t-1} | x_t) 的均值、方差和预测的 x_0。
    这是扩散模型的核心函数：用网络输出计算去噪分布。
    """
```

**关键逻辑**：
1. **根据 model_mean_type 决定网络预测什么**：
   - `EPSILON`：预测噪声 ε（本项目默认）
   - `START_X`：直接预测 x_0
   - `PREVIOUS_X`：直接预测 x_{t-1}

2. **从网络输出得到 pred_xstart**：
   ```python
   if self.model_mean_type == ModelMeanType.EPSILON:
       pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
   ```

3. **用 pred_xstart 计算均值**（关键！）：
   ```python
   model_mean, _, _ = self.q_posterior_mean_variance(
       x_start=pred_xstart, x_t=x, t=t
   )
   ```

**核心思想**：
- 网络预测噪声（或 x_0）→ 得到 pred_xstart → **用同一套 q 的后验公式**计算均值
- 方差由 schedule 固定（`ModelVarType.FIXED_SMALL`）
- 这样 p(x_{t-1}|x_t) 的均值就确定了，可以采样

**为什么这样设计**：
- 训练时：网络学的是"给定 x_t 和 t，预测噪声 ε"
- 生成时：用预测的 ε 反推 pred_xstart，再用 q 的后验公式算均值
- **同一套公式**，只是训练时用真实 x_0，生成时用预测 x_0

### 2.5 training_losses：ELBO → 噪声 MSE

**位置**：`diffusion/diffusion.py` 第 854-919 行

**函数签名**：
```python
def training_losses(self, rng_key, model_forward, x_start, t):
    """
    计算训练损失：随机选 t，加噪得到 x_t，让网络预测噪声，计算 MSE
    """
```

**流程**：
```python
# 1. 随机采样噪声
noise = jax.random.normal(rng_key, x_start.shape, dtype=x_start.dtype)

# 2. 前向加噪：x_0 → x_t
x_t = self.q_sample(x_start, t, noise=noise)

# 3. 网络预测噪声
model_output = model_forward(x_t, self._scale_timesteps(t))

# 4. 计算 MSE（如果 model_mean_type == EPSILON，target = noise）
target = noise  # 对于 EPSILON 类型
terms["mse"] = mean_flat((target - model_output)**2)
terms["loss"] = terms["mse"]
```

**对应公式**：
$$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t,x_0,\epsilon}\big[ \| \epsilon - \epsilon_\theta(x_t, t \mid s) \|^2 \big]$$

**ELBO → MSE 的关系**：
- **宏观目标**：max $\log p_\theta(x_0)$（不可算）
- **用 ELBO 代替**：ELBO = $\log p_\theta(x_0)$ 的下界
- **ELBO 展开**：包含 $\sum_t D_{\mathrm{KL}}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))$
- **固定方差下**：KL 只与均值差有关，最小化 KL = 最小化均差的平方
- **噪声参数化**：用预测 ε 的参数化 → **最小化 KL = 最小化 $\|\epsilon - \epsilon_\theta\|^2$**
- **结论**：训练时的 MSE 就是 ELBO 在固定方差 + 噪声参数化下的形式

**关键理解**：
- **ELBO 只在训练阶段体现**：它规定了损失取 MSE
- **生成阶段不用 ELBO**：只用训练好的网络输出 + 均值公式 + 固定方差采样

### 2.6 p_sample 和 p_sample_loop：DDPM 采样

**位置**：`diffusion/diffusion.py` 第 636-757 行

**p_sample（单步采样）**：
```python
def p_sample(self, rng, model_output, x, t, ...):
    """
    从 x_t 采样 x_{t-1}：根据网络预测的噪声，计算均值，再加随机噪声采样。
    """
    # 计算 p(x_{t-1}|x_t) 的均值 μ 和方差 σ²
    out = self.p_mean_variance(model_output, x, t, clip_denoised=clip_denoised)
    # 采样随机噪声 ε ~ N(0,I)
    noise = jax.random.normal(rng, x.shape, dtype=x.dtype)
    # 采样公式：x_{t-1} = μ + σ·ε（当 t != 0 时）
    sample = out["mean"] + nonzero_mask * np.exp(0.5 * out["log_variance"]) * noise
    return {"sample": sample, "pred_xstart": out["pred_xstart"]}
```

**p_sample_loop（完整采样循环）**：
```python
def p_sample_loop(self, rng_key, model_forward, shape, ...):
    """
    扩散模型的生成循环：从噪声 x_T 逐步去噪到 x_0（动作）。
    """
    # 1. 从纯噪声开始
    x = jax.random.normal(sample_key, shape)  # x_T ~ N(0,I)
    
    # 2. 循环 T 步（从 T-1 到 0）
    indices = list(range(self.num_timesteps))[::-1]
    for i in indices:
        t = np.ones(shape[:-1], dtype=np.int32) * i
        # 网络预测噪声
        model_output = model_forward(x, self._scale_timesteps(t))
        # 采样 x_{t-1}
        out = self.p_sample(sample_key, model_output, x, t, ...)
        x = out["sample"]
    
    # 3. 返回最终动作
    return x
```

**关键理解**：
- **DDPM**：每步都加随机噪声，必须完整 T 步
- **随机性**：来自后验分布的方差 σ（由 beta schedule 决定），不能为 0

### 2.7 ddim_sample 和 ddim_sample_loop：DDIM 采样

**位置**：`diffusion/diffusion.py` 第 759-920 行

**ddim_sample（单步采样）**：
```python
def ddim_sample(self, rng_key, model_output, x, t, eta=0.0, ...):
    """
    用 DDIM 方式从 x_t 采样 x_{t-1}：确定性采样，可以跳步，比 DDPM 快。
    """
    # 计算 p(x_{t-1}|x_t) 的均值（和 DDPM 相同）
    out = self.p_mean_variance(model_output, x, t, clip_denoised=clip_denoised)
    # 从 pred_xstart 反推噪声 ε
    eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
    # DDIM 公式需要的系数
    alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
    alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
    # sigma：DDIM 的方差（当 eta=0 时，sigma=0，完全确定性）
    sigma = eta * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * np.sqrt(1 - alpha_bar / alpha_bar_prev)
    # DDIM 采样公式
    mean_pred = (
        out["pred_xstart"] * np.sqrt(alpha_bar_prev) +
        np.sqrt(1 - alpha_bar_prev - sigma**2) * eps
    )
    # DDIM 采样：x_{t-1} = mean_pred + sigma * noise
    # 当 eta=0 时，sigma=0，完全确定性（可以跳步）
    sample = mean_pred + nonzero_mask * sigma * noise
    return {"sample": sample, "pred_xstart": out["pred_xstart"]}
```

**关键理解**：
- **DDIM 的 mean_pred ≠ DDPM 的 μ**：它们是不同的公式
- **确定性（eta=0）**：sigma=0，没有随机项，每次结果相同
- **可以跳步**：因为确定性，可以从任意时间步 t 直接跳到任意时间步 s（s < t）

---

## 三、代码结构：nets.py 核心组件

### 3.1 TimeEmbedding：时间步嵌入

**位置**：`diffusion/nets.py` 第 74-85 行

**作用**：
- 把离散时间步 t 编码成连续向量
- 使用正弦位置编码（sinusoidal embedding）

**为什么需要**：
- 扩散模型需要知道"当前是第几步去噪"
- 网络输入：`(state, action, time_embed)` → 输出：预测的噪声

### 3.2 PolicyNet：策略网络主体

**位置**：`diffusion/nets.py` 第 95-121 行

**结构**：
```python
def __call__(self, state, action, t):
    time_embed = TimeEmbedding(...)(t)
    x = jnp.concatenate([state, action, time_embed], axis=-1)  # 拼接输入
    # 多层 MLP
    for feat in self.arch:
        x = nn.Dense(feat)(x)
        x = self.act(x)
    x = nn.Dense(self.output_dim)(x)  # 输出：预测的噪声（或 x_0）
    return x
```

**输入输出**：
- **输入**：`(state, action=x_t, t)` → 当前状态、加噪后的动作、时间步
- **输出**：预测的噪声 ε（如果 `model_mean_type == EPSILON`）

**训练时**：
- `forward(obs, x_t, t)` → 预测噪声
- `loss(obs, x_0, t)` → 调用 `training_losses` 计算 MSE

### 3.3 DiffusionPolicy：扩散策略封装

**位置**：`diffusion/nets.py` 第 124-313 行

#### 3.3.1 loss：训练损失
```python
def loss(self, rng_key, observations, actions, ts):
    terms = self.diffusion.training_losses(
        rng_key,
        model_forward=partial(self.base_net, observations),
        x_start=actions,  # x_0 = 真实动作
        t=ts
    )
    return terms
```

**流程**：
1. 对真实动作 `actions`（x_0）随机选时间步 `ts`
2. 调用 `training_losses`：加噪 → 预测 → MSE

#### 3.3.2 ddpm_sample：DDPM 采样
```python
def ddpm_sample(self, rng, observations, deterministic=False, repeat=None):
    shape = observations.shape[:-1] + (self.action_dim,)
    return self.diffusion.p_sample_loop(
        rng_key=rng,
        model_forward=partial(self.base_net, observations),
        shape=shape,
        clip_denoised=True,
    )
```

**流程**（在 `p_sample_loop` 中）：
1. 从 $x_T \sim \mathcal{N}(0,I)$ 开始
2. 逐步去噪：`t = T, T-1, ..., 1`
   - 调用 `base_net(obs, x_t, t)` 预测噪声
   - 调用 `p_mean_variance` 得到均值
   - 采样 $x_{t-1} \sim \mathcal{N}(\text{mean}, \text{variance})$
3. 返回 $x_0$（动作）

#### 3.3.3 ddim_sample：DDIM 采样
```python
def ddim_sample(self, rng, observations, deterministic=False, repeat=None):
    return self.diffusion.ddim_sample_loop(
        rng_key=rng,
        model_forward=partial(self.base_net, observations),
        shape=shape,
        clip_denoised=True,
    )
```

**特点**：
- 确定性采样（给定 rng，结果固定）
- 可跳步（用更少步数）

#### 3.3.4 dpm_sample：DPM-Solver 采样（加速）

**位置**：`diffusion/nets.py` 第 194-274 行

**流程**：
```python
def dpm_sample(self, rng, observations, deterministic=False, repeat=None):
    # 1. 构造噪声调度
    ns = NoiseScheduleVP(schedule='discrete', alphas_cumprod=self.diffusion.alphas_cumprod)
    
    # 2. 创建 DPM-Solver
    dpm_sampler = DPM_Solver(
        model_fn=wrap_model(partial(self.base_net, observations)),
        noise_schedule=ns,
        predict_x0=self.diffusion.model_mean_type is ModelMeanType.START_X,
    )
    
    # 3. 从噪声开始，调用 DPM-Solver 采样
    x = jax.random.normal(rng, shape)  # x_T ~ N(0,I)
    out = dpm_sampler.sample(x, steps=self.dpm_steps, t_end=self.dpm_t_end)
    return out
```

**优势**：
- **DDPM**：需要 T 步（如 1000 步）完整去噪
- **DPM**：只需约 15 步 ODE 求解，速度更快，质量相当

---

## 四、理论详解：DPM-Solver 与数值方法

### 4.1 SDE（随机微分方程）vs ODE（常微分方程）

**SDE = Stochastic Differential Equation（随机微分方程）**

- **形式**：`dx = f(x,t)·dt + g(t)·dw`
  - `f(x,t)·dt`：确定性漂移项
  - `g(t)·dw`：随机扩散项，`dw` 是布朗运动的微分
- **DDPM 的前向过程是 SDE**：
  ```
  dx = -β(t)/2 · x·dt + √β(t)·dw
  ```
- **特点**：有随机性，每次采样结果可能不同

**ODE = Ordinary Differential Equation（常微分方程）**

- **形式**：`dx/dt = f(x, t)`
- **特点**：确定性，给定初始条件，解是唯一的

**为什么可以去掉随机项变成 ODE？**

关键理论：**概率流 ODE（Probability Flow ODE）**

- 对于扩散过程的 SDE：`dx = f(x,t)·dt + g(t)·dw`
- 存在一个对应的 ODE：`dx/dt = f(x,t) - 0.5·g²(t)·∇_x log p_t(x)`
- **等价性（为什么“可以”去掉随机项）**：若初始点 $x_T$ 的分布相同（如都从 $\mathcal{N}(0,I)$ 采样），则**在任意时刻 t，沿 ODE 演化得到的 $x_t$ 的分布 = 沿 SDE 采样得到的 $x_t$ 的分布**。即：我们不是在“随便丢掉随机性”，而是换了一条**确定性但分布等价**的演化方程来解。
- **直观理解**：
  - SDE：像"随机游走"，每次路径不同，但大量路径在 t 时刻的**分布**是确定的
  - ODE：像这条分布的“平均路径”/概率流；沿这条流走，**每个时刻 t 的边际分布**与 SDE 一致
  - 所以：用 ODE 算出来的 $x_0$，在分布意义下和用 SDE 采样出来的 $x_0$ 一致，因此可以放心用 ODE 替代 SDE 做采样

**为什么去掉随机项后可以“大步长、少步数”求解？**

- **SDE 每步有随机项**：$x_{t-1} = \mu + \sigma\cdot\varepsilon$，$\varepsilon$ 是随机噪声。轨迹不光滑，每步都有方差；即使用高阶方法，**随机性带来的误差**不会随阶数提高而按 h 的幂次消失，所以 DDPM 必须小步长、多步数才能控制方差。
- **ODE 是确定性的**：轨迹光滑，没有每步的随机扰动。数值误差只来自**截断误差**（泰勒余项），而高阶方法可以把截断误差压到 O(h²)、O(h³)、O(h⁴)。因此**在相同精度下**，可以用更大步长 h，从而少步数（如 15 步）就从 $x_T$ 解到 $x_0$。
- **小结**：不是“因为去掉随机项所以随便就能大步长”，而是：去掉随机项后我们解的是**分布等价的 ODE**，而 ODE 的平滑性使得**高阶数值方法**成立，从而在少步数下也能达到可接受的精度。

### 4.2 lambda_t（half-logSNR）的作用

**定义**：`lambda_t = log(alpha_t) - log(sigma_t) = 0.5·log(ᾱ_t / (1-ᾱ_t))`

- `alpha_t = √ᾱ_t`（均值系数）
- `sigma_t = √(1-ᾱ_t)`（标准差）
- `lambda_t` 是"信噪比的对数的一半"，描述了噪声水平

**为什么用 lambda_t？**

- `lambda_t` 是时间的单调函数，可以作为一个"新的时间坐标"
- 在 lambda 空间中，ODE 的形式更简单，数值求解更稳定
- DPM-Solver 在 lambda 空间中求解，而不是直接在 t 空间中

### 4.3 为什么高阶方法可以用更大的步长？

**核心原理：用泰勒展开理解误差**

#### 4.3.1 h 是什么？

**h = 步长（step size）**

- **定义**：从一个时间点 t 到下一个时间点 t+h 的间隔
- **例子**：
  - 如果从 t=0 到 t=1，用 100 步：h = 1/100 = 0.01
  - 如果从 t=0 到 t=1，用 10 步：h = 1/10 = 0.1
  - 如果从 t=0 到 t=1，用 5 步：h = 1/5 = 0.2
- **直观理解**：
  - h 就像"每一步走多远"
  - h 小 = 小步走（步数多，但每步精确）
  - h 大 = 大步走（步数少，但每步可能不够精确）

#### 4.3.2 为什么误差与 h 的幂次有关？

**核心：泰勒展开的余项理论**

我们要求解：`dx/dt = f(x,t)`，已知 x(t)，求 x(t+h)

**真实解可以用泰勒展开**：
```
x(t+h) = x(t) + h·x'(t) + (h²/2!)·x''(t) + (h³/3!)·x'''(t) + (h⁴/4!)·x''''(t) + ...
```

**关键理解**：
- 这是一个**无穷级数**，包含 h、h²、h³、h⁴、... 等项
- 每一项的系数是 x 的导数（x'(t)、x''(t)、x'''(t)、...）
- 当 h 很小时，h² 比 h 小得多，h³ 比 h² 小得多，以此类推

**不同阶方法的近似**：

1. **1 阶方法（Euler）**：
   - 近似：`x(t+h) ≈ x(t) + h·f(x,t)` = `x(t) + h·x'(t)`
   - **只用了泰勒展开的前 2 项**
   - **被忽略的项**：`(h²/2!)·x''(t) + (h³/3!)·x'''(t) + ...`
   - **误差** = 被忽略的第一项 = `(h²/2!)·x''(t) + ...`
   - 当 h→0 时，主要项是 h² 的倍数，所以误差 = **O(h²)**
   
   **为什么是 O(h²)？**
   - 误差的主要部分是 `(h²/2!)·x''(t)`
   - 当 h 很小时，h³、h⁴ 等项更小，可以忽略
   - 所以误差 ≈ C·h²（C 是常数），即 O(h²)

2. **2 阶方法**：
   - 近似：`x(t+h) ≈ x(t) + h·f(x + h/2·f, t + h/2)`
   - 通过巧妙设计，**精确匹配了泰勒展开的前 3 项**：`x(t) + h·x'(t) + (h²/2!)·x''(t)`
   - **被忽略的项**：`(h³/3!)·x'''(t) + (h⁴/4!)·x''''(t) + ...`
   - **误差** = 被忽略的第一项 = `(h³/3!)·x'''(t) + ...`
   - 当 h→0 时，主要项是 h³ 的倍数，所以误差 = **O(h³)**

3. **3 阶方法**：
   - 近似：`x(t+h) ≈ x(t) + h·(c₁·f₁ + c₂·f₂ + c₃·f₃)`
   - 通过巧妙设计，**精确匹配了泰勒展开的前 4 项**
   - **被忽略的项**：`(h⁴/4!)·x''''(t) + ...`
   - **误差** = 被忽略的第一项 = `(h⁴/4!)·x''''(t) + ...`
   - 当 h→0 时，主要项是 h⁴ 的倍数，所以误差 = **O(h⁴)**

**为什么误差是 h 的幂次？**

**数学原理：泰勒展开的余项公式**

对于 n 阶方法，它精确匹配泰勒展开的前 n+1 项：
```
x(t+h) = x(t) + h·x'(t) + ... + (hⁿ/n!)·x⁽ⁿ⁾(t) + Rₙ₊₁
```

其中**余项 Rₙ₊₁**（被忽略的部分）：
```
Rₙ₊₁ = (hⁿ⁺¹/(n+1)!)·x⁽ⁿ⁺¹⁾(ξ)
```

其中 ξ 在 [t, t+h] 之间。

**所以**：
- n 阶方法的误差 = Rₙ₊₁ = O(hⁿ⁺¹)
- 这就是为什么 1 阶方法误差 O(h²)，2 阶方法误差 O(h³)，3 阶方法误差 O(h⁴)

---

### 4.4 DPM-Solver 复试口述总结（几句话流利说清）

**原理**：扩散的反向过程本来是一条带随机项的 SDE；理论上存在一条与之**分布等价**的确定性 ODE，叫概率流 ODE。也就是说，用这条 ODE 从噪声积分到 $x_0$，得到的样本分布和用 SDE 采样是一样的。

**为什么能用**：概率流 ODE 在任意时刻 t 的边际分布和 SDE 一致，所以“去掉随机项”不是乱删，而是换成等价的 ODE 来解，理论上不损失分布正确性。

**怎么操作**：把离散时间扩散看成连续时间（时间 t 连续），满足一条 ODE；然后用 **DPM-Solver** 对这条 ODE 做数值积分。核心思想是**泰勒展开**：用高阶方法（2 阶、3 阶）多匹配泰勒展开的几项，截断误差就是 O(h³)、O(h⁴)，所以在相同精度下可以用更大步长 h，少走几步就从 $x_T$ 解到 $x_0$。还会用 lambda_t（half-logSNR）做时间坐标，让 ODE 形式更简单、数值更稳。

**成果**：在**不损失分布质量**的前提下，用很少的步数（比如 15 步）就能完成采样，推理速度比 DDPM 的几百上千步快很多，适合部署和评估。

---

**为什么高阶项被忽略？**
- 当 h 很小时，h² 比 h 小得多，h³ 比 h² 小得多
- 例如：h=0.1 时，h²=0.01，h³=0.001，h⁴=0.0001
- 所以高阶项（h³、h⁴）的影响很小，可以忽略

**为什么误差越小，步长可以越大？**

假设要从 t=0 积分到 t=1，要求总误差 < 0.01：

- **1 阶方法（误差 O(h²)）**：
  - 总误差 ≈ C·h（因为 N=1/h，总误差 = N·C·h² = C·h）
  - 要保证 C·h < 0.01，所以 **h < 0.01**，需要 **100 步**

- **2 阶方法（误差 O(h³)）**：
  - 总误差 ≈ C·h²
  - 要保证 C·h² < 0.01，所以 **h < 0.1**，只需要 **10 步**

- **3 阶方法（误差 O(h⁴)）**：
  - 总误差 ≈ C·h³
  - 要保证 C·h³ < 0.01，所以 **h < 0.22**，只需要 **5 步**

**关键洞察**：
- 误差是 h 的幂次：1 阶 O(h²)，2 阶 O(h³)，3 阶 O(h⁴)
- 当 h 增大时，h⁴ 增长比 h² 慢得多
- 所以高阶方法可以用更大的 h 而保持误差不变

**直观理解**：
- 1 阶方法：用直线近似（误差大，需要小步长）
- 2 阶方法：用抛物线近似（误差较小，可以用更大步长）
- 3 阶方法：用三次曲线近似（误差更小，可以用更大步长）

**实际效果**：
- 1 阶：1000 步 × 1 次/步 = 1000 次网络调用
- 3 阶：15 步 × 3 次/步 = 45 次网络调用
- 速度提升：约 22 倍！

---

## 五、前向/反向流程图

### 5.1 训练阶段流程图

```
训练时（training_losses）：
┌─────────────────────────────────────────────────────────┐
│ 输入: (obs, action=x_0)                                 │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ 随机选 t ~ Uniform(0,T)│
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ 采样噪声 ε ~ N(0,I)   │
        └───────────────────────┘
                    │
                    ▼
        ┌──────────────────────────────────────────┐
        │ q_sample: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε   │
        │ (前向加噪，固定过程)                      │
        └──────────────────────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────────────────────┐
        │ base_net(obs, x_t, t) → ε_pred           │
        │ (网络预测噪声)                            │
        └──────────────────────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────────────────────┐
        │ loss = ||ε - ε_pred||²                    │
        │ (MSE，对应 ELBO)                          │
        └──────────────────────────────────────────┘
```

**关键点**：
- 输入 `(obs, action=x_0)`：obs 为状态，x_0 为真实动作（来自离线数据集的 `batch['actions']`）。
- 只随机选一个 t，不跑完整链
- 前向加噪是固定的，不学习
- 网络学的是"给定 (s, x_t, t)，预测噪声 ε"

### 5.2 生成阶段流程图（DDPM）

```
生成时（ddpm_sample / p_sample_loop）：
┌─────────────────────────────────────────────────────────┐
│ 输入: obs (只有状态)                                    │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ x_T ~ N(0,I)           │
        │ (从噪声开始)            │
        └───────────────────────┘
                    │
                    ▼
        ┌──────────────────────────────────────────┐
        │ for t in [T, T-1, ..., 1]:               │
        │                                           │
        │   1. ε_pred = base_net(obs, x_t, t)      │
        │      (预测噪声)                           │
        │                                           │
        │   2. pred_xstart = _predict_xstart_from_eps│
        │      (从 ε_pred 反推 x_0)                 │
        │                                           │
        │   3. mean = q_posterior_mean_variance(    │
        │         pred_xstart, x_t, t)[0]          │
        │      (用 q 的后验公式算均值，             │
        │       只是 x_0 换成预测值)                │
        │                                           │
        │   4. x_{t-1} ~ N(mean, variance)         │
        │      (采样，方差由 schedule 固定)         │
        └──────────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ 返回 x_0 (动作)        │
        └───────────────────────┘
```

**关键点**：
- 必须跑完整反向链（T 步）
- 每一步都用网络预测，再用 q 的后验公式算均值
- 生成时没有真实 x_0，只能用预测的 x_0

### 5.3 前向 vs 后向对比图

```
前向过程 q(x_t|x_0) [固定，不学习]
─────────────────────────────────────
x_0 ──[加噪]──> x_1 ──[加噪]──> x_2 ──...──> x_T
      q(x_1|x_0)    q(x_2|x_1)         q(x_T|x_{T-1})
      
公式: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
用途: 训练时对真实动作加噪


后向过程 p(x_{t-1}|x_t) [学习]
─────────────────────────────────────
x_T ──[去噪]──> x_{T-1} ──[去噪]──> ... ──[去噪]──> x_0
      p(x_{T-1}|x_T)    p(x_{T-2}|x_{T-1})      p(x_0|x_1)
      
步骤: 
1. 网络预测 ε_pred = base_net(obs, x_t, t)
2. 反推 pred_xstart
3. 用 q 的后验公式算均值
4. 采样 x_{t-1} ~ N(mean, variance)

用途: 训练时学去噪，生成时从噪声得到动作
```

---

## 六、训练 vs 生成：完整流程对比

### 6.1 训练阶段（training_losses）

**输入**：`(obs, action=x_0)`（状态和真实动作）

**流程**：
1. 随机选时间步 `t ~ Uniform(0, T)`
2. 采样噪声 `ε ~ N(0,I)`
3. 前向加噪：`x_t = q_sample(x_0, t, ε)` = $\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
4. 网络预测：`ε_pred = base_net(obs, x_t, t)`
5. 计算损失：`loss = ||ε - ε_pred||²`

**关键**：
- **不跑完整反向链**：只随机选一个 t，加噪一次，预测一次，得到的是**扩散的 MSE 损失**（即 diff_loss）。
- **训练数据来源**：DQL/EDP 中，每次训练 step 的 `(obs, action=x_0)` 来自 **离线数据集**（如 D4RL）：`batch = dataset.sample()`，其中 `observations` 为状态、`actions` 为真实动作（即 x_0）。
- **与 DQL/EDP 的关系**：在 DQL/EDP 里，策略还受 **guide loss** 约束：用**同一次前向**得到的 **pred_xstart** 当动作，再用 **Q 网络前向**得到的 Q(s, pred_astart) 构造 guide loss；不跑完整生成（**action approximation**）。本节只描述扩散本身的训练（noise MSE）。

### 6.2 生成阶段（ddpm_sample / dpm_sample）

**输入**：`obs`（只有状态）

**流程**（以 DDPM 为例）：
1. 初始化：`x_T ~ N(0,I)`
2. 逐步去噪：`for t in [T, T-1, ..., 1]`
   - `ε_pred = base_net(obs, x_t, t)`（预测噪声）
   - `pred_xstart = _predict_xstart_from_eps(x_t, t, ε_pred)`（反推 x_0）
   - `mean = q_posterior_mean_variance(pred_xstart, x_t, t)[0]`（用 q 的后验公式算均值）
   - `x_{t-1} ~ N(mean, variance)`（采样）
3. 返回 `x_0`（动作）

**关键**：
- **跑完整反向链**：从 x_T 到 x_0，每一步都要去噪；**x_T 是随机采样的**（~ N(0,I)），不是由 s 闭式算出的。
- **RL 视角**：生成部分的**输入**只有 state s，**输出**是动作 x_0；**t** 是去噪链的步数下标（每步一个 t），不是外部输入。
- **用同一套公式**：生成时用的 `q_posterior_mean_variance` 和训练时的公式一样，只是 x_0 换成预测值。
- **在 DQL/EDP 中**：生成只在**评估/部署**需要输出动作时执行，训练循环内不跑生成。

---

## 七、DDPM 与 VAE/ELBO 的关系

### 7.1 相似性

**VAE**：
- 潜变量：z
- 生成：$p_\theta(x|z)$
- 变分分布：$q_\phi(z|x)$
- 目标：max $\log p_\theta(x)$，用 ELBO 代替

**DDPM**：
- 潜变量：整条链 $x_{1:T}$
- 生成：$p_\theta(x_0|x_{1:T})$（逐步去噪）
- 变分分布：$q(x_{1:T}|x_0)$（固定，不学习）
- 目标：max $\log p_\theta(x_0)$，用 ELBO 代替

**共同点**：
- 都是变分生成模型
- 都用 ELBO 近似难算的 log p(x)
- 都学一个生成分布，用变分分布辅助

### 7.2 差异

**VAE**：
- 变分分布 $q_\phi$ 是学习的（编码器）
- ELBO 包含重建项 + KL 项

**DDPM**：
- 变分分布 $q$ 是固定的（前向加噪过程）
- ELBO 在固定方差 + 噪声参数化下 → **直接变成 MSE**
- 训练"看起来"只是 MSE，但本质是最大化 ELBO

### 7.3 为什么训练是 MSE

**推导**：
1. ELBO = $\log p_\theta(x_0)$ 的下界
2. ELBO 包含：$\sum_t D_{\mathrm{KL}}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))$
3. 固定方差：KL 只与均值差有关
4. 噪声参数化：用预测 ε → 均值差 = 噪声差
5. **最小化 KL = 最小化 $\|\epsilon - \epsilon_\theta\|^2$** = MSE

**结论**：
- **ELBO 只在训练阶段体现**：它规定了损失取 MSE
- **生成阶段不用 ELBO**：只用训练好的网络 + 均值公式 + 采样

---

## 八、关键代码位置索引

| 功能 | 代码位置 | 关键函数 |
|------|---------|---------|
| **β/α 调度** | `diffusion.py:148-254` | `__init__`, `get_named_beta_schedule` |
| **前向加噪** | `diffusion.py:305-343` | `q_sample` |
| **前向后验（闭式）** | `diffusion.py:345-372` | `q_posterior_mean_variance` |
| **后向去噪（学习）** | `diffusion.py:398-540` | `p_mean_variance` |
| **DDPM 单步采样** | `diffusion.py:636-686` | `p_sample` |
| **DDPM 采样循环** | `diffusion.py:688-757` | `p_sample_loop` |
| **DDIM 单步采样** | `diffusion.py:759-826` | `ddim_sample` |
| **DDIM 采样循环** | `diffusion.py:857-920` | `ddim_sample_loop` |
| **训练损失** | `diffusion.py:854-919` | `training_losses` |
| **时间嵌入** | `nets.py:74-85` | `TimeEmbedding` |
| **策略网络** | `nets.py:95-121` | `PolicyNet` |
| **扩散策略封装** | `nets.py:124-313` | `DiffusionPolicy` |
| **DDPM 采样** | `nets.py:164-192` | `ddpm_sample` → `p_sample_loop` |
| **DDIM 采样** | `nets.py:276-292` | `ddim_sample` → `ddim_sample_loop` |
| **DPM 采样** | `nets.py:194-274` | `dpm_sample` → `DPM_Solver` |

---

## 九、当天需要能做到

✅ **口头说清楚一次扩散训练 step**：
- 给定 (s, a=x_0)，随机选 t
- 加噪得到 x_t = q_sample(x_0, t, ε)
- 网络预测 ε_pred = base_net(s, x_t, t)
- 计算 MSE = ||ε - ε_pred||²

✅ **懂得推理时如何生成动作**：
- 给定状态 s，从 x_T ~ N(0,I) 开始
- 逐步去噪：x_t → 预测 ε → 反推 pred_xstart → 用 q 的后验公式算均值 → 采样 x_{t-1}
- 重复直到 x_0（动作）

✅ **能回答"DDPM 和 VAE/ELBO 的关系"**：
- DDPM 是变分生成模型，和 VAE 类似
- 都用 ELBO 近似难算的 log p(x)
- DDPM 的变分分布固定，ELBO 在固定方差+噪声参数化下 → MSE
- 训练时的 MSE 本质是最大化 ELBO

✅ **能解释三种采样方法的区别**：
- DDPM：随机性，完整 T 步，必须一步步来
- DDIM：确定性（eta=0），可以跳步，更快
- DPM-Solver：连续时间 ODE 求解，高阶数值方法，步数最少

✅ **能解释 DPM-Solver 的理论基础**：
- SDE → ODE（概率流 ODE）
- lambda_t（half-logSNR）的作用
- 为什么高阶方法可以用更大步长（泰勒展开 + 误差分析）

✅ **能澄清 x_0 与 observations**：
- x_0 = action（动作），observations = state（状态）；二者都不是 (s, a) 二元组；训练时 x_0 来自离线数据集的 `batch['actions']`。

---

## 十、Day 2 自测参考答案（折叠，答题时不要展开）

<details>
<summary>点击展开参考答案</summary>

1. **扩散策略的本质**
   - 用扩散模型表示策略 π(a|s)，给定状态 s，通过扩散去噪从噪声生成动作 a
   - 训练时：用真实动作 x_0 加噪得 x_t，让网络预测噪声、算 MSE，从而学会去噪；不跑完整反向链
   - 推理/生成时：**输入**只有 state s，**输出**是动作 x_0；从 x_T ~ N(0,I) 采样，逐步去噪得到 x_0；此时才跑完整反向链

2. **前向 vs 后向的区别**
   - 前向 q(x_t|x_0)：固定过程，不学习，训练时对真实动作加噪
   - 后向 p(x_{t-1}|x_t)：学习过程，依赖网络输出，训练和生成都用

3. **训练 vs 生成的区别**
   - 训练时：只随机选一个 t，加噪一次，预测一次，不跑完整链（太慢且不需要）
   - 生成时：必须跑完整反向链，从 x_T 到 x_0，每一步都依赖前一步的结果

4. **训练时 x_0 从哪来？observations 与 x_0 分别是什么？**
   - x_0 来自**离线数据集**：`batch = dataset.sample()`，`batch['actions']` 即真实动作（x_0）
   - `observations` = state（状态），`actions` = x_0（动作）；二者都不是 (s, a) 二元组；扩散作用在动作空间，状态作为条件

5. **q_sample vs p_mean_variance**
   - `q_sample`：前向加噪，固定过程，不依赖网络，训练时用
   - `p_mean_variance`：后向去噪，依赖网络输出，用 q 的后验公式算均值，生成时用

6. **training_losses 的流程**
   - 随机选 t → 采样噪声 ε → 前向加噪得到 x_t → 网络预测 ε_pred → 计算 MSE = ||ε - ε_pred||²
   - target = 真实噪声 ε（如果 EPSILON 类型）
   - 训练"看起来只是 MSE"：工程实现上只看到 MSE loss
   - "本质是 ELBO"：MSE 来自 ELBO 的推导，在固定方差+噪声参数化下，ELBO 的 KL 项 → MSE

7. **pred_xstart 是什么？训练时和生成时怎么用？**
   - pred_xstart：由当前 x_t 和网络预测的 ε 反推得到的「预测的 x_0」，即 _predict_xstart_from_eps(x_t, ε, t)
   - **训练时**（如 DQL）：同一次前向得到 pred_xstart，用作「当前策略给出的动作」，用于算 Q(s, pred_xstart) 和 guide loss，不跑完整反向链（action approximation）
   - **生成时**：DDPM/DDIM/DPM 每步都会算当前步的 pred_xstart，最后一步的 pred_xstart 就是输出的动作 x_0

8. **三种采样方法的区别**
   - DDPM：完整 T 步（如 1000 步），离散时间采样，随机性，必须一步步来
   - DDIM：可以跳步（如 100 步），离散时间确定性采样，确定性（eta=0），可以跳步
   - DPM-Solver：少步（如 15 步），连续时间 ODE 求解，用高阶数值方法大步长求解

9. **SDE vs ODE**
   - SDE = Stochastic Differential Equation（随机微分方程），有随机项 dw
   - ODE = Ordinary Differential Equation（常微分方程），确定性
   - 概率流 ODE：对于 SDE，存在对应的 ODE，其解的分布与 SDE 的采样分布相同
   - DPM-Solver 用概率流 ODE 代替 SDE，去掉随机项，用数值方法求解

10. **为什么高阶方法可以用更大步长**
   - 用泰勒展开：n 阶方法精确匹配泰勒展开的前 n+1 项，误差 = O(hⁿ⁺¹)
   - 1 阶：误差 O(h²)，需要 h 很小（如 1000 步）
   - 2 阶：误差 O(h³)，可以用更大步长（如 100 步）
   - 3 阶：误差 O(h⁴)，可以用更大步长（如 15 步）
   - 虽然高阶方法每步调用更多网络，但总步数减少，总体更快

11. **lambda_t 的作用**
   - lambda_t = 0.5·log(ᾱ_t / (1-ᾱ_t))，是"half-logSNR"（信噪比的对数的一半）
   - 它是时间的单调函数，作为新的时间坐标
   - 在 lambda 空间中，ODE 形式更简单，数值求解更稳定

12. **DDPM 和 VAE/ELBO 的关系**
    - 相似性：都是变分生成模型，都用 ELBO 近似难算的 log p(x)
    - 差异：DDPM 的变分分布固定，VAE 的变分分布学习
    - ELBO → MSE：在固定方差+噪声参数化下，ELBO 的 KL 项 → MSE
    - 训练时的 MSE 本质是最大化 ELBO

13. **1–2 分钟版训练和生成流程（5 个关键函数）**
    - 训练：`training_losses`（随机选 t → 加噪 → 预测 → MSE）
    - 训练辅助：`q_sample`（前向加噪），`q_posterior_mean_variance`（闭式后验）
    - 生成：`p_sample_loop`（完整反向链），`p_mean_variance`（用网络输出算均值）
    - 生成辅助：`p_sample`（单步采样），`_predict_xstart_from_eps`（反推 x_0）

</details>

---

## 十一、下一步（Day 3 预告）

Day 3 将学习：
- TD3 + 扩散策略的训练流程
- action approximation 如何加速训练
- guide loss 如何与 diff loss 组合
- Q 网络更新和 target network

**今天先理解扩散本身，明天再看 RL 部分！**
