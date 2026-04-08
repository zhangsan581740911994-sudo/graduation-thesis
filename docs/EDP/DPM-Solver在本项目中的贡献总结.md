# DPM-Solver 在本项目中的贡献总结

按「是什么 → 原理 → 如何匹配 → 适配操作 → 效果」梳理。

---

## 1. DPM-Solver 是什么？

**DPM-Solver**（Diffusion Probabilistic Model Solver）是一种**扩散模型的高阶 ODE 采样器**，用于从噪声 $x_T$ 少步、高精度地求解到 $x_0$，而不必像 DDPM 那样跑完整 T 步（如 1000 步）。

- **来源**：论文 *DPM-Solver: Fast Sampling of Diffusion Probabilistic Models with Differential Equations* 等，将扩散 SDE 转为**概率流 ODE**，再用高阶数值方法（1/2/3 阶）求解。
- **在本项目中的角色**：作为**生成/推理阶段**的一种采样方式，与 DDPM、DDIM 并列；训练阶段仍用 DDPM 的 noise MSE，不涉及 DPM-Solver。

---

## 2. 原理层

### 2.1 从 SDE 到 ODE

- **扩散前向**是 SDE：$dx = f(x,t)\,dt + g(t)\,dw$，含随机项 $dw$。
- **概率流 ODE**：存在对应的确定性 ODE，其解的分布与 SDE 采样分布一致：
  - 形式：$dx/dt = f(x,t) - 0.5\,g^2(t)\,\nabla_x \log p_t(x)$。
- **DPM-Solver** 在 ODE 上做数值积分，去掉每步随机性，因此可以**大步长、少步数**求解。

### 2.2 时间坐标：lambda_t（half-logSNR）

- 定义：$\lambda_t = \frac{1}{2}\log(\bar\alpha_t / (1-\bar\alpha_t))$，即 half-logSNR。
- 作用：作为**单调时间坐标**，在 $\lambda$ 空间下 ODE 形式更简单，数值更稳定；DPM-Solver 在连续时间 $t \in [\epsilon, T]$ 上工作，通过 noise schedule 与离散的 $\bar\alpha_n$ 对应。

### 2.3 高阶方法与步长

- **1 阶**（等价 DDIM）：误差 $O(h^2)$，需要小步长、多步。
- **2 阶**：误差 $O(h^3)$，同样精度下可用更大步长。
- **3 阶**：误差 $O(h^4)$，步长可更大。
- 项目里使用 **singlestep DPM-Solver-fast**：在固定 NFE（如 15 次网络调用）下，混合 1/2/3 阶步，在少步内从 $x_T$ 解到 $x_0$。

---

## 3. 在该项目中如何匹配的？

### 3.1 使用场景

- **训练**：不变。仍用 `GaussianDiffusion.training_losses`（随机选 t → 加噪 → 预测 ε → MSE），与 DDPM 一致。
- **生成/评估**：策略需要从「当前状态 $s$」生成「动作 $x_0$」时，可选三种方式之一：
  - **DDPM**：完整 T 步（如 1000 步），每步随机采样。
  - **DDIM**：可跳步的确定性采样（如 100 步）。
  - **DPM**：通过 DPM-Solver 少步 ODE 求解（默认 15 步）。

### 3.2 调用链

- **入口**：`DiffusionPolicy.sample()` 根据 `sample_method in ['ddpm','ddim','dpm']` 调用对应方法；当 `sample_method='dpm'` 时调用 `dpm_sample`。
- **策略 → DPM**：  
  `SamplerPolicy.act()` → `policy.sample()` → `policy.dpm_sample()` → `DPM_Solver(...).sample(x, steps, t_end)`。
- **配置**：`dql.py` 中默认 `dpm_steps=15`、`dpm_t_end=0.001`；策略通过 `DiffusionPolicy(dpm_steps=..., dpm_t_end=...)` 传入，trainer 从 `algo_cfg` 取这两项再传给策略。

### 3.3 与现有扩散组件的对应关系

| 项目组件 | 与 DPM-Solver 的关系 |
|----------|------------------------|
| **GaussianDiffusion** | 提供离散时间噪声调度：`alphas_cumprod`、`sqrt_recip_*` 等；不实现 DPM，只被 DPM 读调度。 |
| **PolicyNet / base_net** | 作为「给定 (obs, x_t, t) 预测 ε」的 model_fn；DPM-Solver 只调用该网络，不改训练目标。 |
| **NoiseScheduleVP (discrete)** | 用 `alphas_cumprod` 构造连续时间 $\alpha_t,\sigma_t,\lambda_t$，供 DPM 在 ODE 中使用。 |
| **训练 / 评估** | 训练永远用 `training_losses`；评估/部署时若 `act_method='dpm'` 或 `sample_method='dpm'`，则用 DPM 生成动作。 |

---

## 4. 做的适配操作是什么？

### 4.1 离散时间 DPM 的对接（NoiseScheduleVP）

- 本项目策略是**离散时间**训练的（$n=0,1,\ldots,N-1$，对应 $\bar\alpha_n$）。
- DPM-Solver 原版支持连续时间；`dpm_solver.py` 中的 **NoiseScheduleVP(schedule='discrete', alphas_cumprod=...)** 将离散 $\bar\alpha_n$ 转为连续 $t$ 上的 $\alpha_t,\sigma_t,\lambda_t$（分段线性插值 log_alpha_t），从而：
  - 训练与推理共用同一套 $\bar\alpha_n$（来自 `GaussianDiffusion` 的 `get_named_beta_schedule`）；
  - ODE 在连续时间上积分，从 $t=T$ 到 $t=t_0$（对应 $t\_end=1/N$，如 0.001）。

### 4.2 噪声预测模型接口（model_fn）

- 网络接口是「(x, t) → ε」且 **t 为离散下标**（0~N-1）；DPM-Solver 需要**连续时间** $t \in [1/N, 1]$。
- **wrap_model**（`nets.py` 中 `dpm_sample` 内）做了两件事：
  1. **时间对齐**：把 DPM 传入的连续 $t$ 转成离散下标：  
     `t_discrete = (t - 1/total_N) * total_N`，再传入 `base_net(obs, x, t_discrete)`。
  2. **可选噪声裁剪**：用 `sqrt_recip_alphas_cumprod` 等做 clip，避免预测噪声过大导致数值不稳定。

这样，**同一套 base_net（训练时用的）** 无需改输入输出，就能被 DPM-Solver 直接当作「噪声预测 model_fn」使用。

### 4.3 预测类型与参数

- **predict_x0**：本项目是**噪声预测**（ε），故 `predict_x0=False`，使用 DPM-Solver 的噪声预测形式（非 DPM-Solver++）。
- **steps / t_end**：  
  - `dpm_steps=15`：NFE=15，用 15 次网络前向完成从 $x_T$ 到 $x_0$。  
  - `dpm_t_end=0.001`：对应离散时 $t_0=1/N$（N=1000），与训练时的时间范围一致。
- **condition**：obs 通过 `partial(self.base_net, observations)` 固定，因此 model_fn 实际是 (x, t) → ε，满足 DPM-Solver 的「条件扩散」用法（条件不随时间变）。

### 4.4 策略与 trainer 的配置传递

- **DiffusionPolicy**：增加 `sample_method`、`dpm_steps`、`dpm_t_end`；`sample()` 根据 `sample_method` 分发到 `ddpm_sample` / `ddim_sample` / `dpm_sample`。
- **SamplerPolicy**：支持 `act_method='dpm'` 或 `'dpmensemble'`，内部调用 `policy.dpm_sample`。
- **Trainer**：从 `algo_cfg.dpm_steps`、`algo_cfg.dpm_t_end` 读入并传给策略；评估时若 `act_method` 含 `dpm`，则用 DPM 采样做评估。

---

## 5. 效果是什么？

### 5.1 速度（采样效率）

- **DDPM**：需完整 T 步（如 1000 步），每步 1 次网络调用 → 1000 次 NFE。
- **DPM-Solver**：默认 15 步（15 次 NFE）即可从 $x_T$ 解到 $x_0$，**推理时显著更快**（约数十倍量级），便于评估和部署。

### 5.2 质量

- 在相同策略权重下，用 DPM 少步采样得到的动作与 DDPM/DDIM 多步采样**质量相近**（经验上 15 步 DPM 可与上百步 DDIM 或完整 DDPM 相比），从而在**不显著损失策略质量**的前提下大幅减少推理成本。

### 5.3 可选性与一致性

- 与 DDPM/DDIM **接口统一**：同一套 `DiffusionPolicy`，仅通过 `sample_method` 和 `act_method` 切换采样方式；训练流程、loss、数据流不变。
- 评估时可同时跑多种采样方式（如 `act_method='ddpm-dpm'`），对比不同采样器的回报与延迟，便于选型。

---

## 小结（一句话）

**DPM-Solver 在本项目中作为「少步、高阶 ODE 采样器」接入扩散策略的生成端，通过离散时间 schedule 对接、连续时间 model_fn 包装和统一策略接口，在保持与 DDPM 训练兼容的前提下，显著加速推理并保持可接受的策略质量。**
