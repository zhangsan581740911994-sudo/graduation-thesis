## EDP 项目复试问题清单

> 说明：本文件整理了可能围绕 EDP / Diffusion‑QL 项目提的问题，方便刷题与模拟面试。可按模块准备答案。

---

### 一、总体理解与背景

1. 请你用 1 分钟介绍一下这个离线强化学习项目在做什么？  
<details>
<summary>参考答案</summary>

这个项目在做**离线强化学习中的扩散策略优化**。用扩散模型表示策略 π(a\|s)，在 D4RL 等固定数据集上训练一个能生成高质量动作的策略。核心是在 Diffusion‑QL 框架上，提高训练/生成效率，并让扩散策略兼容 TD3、CRR、IQL 等不同离线 RL 算法。

</details>

2. 为什么要在离线强化学习里用扩散模型做策略，而不是普通高斯策略？  
<details>
<summary>参考答案</summary>

高斯策略假设动作分布是单峰、相对简单，在复杂任务里容易学到「平均动作」。扩散策略把 π(a\|s) 看成生成模型，可以表达**多模态、非高斯**的复杂动作分布，覆盖更多高质量动作；在离线数据足够丰富时，表达能力明显更强。

</details>

3. Diffusion‑QL 原始框架的整体思路是什么？EDP 相比 Diffusion‑QL 做了哪两点最关键的改进？  
<details>
<summary>参考答案</summary>

Diffusion‑QL：用扩散模型表示策略，训练时联合优化三个部分：扩散 MSE（拟合数据）、Q 网络的 TD 损失、以及用 Q 引导策略的 guide loss；评估时从噪声完整去噪得到动作。EDP 的两点关键改进是：**1）Action Approximation + DPM‑Solver 提升训练/生成效率；2）用 ELBO / 高斯近似构造 log π，使扩散策略兼容 CRR、IQL 等需要 log π 的算法。**

</details>

4. 这个项目中，训练阶段和评估阶段分别在代码哪里实现，各自做了什么？  
<details>
<summary>参考答案</summary>

训练循环在 `diffusion/trainer.py` 的 `train()` 中：每步 `self._dataset.sample()` 取 batch，经 `batch_to_jax` 后调用 `self._agent.train(batch)`，内部在 `diffusion/dql.py` 的 `_train_step_*` 里计算 value loss、diff_loss、guide_loss 并更新参数。评估在同一个 `train()` 里按 `eval_period` 触发，调用 `TrajSampler.sample(SamplerPolicy)` 在环境中滚轨迹，`SamplerPolicy` 用 ddpm/dpm/*ensemble 采样动作，最后用 D4RL 的 `get_normalized_score` 计算 normalized return。

</details>

---

### 二、扩散模型相关

5. 扩散策略的本质是什么？训练时和生成时分别在做什么？  
<details>
<summary>参考答案</summary>

本质：**用条件扩散模型表示策略 π(a\|s)**。训练时，对离线数据中的动作 x₀ 在随机时间步 t 加噪得到 x_t，让网络预测噪声 ε，并用 MSE 训练去噪能力（同时叠加 RL 的 guide loss）；生成时，给定状态 s，从噪声 x_T 开始逐步去噪到 x₀，当作动作输出。

</details>

6. 前向加噪 \(q(x_t \mid x_0)\) 和后向去噪 \(p(x_{t-1} \mid x_t)\) 有什么区别？哪个是固定的，哪个需要学习？  
<details>
<summary>参考答案</summary>

前向 \(q(x_t\|x_0)\) 是**固定的加噪过程**：给干净动作 x₀ 加入高斯噪声，公式 \(x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon\)。后向 \(p(x_{t-1}\|x_t)\) 是**要学习的去噪分布**，均值由网络输出（噪声或 x₀ 预测）决定，方差通常由噪声调度固定。训练时用真实 x₀ 的闭式后验 q 当目标，让 p 去拟合。

</details>

7. 为什么训练时的损失看起来只是噪声的 MSE，本质上是在最大化 ELBO？  
<details>
<summary>参考答案</summary>

宏观目标是最大化 \(\log p_\theta(x_0)\)，难以直接计算，所以用 ELBO 作为下界。ELBO 展开后包含一系列 KL\((q(x_{t-1}\|x_t,x_0)\|p_\theta(x_{t-1}\|x_t))\)。在 DDPM 中方差固定且采用「预测噪声」参数化后，这些 KL 的和等价于**噪声预测的 MSE**，因此训练看起来是 MSE，本质是在优化 ELBO。

</details>

8. DDPM、DDIM 和 DPM‑Solver 三种采样方法有什么区别，项目里为什么要用 DPM‑Solver？  
<details>
<summary>参考答案</summary>

DDPM：随机采样，每步加噪声，通常要跑上百或上千步；DDIM：确定性采样（eta=0）、可以跳步，用更少步数；DPM‑Solver：把反向过程看成 ODE，用高阶数值方法在连续时间上求解，只需十几步即可达到接近 DDPM 的质量。项目用 DPM‑Solver 是为了在评估/部署时**大幅减少网络前向次数，提升生成速度**。

</details>

---

### 三、RL / DQL / EDP 设计相关

9. 这个项目里，扩散损失 diff_loss 和引导损失 guide_loss 各自的作用是什么？  
<details>
<summary>参考答案</summary>

diff_loss 是扩散训练的噪声 MSE，让策略在给定状态下**拟合数据中的动作分布**，可以理解为带扩散的 BC。guide_loss 利用 Q 网络信息把策略**往高价值动作方向拉**：TD3 里是 −λQ(s, â)，CRR/IQL 里是优势加权的 −E\[权重·log π(a\|s)]。两者合起来既模仿数据，又做强化学习。

</details>

10. 为什么在训练阶段可以只用单步预测的 \(\hat{x}_0\)（action approximation），而不跑完整的反向扩散链？  
<details>
<summary>参考答案</summary>

guide_loss 只需要一个「当前策略给出的动作」来让 Q 打分，并不要求每一步都生成最高质量的动作。单步 \(\hat{x}_0\) 是用固定闭式从 x_t 和预测噪声反推出来的，在数据附近已经有合理方向；用它算 −λQ(s, \(\hat{x}_0\)) 的梯度方向仍然正确。相比完整链，单步近似**成本低很多**，可以做更多更新步，实践表明性能接近甚至不差。

</details>

11. 在 TD3、CRR、IQL 这三种算法中，策略是怎样利用 Q 网络的信息更新的？  
<details>
<summary>参考答案</summary>

TD3：用确定性策略梯度，策略损失为 −λQ(s, â)，梯度通过 Q 反传到策略参数。CRR：先算优势 A=Q−E\[Q]，用 \(\lambda(A)\) 当权重，对 log π(a\|s) 做加权 MLE（−E\[λ·log π]）。IQL：先用 expectile 学 V，再用 (Q−V)/τ 的指数权重做 AWR：−E\[exp((Q−V)/τ)·log π(a\|s)]。三者都用 Q 评价动作，但用法不同。

</details>

12. 为什么 TD3 不需要 log π，但 CRR 和 IQL 必须要 log π？  
<details>
<summary>参考答案</summary>

TD3 用的是**确定性策略梯度（DPG）**，梯度形式是 \(\nabla J = E[\nabla_a Q(s,a)\|_{a=\pi(s)} \cdot \nabla_\theta \pi(s)]\)，不涉及 log π。CRR、IQL 走的是**加权 MLE 路径**，目标形如 −E\[权重·log π(a\|s)]，根据策略梯度定理需要显式的 log π(a\|s)，因此必须能计算或近似 log π。

</details>

13. 扩散策略原生为什么不能直接接 CRR / IQL？EDP 是如何用 ELBO 或高斯近似构造 log π 的？  
<details>
<summary>参考答案</summary>

扩散策略本身只定义了从噪声到动作的生成过程 π(a\|s)，但 π(a\|s) 的密度是对整条噪声链积分得到的**边际分布**，没有简单闭式，所以 log π(a\|s) 难算。EDP 在单步预测的 \(\hat{x}_0\) 处用两种近似：一种是构造以 \(\hat{x}_0\) 为均值、方差由 `policy_dist` 学习的**对角高斯** action_dist，用其 log_prob 近似 log π；另一种是用扩散 ELBO 的**加权噪声 MSE（−ts_weights×mse）** 近似 log π。

</details>

14. 双 Q 和 target network 在本项目中分别解决了什么问题？代码里是怎样实现的？  
<details>
<summary>参考答案</summary>

双 Q 用于**减轻过估计**：对同一动作 a′ 用 Q1、Q2 两个网络打分，TD target 里用 min(Q1′, Q2′)。target network 用于**稳定 TD 目标**：用一份缓慢更新的参数来计算 TD 目标，避免目标随当前网络震荡太大。代码在 `dql.py` 的 `get_value_loss` 中使用 min Q 构造 target，并在 `_train_step_*` 末尾用 `update_target_network(main_params, tgt_params, tau)` 做软更新。

</details>

---

### 四、数据与工程实现相关

15. 离线数据集是如何接入代码的？从 D4RL 数据到 `agent.train(batch)` 的数据流大概是怎样的？  
<details>
<summary>参考答案</summary>

命令行只传环境名，例如 `--env hopper-medium-v2`。在 `_setup_d4rl()` 中调用 `get_d4rl_dataset(env, nstep, gamma, norm_reward)`，内部用 D4RL API 读原始数据，再通过 `traj_dataset.get_nstep_dataset` 做按轨迹切分和 n‑step 处理，返回一个 dict。然后用 `data.Dataset(dict)` + `RandSampler` 封装；训练时 `self._dataset.sample()` 拿到 batch，经 `batch_to_jax` 后传给 `self._agent.train(batch)`。

</details>

16. n‑step 回报是在什么地方计算的，公式是什么，为什么对离线 RL 有帮助？  
<details>
<summary>参考答案</summary>

n‑step 在 `utilities/traj_dataset.get_nstep_dataset` 里按轨迹计算。公式是 \(R_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1}r_{t+n-1}\)，对应的 next_state 取第 t+n 步的状态。相较于 1‑step，n‑step 目标方差更小、信息更丰富，在离线 RL 中通常能**提升稳定性和性能**。

</details>

17. 这个项目的日志和评估是怎么做的？normalized return 是如何计算出来的？  
<details>
<summary>参考答案</summary>

日志：`DiffusionTrainer` 中用 `WandBLogger` 记录到 WandB，同时用 `viskit.logging` 把 tabular 指标写入本地 `progress.csv` 并在终端打印。评估：用 `TrajSampler.sample(SamplerPolicy)` 在环境中滚多条轨迹，统计每条的总回报和长度，再调用 D4RL 提供的 `env.get_normalized_score(raw_return)` 把原始回报归一化到 [0,1] 或百分制，记为 normalized return。

</details>

18. 训练时终端里打印的 loss / return 和本地日志、WandB 之间是什么关系？  
<details>
<summary>参考答案</summary>

每一轮训练结束时，代码把 metrics 填进一个字典，同时：1）调用 `self._wandb_logger.log(metrics)` 把同一批指标上传到 WandB；2）调用 `logger.record_dict(viskit_metrics)` + `logger.dump_tabular()` 把指标写进本地 tabular 日志并打印到终端。所以终端输出、本地 `progress.csv` 和 WandB 看见的是**同一份指标的不同展示方式**。

</details>

---

### 五、结果与反思类

19. 在你自己的实验中，使用 Action Approximation 和 DPM‑Solver 前后，训练 / 评估时间和性能有什么变化？  
<details>
<summary>参考答案</summary>

在相同环境和步数设置下，开启 Action Approximation 后，每步训练只需要一次噪声网络前向，整体训练时间明显缩短；评估阶段用 DPM‑Solver 代替 DDPM 后，每次评估的采样时间也大幅减少。实验中 normalized return 与完整链/DDPM 相比基本持平，说明在这个任务上**效率提升几乎没有损失性能**。

</details>

20. 你觉得这个工作还有哪些可以改进或扩展的方向？比如算法、网络结构、或者工程实现上？  
<details>
<summary>参考答案</summary>

可以从几个方向扩展：算法上，可以尝试和 CQL、CQL‑style conservative 方法结合，进一步控制 OOD 动作；网络上，可以引入更强的表示（如 Transformer‑based policy）或更好的时间步编码；工程上，可以在更大规模数据集和更真实的离线场景（例如机器人日志）上验证，并系统比较不同 act_method / EAS 配置对实际部署性能的影响。

</details>

---

### 六、扩展问题

21. 训练发散的含义是什么？  
<details>
<summary>参考答案</summary>

在强化学习中，**训练发散**通常指模型参数更新失去了控制，导致损失函数（如 Q loss）激增到 `NaN` 或无穷大，或者 Q 值的估计无限膨胀。此时策略完全失效，输出极大或极小的异常动作。在深度学习层面，这往往伴随着梯度爆炸。

</details>

22. 为什么过估计会导致发散？  
<details>
<summary>参考答案</summary>

离线 RL 中更新 Q 值的目标是 `TD target = r + γ * Q(s', a')`。
如果因为动作分布偏移，策略选出了一个数据外（OOD）的动作，Q 网络往往会对其给出错误的**高估**。
这个被高估的 Q 值一旦进入 `TD target`，在下一步更新时就会把当前状态的 Q 值也拉高。随着迭代，这种高估会沿着时间步逆向传播，形成**正反馈循环**，导致 Q 值像滚雪球一样无限膨胀，最终引发数值溢出和训练发散。这也是为什么必须引入“双 Q 取 min”或“保守估计（如 IQL 的 expectile）”来打断这种正反馈。

</details>

23. 从解决梯度爆炸和超参设置的角度，本项目是如何保证训练收敛的？
<details>
<summary>参考答案</summary>

为了保证联合训练（扩散策略和 Q 网络）稳定收敛，本项目采取了三层保障：
1）**全局梯度裁剪（Global Gradient Clipping）**：在优化器中使用 `optax.clip_by_global_norm`，并针对不同环境设置特定的阈值（如 `gn`），掐断了 TD target 震荡引发的梯度爆炸。
2）**合理的学习率调度（LR & Decay）**：使用带权重衰减的 AdamW 优化器，配合平滑的**余弦退火衰减**，让网络在训练后期步伐放缓，避免在最优点附近来回震荡。
3）**数值层面的多重截断保护**：对前向反向数值做了多处 `Clip`。例如：用 `jnp.clip` 把输入观测和输出动作截断在合理范围内；在扩散反推 $\hat{x}_0$（`clip_denoised`）时进行约束，防止极端数值污染损失函数的计算。

</details>

24. 为什么防止 OOD (分布外动作) 可以帮助做到训练收敛？和梯度爆炸有关吗？
<details>
<summary>参考答案</summary>

防止 OOD 的核心作用是**切断了引发梯度爆炸的“毒水源”**。
在离线 RL 里，Q 网络只在数据集里见过的 $(s, a)$ 处是准确的。如果策略生成了一个 OOD 动作，Q 网络在未定义的区域很容易瞎猜出一个极大值（过估计）。
当这个极大的 $Q(s', a_{OOD})$ 被代入 $TD\_target$ 公式计算 Q 网络的 MSE 损失时，会产生极大的 Loss，进而计算出极大的梯度（也就是梯度爆炸）。更可怕的是，这会让整个 Q 网络面目全非，然后策略又会利用错误的 Q 网络继续寻找更奇怪的动作，陷入恶性循环。
因此，通过扩散模型的**行为克隆损失（diff_loss）**把生成的动作死死限制在真实数据分布内（防止 OOD），就能保证查询 Q 网络时总是得到相对准确、平稳的 Q 值，从而从根源上避免了异常大梯度和发散。

</details>

25. 策略梯度的损失函数中，是否有通过超参来控制 diff_loss 和 guide_loss 的权重？
<details>
<summary>参考答案</summary>

是的，项目在损失函数设计中使用了超参来显式控制两者的平衡。
在代码中，策略的总损失公式为：`policy_loss = diff_coef * diff_loss + guide_coef * guide_loss`。
其中 `diff_coef` 和 `guide_coef`（默认均为 1.0）直接控制了“拟合数据（行为克隆）”与“追求高回报（强化学习）”之间的权衡。
此外，针对 TD3 模式下的 `guide_loss = -λ * mean(Q)`，代码还引入了超参 **`alpha`**。为了避免 Q 值尺度不同导致的梯度爆炸或消失，代码中动态计算了自适应的缩放系数 `λ = alpha / stop_gradient(mean(|Q|))`。这意味着 `alpha` 决定了引导梯度的绝对强度，而不用担心 Q 值的绝对大小。
</details>


