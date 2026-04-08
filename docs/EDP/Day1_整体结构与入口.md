## Day 1：整体结构、入口与执行流程

> 目标：弄清楚「这个项目整体在做什么」和「从命令行到训练/评估的完整执行顺序」，先只看骨架，不抠公式。

---

### Day 1 提问自测（先尝试自己回答）

> 建议：真正答题时只看这部分，不要往下翻到「参考答案」。

1. 用不超过 3 句话，说明 EDP 这个项目整体在干什么？（可以提到扩散策略、离线 RL、三块主要贡献）  
2. 在代码执行顺序里，从命令行 `python -m diffusion.trainer ...` 到真正开始**第一个 epoch**的训练，中间依次会调用哪些关键函数/方法？请按顺序说出它们的名字。  
3. `DiffusionTrainer.__init__` 和 `DiffusionTrainer._setup` 分别负责什么？为什么要把「读配置」和「真正建网络/数据」分成两步？  
4. 在 `_setup_dataset` / `_setup_d4rl` 里，D4RL 数据和评估环境是怎么被接入到整个训练流程里的？（数据流和 env 入口分别是什么对象？）  
5. 简单说说「训练阶段」和「生成/评估阶段」在实际代码里分别对应哪些部分？（不用讲公式，只要指出大概在哪些函数里发生）  
6. 你现在如何区分这三个角色：`DiffusionPolicy`、`DiffusionQL`、`DiffusionTrainer`？它们各自的一句话职责是什么？  
7. EAS（Energy-based Action Selection）在这个项目里扮演了什么角色？它是训练算法的一部分，还是评估策略的一部分？大致是怎么做的？  
8. 为什么说扩散策略是一个「分布」而不是一个「确定函数」？在评估时我们是怎么从这个分布里拿到具体动作的？  
9. 从离线数据的角度看，D4RL / RLUP 这两个数据源在项目中的位置分别是什么？在没有与环境在线交互的前提下，我们是如何完成训练和评估的？  
10. 如果复试老师问你：「请你从代码角度，把 EDP 的完整执行流程（从命令行到输出 normalized return）用 1–2 分钟讲一遍」，你会按什么顺序讲？列出你会提到的至少 5 个关键节点（函数/文件级别）。  
11. （加分）你能否用一句话概括：Day 1 你主要学会了什么？后面的几天你打算在这个基础上进一步搞清哪些问题？  
12. （加分）`_setup()` 里为什么先 `_setup_dataset` / `_setup_policy` / `_setup_qf`，最后才 `_algo(...)`？这些步骤之间的依赖关系是什么？

---

### 1. EDP 在干什么（高层目标）

- 用**扩散模型**做策略：$\pi(a\mid s)$ 不再是简单高斯，而是「给定状态 s，通过扩散去噪从噪声生成动作 a」的复杂分布。  
- 结合**离线 RL**（只用 D4RL/RLUP 的旧数据）：
  - 扩散 loss：拟合数据动作分布（行为克隆 + ELBO→MSE）。  
  - Guide loss（TD3/CRR/IQL）：用 Q/优势把策略往高回报方向拉。  
- 论文三块贡献：
  - **训练加速**：action approximation（训练时算 guide loss 不跑完整反向链，用预测的 $\hat{x}_0$）。  
  - **生成加速**：DPM-Solver / DDIM 少步采样。  
  - **算法兼容性**：通过 ELBO 近似 log π，让扩散策略能接 CRR/IQL 这类需要 log π 的算法，而不仅是 TD3。

---

### 2. 执行流程总览（从命令行到评估）

1. **程序入口**  
   - 命令行：`python -m diffusion.trainer --env ...`。  
   - 文件：`diffusion/trainer.py`  
     - `if __name__ == '__main__': main()` → `trainer = DiffusionTrainer()` → `trainer.train()`。

2. **Trainer.__init__（只做配置，不建网络）**  
   - 文件：`diffusion/trainer.py`，`DiffusionTrainer.__init__`。  
   - 工作：
     - 读 `absl.flags.FLAGS`；  
     - 写回一些超参（如 `algo_cfg.max_grad_norm`、`lr_decay_steps`）；  
     - 选择激活函数：默认 `activation="mish"`，否则用 `jax.nn.<name>`；  
     - 记录 env 类型（`ENV_MAP`）。  
   - 此时**还没有** `_dataset/_policy/_agent`，只是准备好配置。

3. **_setup()（真正建数据、网络和 agent）**  
   - 在 `train()` 的第一行调用：`self._setup()`。  
   - 内部依次：
     - `_setup_logger()`：建 WandBLogger + 本地 logger。  
     - `_setup_dataset()`：
       - 调 `_setup_d4rl()` 或 `_setup_rlup()`：
         - `utilities/replay_buffer.get_d4rl_dataset(env, nstep, gamma, norm_reward)`：从 D4RL 读离线数据、做 n-step。  
         - 套一层 `Dataset` + `RandSampler`，得到可 `sample()` 的数据集。  
         - 用 `TrajSampler(env, max_traj_length)` 封装评估用 env，记为 `self._eval_sampler`。  
       - 记录观测维度、动作维度、最大动作、target_entropy。  
     - `_setup_policy()`：建 `GaussianDiffusion` + `DiffusionPolicy`（扩散策略网络）。  
     - `_setup_qf()` / `_setup_vf()`：建 `Critic`（qf1/qf2）和 `Value`（V，IQL 用）。  
     - `_algo(...)`：用上面这些模块构造 `DiffusionQL` agent。  
     - 建 `SamplerPolicy(self._agent.policy, self._agent.qf)`：评估时「obs → action」的封装。

4. **训练循环（每个 epoch）**  
   - 训练段：
     - 每步：`batch = self._dataset.sample()` → `batch_to_jax(batch)` → `metrics.update(self._agent.train(batch))`。  
     - `DiffusionQL.train(batch)` 内部调用 `_train_step` → `_train_step_td3/_crr/_iql`：  
       - `get_value_loss`：Q/V 的 TD 更新。  
       - `get_diff_loss`：扩散 MSE + pred_astart（action approximation）。  
       - guide loss：TD3/CRR/IQL 各自的定义。  
   - 评估段（按 `eval_period`）：
     - 为每个 `act_method`（ddpm/dpm/ensemble 等）设置 `self._sampler_policy.act_method`；  
     - 调 `self._eval_sampler.sample(self._sampler_policy.update_params(self._agent.train_params), eval_n_trajs, ...)`：  
       - 每一步：`SamplerPolicy(obs)` → 调 `DiffusionPolicy.ddpm_sample/dpm_sample` 生成动作（+ 可选 Q 选动作 = EAS）；  
       - 收集完整轨迹。  
     - 计算：return / traj_length / **normalized return**（用 D4RL 提供的 `get_normalized_score`）。

5. **保存（可选）**  
   - 若配置 `save_model=True`，在 `trainer.py` 里保存 agent 参数、配置等。

---

### 3. 你已经掌握的 Day 1 关键点回顾

- 知道「**谁是模型**」（`DiffusionPolicy`）、谁是「**算法壳**」（`DiffusionQL`）、谁负责「**流程调度**」（`DiffusionTrainer` + `SamplerPolicy`）。  
- 能说出训练 vs 生成在流程里的位置：  
  - 训练（阶段 3）：用 `training_losses` 做噪声 MSE + 在 `dql.py` 里算 guide loss，不跑完整去噪链；  
  - 生成（阶段 4）：评估时 `SamplerPolicy` 用 DDPM / DPM-Solver 完整去噪生成动作。  
- 明白 D4RL/RLUP 是**离线数据源**，通过 `_setup_dataset` → `Dataset`/`TrajSampler` 接好；评估 env 是在 `_setup_d4rl/_setup_rlup` 里 `gym.make(...)` 出来的。  
- 理解 EAS 的位置：评估时多采样 + 用 Q 选动作，是**在 D4RL normalized score 之前的一层“选动作策略”**，不是新的训练算法。

后面各天的内容（扩散细节、TD3/CRR/IQL 三种 guide loss、action approximation、DPM-Solver 等）会分别放在对应的 DayN 文档里逐步补全。暂时 Day 1 先只牢牢抓住「整体流程和各文件分工」即可。

---

### Day 1 自测参考答案（折叠，答题时不要展开）

<details>
<summary>点击展开参考答案</summary>

1. **EDP 在干什么**  
   - 用扩散模型表示策略 $\pi(a\mid s)$，在离线强化学习设定下，从旧数据中学一个高表达力的策略。  
   - 通过扩散 MSE 拟合数据分布 + TD3/CRR/IQL 的 guide loss 把策略往高 Q / 高优势方向拉。  
   - 论文的三块贡献：训练加速（action approximation）、生成加速（DPM-Solver / DDIM）、算法兼容性（用 ELBO 近似 log π，使扩散策略能接 CRR/IQL）。

2. **从命令行到训练的调用顺序**  
   - 命令行：`python -m diffusion.trainer --env ...`。  
   - `diffusion/trainer.py`：`if __name__ == '__main__': main()`。  
   - `main()` 里：`trainer = DiffusionTrainer()`（调用 `__init__`，只做配置）。  
   - 然后 `trainer.train()`：第一行 `self._setup()`（真正建 logger / dataset / env / policy / Q/V / DiffusionQL / SamplerPolicy）。  
   - 接着在 `train()` 里进入 epoch 循环：每步 `batch = self._dataset.sample()` → `self._agent.train(batch)`；按 eval_period 做评估循环。

3. **__init__ vs _setup 的区别**  
   - `__init__`：只负责**读和整理配置**（FLAGS）、选激活函数、写回一些衍生超参（如 `max_grad_norm`、`lr_decay_steps`）、确定 env 类型；不创建任何大对象（没有 dataset/policy/agent）。  
   - `_setup`：在真正训练前，一次性**构建所有重型组件**（logger、Dataset+TrajSampler、DiffusionPolicy、Critic/Value、DiffusionQL、SamplerPolicy）。  
   - 分离的好处：配置和构建解耦；便于测试、便于在不改构造逻辑的前提下调整 `_setup_*`。

4. **D4RL 数据和评估环境是如何接入的**  （不懂！！！！！！感觉确实可能会问到）
   - 在 `_setup_d4rl()` 里：  
     - `env = gym.make(env_id)`（必要时用 PyBullet fallback），再用 `TrajSampler(env, max_traj_length)` 封装为 `eval_sampler`；  
     - `dataset = get_d4rl_dataset(env, nstep, discount, norm_reward)` 从 D4RL 读取离线数据并做 n-step；  
     - 用 `Dataset(dataset)` + `RandSampler` 封装为可 `sample()` 的 dataset。  
   - `_setup_dataset()` 统一：根据 `DATASET_MAP` 选 D4RL 或 RLUP 路径，返回 `self._dataset` 和 `self._eval_sampler`；并用 `eval_sampler.env` 的 space 设置 `_observation_dim/_action_dim/_max_action`。

5. **训练 vs 生成在代码里的位置**  
   - 训练：`DiffusionTrainer.train` 的训练段（阶段 3）：  
     - `dataset.sample()` 抽 batch；  
     - `self._agent.train(batch)` 调用 `DiffusionQL.train`，内部 `_train_step_*`：  
       - `get_value_loss`：Q/V 的 TD 更新；  
       - `get_diff_loss`：扩散 MSE + pred_astart；  
       - guide loss：TD3/CRR/IQL 的 RL 头。  
   - 生成/评估：`train()` 里的评估段（阶段 4）：  
     - `self._eval_sampler.sample(self._sampler_policy.update_params(self._agent.train_params), ...)`：  
       - 每步 `SamplerPolicy(obs)` → `DiffusionPolicy.ddpm_sample/dpm_sample`（+ Q 做 EAS）；  
       - 收集轨迹、算 normalized return。

6. **DiffusionPolicy / DiffusionQL / DiffusionTrainer 的一句话职责**  
   - `DiffusionPolicy`：给定状态和噪声步，从噪声逐步去噪生成动作的「扩散策略网络」，同时提供训练用的 loss（噪声 MSE）。  
   - `DiffusionQL`：把扩散策略和 Q/V 组装成离线 RL 算法的「算法壳」，在扩散 MSE 上叠加 TD3/CRR/IQL 的 value loss + guide loss。  
   - `DiffusionTrainer`：从命令行入口调度整个流程的「训练器」，负责建 env / dataset / 网络 / agent / SamplerPolicy，并跑训练和评估循环。

7. **EAS 的角色**  
   - EAS = Energy-based Action Selection，是**评估/控制时的选动作机制**，不是训练算法本身的一部分。  
   - 做法：给定状态 s，从策略采 N 个候选动作 $a_i$，用 Q(s,a_i) 打分，然后选 Q 最大的，或按 $\propto e^{Q(s,a_i)}$ 抽样。  
   - 论文声明 “All results will be reported based on EAS”，说明所有表里分数都是在评估时用这套「多采样 + Q 选」机制算出来的。

8. **为什么说扩散策略是一个分布？评估时如何拿到动作？**  
   - 扩散策略给定状态 s，会通过一条噪声链从 $x_T$ 去噪到 $x_0$，这本质上定义了一个在动作空间上的复杂分布 $\pi(a\mid s)$；每次采样的轨迹不同，得到的动作也可能不同。  
   - 评估时：`SamplerPolicy(obs)` 调用 `DiffusionPolicy.ddpm_sample/dpm_sample` 等采样函数，从这个分布里采一个（或多个，再用 Q 选），得到具体的动作。

9. **D4RL / RLUP 的位置，以及如何离线完成训练和评估**  
   - 它们是**离线数据源**：给出很多条预先收集好的 $(s,a,r,s')$ 轨迹，训练时只从这些数据里采样 batch，不和环境实时交互。  
   - 训练：用 `get_d4rl_dataset` / `RLUPDataset` 把数据读进来，`Dataset.sample()` 抽 batch，喂给 `DiffusionQL.train(batch)`；  
   - 评估：用同一个 env（`eval_sampler.env`）配合当前策略 `SamplerPolicy` 去滚新轨迹，再用 D4RL 的 normalized score 来打分。

10. **1–2 分钟版执行流程（5 个关键节点）**  
   - 入口：`trainer.py` 的 `main()` → `DiffusionTrainer.__init__`（配置）→ `train()`。  
   - 初始化：`_setup()` 里 `_setup_logger/_setup_dataset/_setup_policy/_setup_qf/_setup_vf/_algo(...)`；dataset 和 eval_sampler 在这里建好。  
   - 训练循环：每步 `Dataset.sample()` → `DiffusionQL.train(batch)`（value loss + diff loss + guide loss）。  
   - 评估循环：`eval_sampler.sample(SamplerPolicy(update_params(...)))`，SamplerPolicy 用扩散策略和 Q 做动作生成/EAS。  
   - 指标：在 `trainer.py` 里统计 return、traj length、normalized return（D4RL 提供的评分函数）。

11. **Day 1 的收获与后续计划**（示例）  
   - 今天主要学会：把 EDP 看成「扩散策略（policy）+ DiffusionQL 算法壳 + trainer 流程调度」，能从命令行一路说到 normalized return。  
   - 后面几天要在这个结构上进一步搞清：扩散训练细节（ELBO→MSE）、TD3/CRR/IQL 三种 guide loss 的具体形式、action approximation 怎么加速训练、DPM-Solver/DDIM 怎么加速生成，以及 CRR/IQL + ELBO 近似 log π 如何实现算法兼容性。

12. **_setup 的调用顺序与依赖**  
   - `_setup_dataset` 最先：提供观测/动作维度、max_action、target_entropy 等，供后续建网络用；同时建好 `eval_sampler`。  
   - `_setup_policy`：需要 observation_dim、action_dim 等（来自 dataset/env），建 DiffusionPolicy。  
   - `_setup_qf` / `_setup_vf`：同样需要维度和 max_action，建 Critic 和 Value。  
   - `_algo(...)` 最后：把已建好的 policy、qf、vf、dataset 等全部传进去，组装成 DiffusionQL agent。  
   - 因此顺序是「先数据与 env → 再各网络模块 → 最后算法壳聚合」。

</details>