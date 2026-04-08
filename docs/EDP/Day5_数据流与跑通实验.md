# Day 5：数据流 + 跑通一个小实验

> **目标**：搞清「从 D4RL/RLUP 读入离线数据 → 封装成可采样 Dataset → 每步 sample() 得到 batch → 喂给 agent.train(batch)」的完整数据路径，并在本机跑通一轮小规模训练。  
> **对应理论**：离线数据集、n-step 回报、batch 随机采样；不涉及扩散或 RL 的损失公式。  
> **说明**：Day 1～4 已理清「训练一步」在 dql.py 里做什么；Day 5 聚焦「数据从哪来、长什么样、如何被取用」。数据流与 `_setup_dataset`、`_setup_d4rl` 紧密相关，评估用 env 也在此阶段建好。

---

### Day 5 提问自测（先尝试自己回答）

> 建议：真正答题时只看这部分，不要往下翻到「参考答案」。

1. 从「D4RL 环境名」到「训练循环里拿到的一个 batch」，中间依次经过哪些函数/类？请按数据流向列出。
2. `get_d4rl_dataset` 返回的 dict 里有哪些 key？和 `agent.train(batch)` 需要的字段是否一致？
3. n-step 是在哪里做的？`get_nstep_dataset` 的输入、输出分别是什么？sorting 和 norm_reward 各起什么作用？
4. `data/dataset.py` 里的 `Dataset` 和 `utilities/traj_dataset.py` 里的 `Dataset` 是同一个类吗？本项目训练时用的是哪一个？
5. `RandSampler` 做什么用？`dataset.sample()` 内部是如何得到一批 indices 再取数据的？
6. `_setup_d4rl` 里除了拿到 dataset，还做了哪些事（env、reward、action、obs 等）？
7. D4RL 与 RLUP 在项目中的接入点分别是什么？RLUP 的 batch 结构和 D4RL 的有什么不同？
8. 若要「只跑通 pipeline、不关心分数」，你会怎么改命令行参数？跑完后应检查哪些输出或文件？
9. 用自己的话：从「离线数据在磁盘/API」到「agent.train(batch) 收到 JAX 数组」，数据经过了哪几层封装？
10. （加分）n-step 的 reward 公式是什么？为什么做 n-step 有利于离线 RL？
11. （补充）为什么 `traj_dataset.py` 里既有 `Dataset` 又有 `D4RLDataset`？它们各自负责哪一部分工作？  
12. （补充）`get_traj_dataset` 在整条「D4RL → 轨迹 → n-step → get_d4rl_dataset → Trainer」链路里的角色是什么？  
13. （补充）D4RL 原始数据里本身有没有「一条轨迹结尾」的标识？如果有，为什么预处理时还要在 `D4RLDataset` 里重新算 `dones_float`？  
14. （补充）`replay_buffer.py` 里除了 `get_d4rl_dataset` 还有什么？本项目主流程为什么只用它？

---

## 零、理论基础：离线数据与 n-step

> 以下为 Day 5 数据流涉及的最小概念，便于理解「数据从哪来、为何要 n-step」。

### 0.1 离线 RL 的数据假设

- 只有**固定数据集** $\mathcal{D}$，不与环境在线交互；每条样本形如 $(s, a, r, s', \mathrm{done})$（或带 discount）。
- D4RL 提供按**任务 + 质量**划分的数据集（如 `walker2d-medium-v2`）；RL Unplugged 是另一套离线 benchmark。
- 训练时：从 $\mathcal{D}$ 中**随机采样** batch，用于 value loss、diff loss、guide loss；评估时：用当前策略在**评估环境**里滚轨迹，算 return / normalized return。

### 0.2 n-step 与 reward

- **单步**：用 $r_t$ 和 $s_{t+1}$ 做 TD。
- **n-step**：用 $R_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1}$ 和 $s_{t+n}$ 做 TD，方差更小、bias 略大，常能提升离线 RL 表现。
- 代码里 n-step 在 **traj_dataset** 中按轨迹做：先按 trajectory 切分，再对每条轨迹的 reward 做卷积（`np.convolve(rewards, gammas)`），得到 n-step reward 序列；`next_observation` 取对应 n 步后的状态。

### 0.3 batch 内容

- 训练一步需要的 batch 至少包含：`observations`、`actions`、`next_observations`、`rewards`、`dones`（以及 IQL 等可能用到的 `discounts`）。  
- 所有数组在送入 `agent.train` 前会转成 JAX 可用的格式（如 `batch_to_jax(batch)`），维度为 `[batch_size, ...]`。

---

## 一、核心概念速览

### 1.1 数据流总览（D4RL 路径）

| 阶段 | 做什么 | 主要位置 |
|------|--------|----------|
| 1. 原始数据 | D4RL（`d4rl.qlearning_dataset`）或 RLUP 读入 (s,a,r,s',done)；D4RL 路径由 `get_d4rl_dataset` 入口，内部经 `get_traj_dataset`→`get_nstep_dataset` | `replay_buffer.get_d4rl_dataset`（内部调 `traj_dataset.get_nstep_dataset`） |
| 2. n-step + 排序 | 按轨迹切分、n-step reward、可选按 return 排序、可选 reward 归一化 | `traj_dataset.get_traj_dataset` + `get_nstep_dataset` |
| 3. 封装成 dict | 键：observations, actions, next_observations, rewards, dones(, dones_float) | `replay_buffer.get_d4rl_dataset` 返回值 |
| 4. 包装为 Dataset | 存 dict、支持按 indices 取子集 | `data/dataset.py` 的 `Dataset(data)` |
| 5. 挂接采样器 | 随机 indices，batch_size 个 | `RandSampler`，`dataset.set_sampler(sampler)` |
| 6. 每步取 batch | `dataset.sample()` → sampler.sample() → retrieve(indices) | 训练循环里 `self._dataset.sample()` |
| 7. 喂给 agent | `batch_to_jax(batch)` → `self._agent.train(batch)` | `trainer.py` 训练循环 |

### 1.2 D4RL 与 RLUP 的接入方式

| 数据源 | 入口函数 | 产出 | 备注 |
|--------|----------|------|------|
| **D4RL** | `get_d4rl_dataset(env, nstep, gamma, norm_reward)` | dict，再包成 `Dataset` + `RandSampler` | 内部调 `get_nstep_dataset`，sorting=True |
| **RLUP** | `_setup_rlup()` 里用 `RLUPDataset(...)` | 自带 `sample()` 的 dataset，直接作 `_dataset` | TensorFlow Datasets，batch 结构含 observations, actions, rewards, next_observations, dones 等 |

### 1.3 batch 的 key 与用途

- **observations**：当前状态，扩散条件、Q(s,a) 的 s。  
- **actions**：数据中的动作，扩散的 $x_0$、行为克隆目标。  
- **next_observations**：下一状态，TD target 里的 $s'$。  
- **rewards**：可为单步或 n-step 的 $r$（或 $R^{(n)}$）。  
- **dones**：终止标志，TD 里 $(1-\mathrm{done})\gamma$、轨迹划分等。

---

## 二、代码结构：replay_buffer 与 get_d4rl_dataset

### 2.1 位置与作用

**位置**：`utilities/replay_buffer.py`，`get_d4rl_dataset`（约 123 行起）。

**作用**：对外入口：给定 env、nstep、gamma、norm_reward，返回「已做 n-step、排序、归一化（可选）」的 dict，键为 `observations`, `actions`, `next_observations`, `rewards`, `dones`, `dones_float`，供后续包成 `Dataset` 使用。

### 2.2 逻辑简述

```python
# 始终走 n-step 路径，且 sorting=True
dataset = get_nstep_dataset(env, nstep, gamma, sorting=True, norm_reward=norm_reward)
return dict(
  observations=dataset["observations"],
  actions=dataset["actions"],
  next_observations=dataset["next_observations"],
  rewards=dataset["rewards"],
  dones=dataset["terminals"].astype(np.float32),
  dones_float=dataset["dones_float"],
)
```

- 注意：内部键 `terminals` 转为对外的 `dones`；`dones_float` 保留供轨迹逻辑使用。  
- **replay_buffer.py 在本项目的角色**：该文件还包含 `ReplayBuffer` 类以及 `index_batch`、`partition_batch_train_test`、`subsample_batch` 等 batch 工具函数；EDP 主训练流程**只用到 `get_d4rl_dataset`**，其余为通用 RL 工具箱，其他脚本或实验可能复用。

---

## 三、代码结构：traj_dataset 与 get_nstep_dataset

### 3.1 轨迹切分：split_into_trajectories

**位置**：`utilities/traj_dataset.py`，约 30–49 行。

**作用**：按 `dones_float[i]==1.0` 把整条序列切成多条轨迹，返回 `trajs`（list of list of transition tuples）。  
每个 transition 为 `(obs, action, reward, mask, done_float, next_obs)`。

### 3.2 get_traj_dataset

**位置**：`utilities/traj_dataset.py`，约 128–151 行。

**作用**：  
- 用 `D4RLDataset(env)`（本文件内的类，调 `d4rl.qlearning_dataset(env)`）得到原始数据；  
- 调 `split_into_trajectories(...)` 得到 `trajs`；  
- 若 `sorting=True`，按 `compute_returns(traj)` 对轨迹排序；  
- 若 `norm_reward=True`，用轨迹 return 的极差做归一化。  
返回 `trajs` 和 `raw_dataset`（原始 dict），供 `get_nstep_dataset` 使用。

**补充理解（和上游 D4RL 的关系）**：  
- 可以把一条「trajectory」基本当作一条 **episode** 看：D4RL 把很多 episode 的 transition 按时间**平铺在一起**，并提供 `terminals/timeouts` 等结尾标记。  
- `D4RLDataset` 在构造时会用 `d4rl.qlearning_dataset(env)` 读出原始 dict，并根据 `terminals` 和「状态是否连续」（`observations[i+1]` 是否≈`next_observations[i]`）重新算一个更鲁棒的 `dones_float`，用于判定「这一条是不是 episode 结尾」。  
- `get_traj_dataset` 调 `split_into_trajectories(..., dones_float, ...)`，用这些结尾标记把长序列切成多条轨迹；再根据需要做排序/归一化，作为 **get_nstep_dataset 的前置步骤**。

### 3.3 get_nstep_dataset

**位置**：`utilities/traj_dataset.py`，约 159–190 行。

**作用**：在**已切好的轨迹**上做 n-step：  
- 对每条轨迹的 reward 序列做 `np.convolve(rewards, gammas)[nstep-1:]` 得到 n-step reward；  
- `next_observations` 取「当前索引 + nstep-1」对应的 next_obs（若超出轨迹末尾则取最后一个）；  
- 输出 dict：`observations`, `actions`, `next_observations`, `rewards`, `terminals`, `dones_float`，与 `get_d4rl_dataset` 期望的输入一致。

**参数**：`env`（或 env 名）、`nstep`（默认 5）、`gamma`（默认 0.9）、`sorting`、`norm_reward`。

> 可以把 `get_nstep_dataset` 理解成「**底层加工函数：按轨迹切分 + 做 n-step 卷积 + 拼成一个内部用的 dict**」，而上层的 `get_d4rl_dataset` 再在它的基础上做一层**“改装/适配”**：统一键名、转成 float32 的 `dones` 等，方便 Trainer 这边直接使用。

---

## 四、代码结构：data/dataset.py 与 RandSampler

### 4.1 Dataset（data/dataset.py）

**位置**：`data/dataset.py`，约 24–55 行。

**作用**：把「键为 observations/actions/... 的 dict」包装成可按索引取 batch 的对象。

| 方法/属性 | 说明 |
|-----------|------|
| `__init__(self, data: dict)` | 存 `_data`、`_keys`，`_sampler` 初始为 None |
| `size()` | `len(self._data[self._keys[0]])` |
| `retrieve(indices)` | 对每个 key 做 `self._data[key][indices, ...]`，返回子 dict |
| `set_sampler(sampler)` | 设置 `_sampler` |
| `sample()` | 调用 `self._sampler.sample()` 得 indices，再 `self.retrieve(indices)` |

- 训练时用的 **Dataset** 是 **data/dataset.py** 的类；**utilities/traj_dataset.py** 里另有同名 `Dataset`（以及子类 `D4RLDataset`），只在「D4RL 原始数据 → 轨迹 → n-step」这条**预处理链路**里用。可以这样记：`traj_dataset.Dataset` 是「通用容器 + sample 工具」，`D4RLDataset` 是在构造时**专门负责“从 D4RL 读数据并填满这个容器”**的子类；采样逻辑是通用的，所以只写在父类里，子类复用即可。

### 4.2 RandSampler

**位置**：`data/sampler.py`，约 23–34 行。

**作用**：给定 `max_size` 和 `batch_size`，`sample()` 返回 `np.random.randint(self._max_size, size=self._batch_size)`，即一批随机索引。

- 在 trainer 中：`sampler = RandSampler(dataset.size(), self._cfgs.batch_size)`，再 `dataset.set_sampler(sampler)`，于是每次 `dataset.sample()` 得到 **batch_size** 个随机样本组成的 dict。

---

## 五、trainer 中的数据接入：_setup_dataset 与 _setup_d4rl

### 5.1 _setup_dataset

**位置**：`diffusion/trainer.py`，约 566–577 行。

**作用**：根据 `DATASET_MAP[self._cfgs.dataset]` 选择 D4RL 或 RLUP；设好 `_obs_mean/_obs_std/_obs_clip` 初值；调用 `_setup_d4rl()` 或 `_setup_rlup()`，返回 `(dataset, eval_sampler)`，赋给 `self._dataset` 和 `self._eval_sampler`。

### 5.2 _setup_d4rl（D4RL 路径）

**位置**：`diffusion/trainer.py`，约 482–547 行。

**流程概览**：

| 步骤 | 内容 | 约行号 |
|------|------|--------|
| 1 | `env = gym.make(env_id)`，失败则尝试 PyBullet fallback | 483–504 |
| 2 | `eval_sampler = TrajSampler(env, max_traj_length)` | 504 |
| 3 | `dataset = get_d4rl_dataset(eval_sampler.env, nstep, discount, norm_reward)` | 513–517 |
| 4 | reward scale/bias：`dataset["rewards"] *= reward_scale` 等；action clip | 519–522 |
| 5 | Kitchen/Adroit/Antmaze：obs 归一化、reward 进一步变换（如 IQL 时 Antmaze reward−1） | 524–540 |
| 6 | `dataset = Dataset(dataset)`（data 包里的 Dataset） | 544 |
| 7 | `sampler = RandSampler(dataset.size(), batch_size)`，`dataset.set_sampler(sampler)` | 545–546 |
| 8 | `return dataset, eval_sampler` | 547 |

- 因此：**数据**来自 `get_d4rl_dataset` → dict → `Dataset` + `RandSampler`；**评估环境**来自同一 env 封装的 `TrajSampler`，用于后续 `eval_sampler.sample(...)`。

---

## 六、动手运行：跑通小实验

### 6.1 依赖与环境

- 安装：`pip install -e .`；按 README 配置 MuJoCo 与 D4RL（或使用 PyBullet fallback）。  
- 若仅验证 pipeline，可减少 epoch 与每 epoch 步数，避免长时间等待。

### 6.2 小规模运行命令示例

```bash
python -m diffusion.trainer --env 'hopper-medium-v2' \
  --logging.output_dir './exp_debug' \
  --algo_cfg.loss_type=TD3 \
  --n_epochs=2 \
  --n_train_step_per_epoch=10
```

- 可根据本机环境换成 `walker2d-medium-v2` 等；无 MuJoCo 时部分 env 会自动 fallback 到 PyBullet（见 trainer 中 `_bullet_fallback`）。

### 6.3 跑完后检查

- 终端是否出现 `agent.*loss`、`average_normalizd_return`（代码里可能拼写如此）或类似指标。  
- `--logging.output_dir` 指定目录下是否生成日志文件。  
- 无报错且能完成 2 个 epoch、每次 10 step，即说明「数据 → sample → train」这条 pipeline 已通。

### 6.4 AutoDL 端完整实验结果小结

- **实验设置**：在 AutoDL 上使用 CUDA 12 镜像 + JAX GPU 版本（`jax[cuda12]`）、MuJoCo 2.1.0 + `mujoco_py`、`walker2d-medium-replay-v2`、`loss_type=TD3`，按原仓库默认配置运行 **2000 epoch × 1000 step/epoch** 的正式训练。  
- **小规模验证**：先用 2 epoch × 100 step 的命令在 AutoDL 上跑通，确认 `jax.devices()` 返回 `CudaDevice(0)`、D4RL 数据成功下载、终端能看到 `agent/*` 与 `average_normalizd_return` 等指标。  
- **最终结果**：在 Walker2d-medium-replay-v2 上得到  
  - `best_normalized_return ≈ 0.934`  
  - `average_normalizd_return ≈ 0.93`，与论文/官方实现报告的约 0.93 分数高度一致，说明复现成功。  
- **训练耗时**：在 1×RTX 4090 上，训练期每个 epoch 约 7–8 秒；最后一个 epoch 由于做了一次完整评估，`eval_time` 增大到约 30 秒，总 `epoch_time` ≈ 40 秒 属正常现象。  
- **总结**：从本机到 AutoDL，已经完整验证了「D4RL 数据 → get_d4rl_dataset → Dataset/RandSampler → DiffusionQL 训练 → D4RL normalized return」这一整条链路，在标准 locomotion 任务上达到接近论文的表现，可作为后续复试或扩展实验的展示结果。

---

## 七、当天需要能做到

- 用 **3–5 句话**说出：从 D4RL 环境名到 `agent.train(batch)` 收到 JAX batch，数据依次经过哪些函数/类（get_d4rl_dataset → get_nstep_dataset → Dataset → RandSampler → sample → batch_to_jax → train）。  
- 能区分：**data/dataset.py** 的 `Dataset`（trainer 用的）与 **utilities/traj_dataset.py** 里的 `Dataset`/`D4RLDataset`（仅 traj 与 n-step 内部用）。  
- 能指出：n-step 在 `get_nstep_dataset` 里做；`_setup_d4rl` 里还负责 env、reward/action 预处理、obs 归一化（部分域）、以及把 dict 包成 `Dataset`+`RandSampler`。  
- 能说明 **replay_buffer.py** 在本项目中的角色：主流程只用 `get_d4rl_dataset`；其余（ReplayBuffer、batch 工具函数）为通用工具箱，EDP 未用。  
- 实际跑通一次上述小实验，并会看终端或日志里的 loss / return 输出。

---

### Day 5 自测参考答案（折叠，答题时不要展开）

<details>
<summary>点击展开参考答案</summary>

1. **从 D4RL 到 batch 的依次经过**  
   - `_setup_d4rl()` 里：`get_d4rl_dataset(env, nstep, gamma, norm_reward)` → 内部 `get_nstep_dataset`（在 traj_dataset 里：`get_traj_dataset` → `split_into_trajectories` → n-step 卷积与拼装）→ 返回 dict。  
   - dict 经 reward/action 等预处理后，`Dataset(dataset)`（data/dataset.py）→ `RandSampler(dataset.size(), batch_size)` → `dataset.set_sampler(sampler)`。  
   - 训练循环：`batch = self._dataset.sample()` → `sampler.sample()` 得 indices → `retrieve(indices)` → `batch_to_jax(batch)` → `self._agent.train(batch)`。

2. **get_d4rl_dataset 返回的 key**  
   - `observations`, `actions`, `next_observations`, `rewards`, `dones`, `dones_float`。与 `agent.train(batch)` 所需一致（trainer 会做 batch_to_jax，字段名一致即可）。

3. **n-step 在哪里做；get_nstep_dataset 输入输出**  
   - n-step 在 **utilities/traj_dataset.py** 的 **get_nstep_dataset** 里做。  
   - 输入：env、nstep、gamma、sorting、norm_reward；内部先调 `get_traj_dataset` 得到 trajs 和 raw_dataset。  
   - 输出：dict，含 observations, actions, next_observations, rewards, terminals, dones_float。  
   - **sorting**：按轨迹 return 排序，便于 n-step 与数据质量相关处理；**norm_reward**：按轨迹 return 极差做 reward 归一化。

4. **两个 Dataset 是否同一个类**  
   - **不是**。**data/dataset.py** 的 `Dataset`：用 dict 存数据，提供 `size()`、`retrieve(indices)`、`set_sampler`、`sample()`，训练时用的 `_dataset` 是它。  
   - **utilities/traj_dataset.py** 的 `Dataset`：是另一类，存 observations/actions/... 等数组和 size，有 `sample(batch_size)`，仅在该文件内被 `get_traj_dataset` 使用（通过 D4RLDataset 继承）。

5. **RandSampler 与 dataset.sample()**  
   - RandSampler：给定 max_size、batch_size，`sample()` 返回 `np.random.randint(max_size, size=batch_size)` 的索引数组。  
   - `dataset.sample()`：先 `indices = self._sampler.sample()`，再 `return self.retrieve(indices)`，即用这批索引从 `_data` 里按 key 取子集，得到一个 batch 的 dict。

6. **_setup_d4rl 里还做了哪些事**  
   - 建 env（含 PyBullet fallback）、建 `TrajSampler` 作 eval_sampler；  
   - 对 get_d4rl_dataset 返回的 dict 做：reward scale/bias、action clip；  
   - Kitchen/Adroit/Antmaze：obs 归一化、reward 再变换（如 Antmaze 下 IQL 时 reward−1）；  
   - 最后用 `Dataset(dataset)` + `RandSampler` 挂接，返回 dataset 和 eval_sampler。

7. **D4RL 与 RLUP 接入点**  
   - **D4RL**：`_setup_d4rl()` → `get_d4rl_dataset` → dict → `Dataset` + `RandSampler`。  
   - **RLUP**：`_setup_rlup()` → `RLUPDataset(...)`，产出自带 `sample()` 的 dataset（TensorFlow Datasets），直接作 `_dataset`；评估 env 用 `DM2Gym(dataset.env)` + `TrajSampler`。  
   - RLUP batch 也含 observations, actions, rewards, next_observations 等，但来自 TFDS，形状/来源与 D4RL 不同。

8. **只跑通 pipeline 的参数与检查**  
   - 减小 `n_epochs`（如 2）、`n_train_step_per_epoch`（如 10），保证很快跑完。  
   - 检查：终端有 loss、average_normalizd_return 等输出；无异常退出；`--logging.output_dir` 下生成日志。

9. **从离线数据到 agent.train(batch) 的封装层次**  
   - 第一层：D4RL/RLUP API 或文件 → 原始 (s,a,r,s',done) 或等价的 dict；  
   - 第二层：n-step、排序、归一化 → 仍是 dict，键统一；  
   - 第三层：`Dataset(dict)`，支持 size/retrieve；  
   - 第四层：`RandSampler` 提供随机 indices，`sample()` = retrieve(sampler.sample())；  
   - 第五层：训练循环里 `batch_to_jax(batch)` 转成 JAX 数组，再 `agent.train(batch)`。

10. **n-step reward 公式与作用**  
    - $R_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1}$；next state 取 $s_{t+n}$。  
    - n-step 降低 TD 目标方差（平均多步 reward），常用于离线 RL 以稳定训练、提升表现。

11. **traj_dataset 里 Dataset 与 D4RLDataset 的分工**  
    - **Dataset（父类）**：通用「数组容器 + sample 工具」——不关心数据从哪来，只要给好 observations/actions/... 等数组，就提供 `sample(batch_size)` 随机抽样。  
    - **D4RLDataset（子类）**：专门负责「从 D4RL 读数据并填满这个容器」——构造时调 `d4rl.qlearning_dataset(env)`，做 clip、算 `dones_float` 等预处理，再 `super().__init__(...)` 把数组交给父类。采样逻辑通用，故只写在父类，子类复用。

12. **get_traj_dataset 在链路里的角色**  
    - 是 **get_nstep_dataset 的前置步骤**：先用 D4RLDataset 读出 D4RL 原始数据，用 `split_into_trajectories(..., dones_float, ...)` 按 episode 切成长序列 → 多条轨迹；再按需做 sorting、norm_reward。  
    - 输出 `trajs` 和 `raw_dataset`，供 get_nstep_dataset 在每条 traj 上做 n-step 卷积与拼装；整条链是 D4RL → get_traj_dataset → get_nstep_dataset → get_d4rl_dataset → Trainer。

13. **D4RL 里有没有轨迹结尾标识？为何还要重算 dones_float？**  
    - **有**：`d4rl.qlearning_dataset(env)` 返回的 dict 里有 `terminals`（以及部分环境有 `timeouts`），可以据此判断某条 transition 是不是 episode 结尾。  
    - **仍要重算 dones_float 的原因**：① 只看 `terminals` 可能漏掉「因 timeout 结束」等情形；② 用「状态是否连续」辅助判断——同一条 episode 内应有 `observations[i+1] ≈ next_observations[i]`，若明显不连续说明发生了 episode 切换，即便 terminals 未标也当作结尾；③ 把多种情况统一成一个 0/1 标记 `dones_float`，后续切轨迹、n-step 边界都只认这一个字段，逻辑更简单、更鲁棒。

14. **replay_buffer.py 里还有什么？为何主流程只用 get_d4rl_dataset？**  
    - 该文件还包含：**ReplayBuffer** 类（可 add_sample/add_batch、sample、generator 等），以及 **index_batch**、**partition_batch_train_test**、**subsample_batch**、**concatenate_batches**、**split_batch**、**split_data_by_traj** 等 batch/数据划分工具函数。  
    - 主流程只用 **get_d4rl_dataset** 是因为：EDP 训练管线只需要「从 D4RL 读数据并整理成统一 dict」这一入口，后续用 `data.Dataset` + `RandSampler` 采样；ReplayBuffer 与上述工具函数是通用 RL 工具箱，本脚本未调用，其他实验或脚本可能复用。

</details>

---

## 八、补充：代码与概念澄清

### 8.1 Dataset 的 size 与 keys

- `Dataset._data` 是 dict，`_keys = list(data.keys())`；`size()` 取任一 key 的长度（所有 key 长度一致）。  
- `retrieve(indices)` 对每个 key 做 `self._data[key][indices, ...]`，要求各 key 在首维上长度相同。

### 8.2 get_d4rl_dataset 与 get_nstep_dataset 的调用关系

- **replay_buffer.get_d4rl_dataset** 内部只调 **traj_dataset.get_nstep_dataset**，不直接调 `get_traj_dataset`；`get_nstep_dataset` 内部会调 `get_traj_dataset`，再在 trajs 上做 n-step 拼装。

### 8.3 训练循环里 batch 的来源

- 训练循环在 **trainer.py**：每步 `batch = self._dataset.sample()`，再 `batch_to_jax(batch)` 转成 JAX 可用的格式后传入 `self._agent.train(batch)`。  
- 因此「数据流」的终点就是 **dql.py** 的 `DiffusionQL.train(batch)` → `_train_step_*` 里用到的 observations, actions, next_observations, rewards, dones。

### 8.4 RLUPDataset 与 data/dataset.py

- **RLUPDataset**（data/dataset.py）是另一套：从 TensorFlow Datasets 读 RL Unplugged，内部有 `sample()` 返回 numpy batch，不经过 RandSampler。  
- 当 `dataset_type == DATASET.RLUP` 时，`_setup_rlup()` 返回的 `dataset` 即 RLUPDataset 实例，直接赋给 `self._dataset`。

### 8.5 batch_to_jax 做什么？

- 在 **utilities/jax_utils.py** 中：`batch_to_jax(batch)` 用 `jax.tree_util.tree_map(jax.device_put, batch)` 把 batch 里每个 value（numpy 数组）搬到 JAX 设备上并转成 JAX 数组类型；**结构（dict、key、shape）不变**，只是元素从 `np.ndarray` 变为 JAX 可参与 `jit`/`grad` 的数组，供 `agent.train(batch)` 使用。

### 8.6 replay_buffer.py 文件角色

- 该文件提供 **get_d4rl_dataset**（EDP 主流程使用的唯一入口）以及 **ReplayBuffer**、**index_batch** 等通用工具；主训练管线只依赖 get_d4rl_dataset，其余为可选工具箱。

---

## 📖 参考文档与代码

- **理论**：`docs/EDP_复试梳理与演讲稿.md`  
  - **3.1 数据流（训练）**、**4.3 流程概览** 中数据与 _setup 相关行  
- **一周计划**：`docs/EDP_一周学习与复现计划.md`  
  - **Day 5** 小节、**项目执行顺序与代码文件对照** 中 2.1 数据准备  
- **代码**：  
  - `utilities/replay_buffer.py`：`get_d4rl_dataset`（约 123 行起）  
  - `utilities/traj_dataset.py`：`split_into_trajectories`、`get_traj_dataset`、`get_nstep_dataset`（约 30、128、159 行起；行号随注释可能有偏移）  
  - `utilities/jax_utils.py`：`batch_to_jax`（约 80 行起）  
  - `data/dataset.py`：`Dataset`（约 24–55 行）  
  - `data/sampler.py`：`RandSampler`（约 23–34 行）  
  - `diffusion/trainer.py`：`_setup_dataset`、`_setup_d4rl`（约 566–577、482–547 行）

---

## 🚀 下一步（Day 5.5 预告）

Day 5.5 将学习：
- 评估时多种采样方式（ddpm / dpm / ensemble）与 EAS（多采动作用 Q 选）
- 论文 4.5 节与「All results based on EAS」的含义
- `SamplerPolicy` 的 `*_act` 与评估循环里 `act_method`、指标记录

**今天把数据流和跑通实验搞定，下一节看评估与 EAS！**

---
