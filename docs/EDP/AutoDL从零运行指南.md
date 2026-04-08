# AutoDL 从零运行 EDP 指南

> 目标：在 AutoDL 上从零配好环境并跑通 EDP 小实验（可选 MuJoCo 或 PyBullet），以及跑完整项目的正式训练。

---

## 零、重要概念：AutoDL 的「系统盘」和「数据盘」

**重要**：以下提到的「系统盘」和「数据盘」都是 **AutoDL 远程机器上的磁盘**，和你的本机 Windows 系统盘完全无关。

- **系统盘**：租用实例时默认分配（通常 20–50G），用于操作系统、预装框架、你创建的环境等。这是实例的「根目录」所在。
- **数据盘**：额外挂载的一块盘（AutoDL 免费提供 50G），用于存代码、数据集、实验结果等。通常挂载在 `/root/autodl-tmp` 或类似路径。

**对 EDP 项目**：50G 免费数据盘足够（D4RL 数据集几 GB，训练日志/checkpoint 通常不会超过 50G）。

---

## 一、AutoDL 准备：开实例、进终端

### 1.1 注册与开机

1. 打开 [AutoDL 官网](https://www.autodl.com/)，注册/登录。
2. 进入 **控制台** → **实例** → **租用新实例**。
3. 选择：
   - **地域**：选延迟低的（如北京、上海）。
   - **GPU**：
     - **小实验**：RTX 4090 / 3090 / 3080 均可。
     - **跑整个项目（正式训练）**：**1×RTX 4090** 足够（EDP 是单卡 JAX，不需要多 GPU）。
   - **镜像**：**优先选 `JAX → 2.1.2`**（已预装 CUDA + jax/jaxlib，兼容 RTX 4090）。  
     备选：`Miniconda 2.5.1`（干净环境，需自己装 JAX+CUDA，稍复杂）。  
     不推荐：PyTorch/TensorFlow 镜像（除非你同时要跑 PyTorch 项目）。
   - **数据盘**：**免费 50G 足够**（D4RL 数据集小，训练日志/checkpoint 通常不会超过 50G）。
   - **系统盘**：默认即可（通常 20–50G，装环境用）。
4. 开机后，在实例列表里看到 **SSH 连接** 或 **打开 JupyterLab**。

### 1.2 进入“终端”（取得远程操作权）

- **方式 A**：点 **JupyterLab** → 菜单里 **File → New → Terminal**，会打开一个网页里的 Linux 终端。
- **方式 B**：用 SSH（实例页会给出命令，例如）：
  ```bash
  ssh -p <端口> root@region-xx.autodl.com
  ```
  在本地 PowerShell/WSL 执行后，输入密码即可进同一台机器的 shell。

之后所有“在 AutoDL 上”的操作，都是在这个终端里执行。

---

## 二、把项目弄到 AutoDL 上（用你本机改过的即可）

**不必重新从 GitHub 下原始仓库。** 用你现在本机已经改过、能跑通的那份最好，这样 JAX、PyBullet 回退、tree_map 等修改都在，避免在 AutoDL 上再踩一遍坑。

任选一种方式把「整个项目目录」放到 AutoDL 的某个目录下（例如 `/root/edp`）。

### 方式 1：本机打 zip 上传（最直接）

1. **本机**：在 `e:\computer_learning\projects\` 下，把 `edp` 文件夹打成 zip（可排除 `edp-venv-py310-wsl`、`experiment_output`、`__pycache__` 等，减小体积）。
2. **AutoDL**：  
   - 若用 JupyterLab：左侧文件树点 **Upload**，选该 zip，上传到 `/root/`（或你当前工作目录）。  
   - 上传后在终端里：
     ```bash
     cd /root
     unzip edp.zip
     mv edp-xxx edp   # 若解压出来带时间戳之类，改名为 edp
     cd edp
     ls
     ```
   确认能看到 `diffusion/`、`requirements.txt`、`scripts/` 等。

### 方式 2：Git（你本机已提交到自己的 GitHub/Gitee）

1. 本机先把你当前改过的代码推到自己的仓库（若还没推）：
   ```bash
   cd /mnt/e/computer_learning/projects/edp
   git remote add myrepo <你的仓库 URL>
   git add .
   git commit -m "local fixes for jax/d4rl"
   git push myrepo main
   ```
2. AutoDL 终端：
   ```bash
   cd /root
   git clone <你的仓库 URL> edp
   cd edp
   ```

### 方式 3：AutoDL 数据盘 / 网盘同步

若你用的是 AutoDL 的「数据盘」或绑定了网盘，可把本机 `edp` 同步到数据盘。  
**注意**：数据盘通常挂载在 `/root/autodl-tmp`（或按 AutoDL 文档说明的路径），你可以：
- 把项目放在数据盘：`/root/autodl-tmp/edp`（这样关机后数据盘内容会保留）
- 或放在系统盘：`/root/edp`（关机后可能清空，以平台说明为准）

建议：**代码和实验结果都放在数据盘**，避免关机丢失。

---

## 三、在 AutoDL 上配 Python 环境（与本机思路一致）

以下均在 **AutoDL 终端** 里执行（路径假设项目在 `/root/edp`，若你在别处请替换）。

### 3.1 确认 Python 版本

```bash
cd /root/edp
python3 --version
```

需要 **3.9 或 3.10**。若是 3.11+，需自建 3.10 环境，例如：

```bash
# 若没有 conda，可先安装 miniconda 或使用系统 py3.10
sudo apt-get update && sudo apt-get install -y python3.10 python3.10-venv
```

### 3.2 创建虚拟环境并激活

```bash
cd /root/edp
python3.10 -m venv venv
source venv/bin/activate
```

（若系统默认就是 3.10，可用 `python3 -m venv venv`。）

### 3.3 安装依赖

**若你选的是 `JAX 2.1.2` 镜像**（推荐）：

```bash
pip install --upgrade pip setuptools wheel

# 先锁 Cython 和 NumPy，避免和 mujoco_py / TensorFlow 冲突
pip install 'Cython<3' 'numpy<2'

# 项目依赖（注意：不要升级 jax/jaxlib，用镜像自带的 GPU 版本）
pip install -r requirements.txt

# 额外（data/dataset 和 dql 用）
pip install tensorflow tensorflow_datasets ml_collections

# 项目可编辑安装
pip install -e .
```

**若你选的是 `Miniconda` 或其他镜像**（需自己装 JAX+CUDA）：

- **方式 A（推荐，一步到位）**：若镜像/驱动支持 CUDA 12（驱动 ≥525），可直接用官方扩展安装（会装最新 jax + GPU 版 jaxlib）：
  ```bash
  pip install --upgrade pip setuptools wheel
  pip install 'Cython<3' 'numpy<2'
  pip install --upgrade "jax[cuda12]"
  pip install -r requirements.txt
  pip install tensorflow tensorflow_datasets ml_collections
  pip install -e .
  ```
- **方式 B（实例是 CUDA 11.8 且暂时不能换镜像时；若可换镜像，优先用 CUDA 12 镜像 + 方式 A）**：JAX 官方 CUDA 页面（`jax_cuda_releases.html`）**只提供到 jaxlib 0.4.x**，没有 0.6.2 的 CUDA wheel。只能在当前页面用「jax 0.4.25 + jaxlib 0.4.25 CUDA 11」组合，并装兼容 jax 0.4 的 optax（例如 0.2.0）：
  ```bash
  pip install --upgrade pip setuptools wheel
  pip install 'Cython<3' 'numpy<2'

  pip uninstall jax jaxlib optax chex -y
  # 1）装 jax 0.4.25（与下面 jaxlib 配套）
  pip install jax==0.4.25 -i https://pypi.org/simple --no-deps
  # 2）装 CUDA 11 版 jaxlib（该页面 cuda11 最高为 0.4.25+cuda11.cudnn86）
  pip install "jaxlib==0.4.25+cuda11.cudnn86" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --no-index
  # 3）装兼容 jax 0.4 的 optax（0.2.7 要求 jax>=0.5.3，故用 0.2.0）
  pip install "optax>=0.2.0,<0.2.7" chex

  pip install -r requirements.txt
  pip install tensorflow tensorflow_datasets ml_collections
  pip install -e .
  ```
  验证 GPU：`python -c "import jax; print(jax.devices())"` 应看到 `CudaDevice`/GPU。若运行训练时报 optax/jax API 不兼容，再考虑换 CUDA 12 镜像用方式 A。

若报错缺少系统库，可先装：

```bash
sudo apt-get update
sudo apt-get install -y build-essential libgl1-mesa-dev libosmesa6-dev patchelf
```

### 3.4（可选）安装 MuJoCo 2.1.0，用正式 D4RL 环境

若希望用 `walker2d-medium-replay-v2` 等 MuJoCo 环境（和论文一致），在 AutoDL 上装一次 MuJoCo 即可：

```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
# 若解压出来是 mujoco210 目录则不用改；若是别的名字，改成 mujoco210
mv mujoco210_linux mujoco210 2>/dev/null || true
cd mujoco210/bin
ln -sf libmujoco210.so libmujoco.so
cd /root/edp
```

之后安装 **mjrl**（D4RL 注册 MuJoCo 环境需要，否则会报 `No module named 'mjrl'` 且 `walker2d-medium-replay-v2` 等未注册）：

```bash
pip install git+https://github.com/aravindr93/mjrl.git
```

之后在运行训练前，需要设置环境变量（见下）。若不装 MuJoCo 或 mjrl，训练时会报错并提示安装，不会再用 PyBullet 替代。

---

## 四、在 AutoDL 上跑小实验（验证 pipeline）

### 4.1 小规模命令（2 epoch × 100 step）

在项目目录下、虚拟环境已激活时执行。

**若已装 MuJoCo**，先设环境变量再跑：

```bash
cd /root/edp
source venv/bin/activate
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
python -m diffusion.trainer \
  --env 'walker2d-medium-replay-v2' \
  --n_epochs=2 \
  --n_train_step_per_epoch=100 \
  --eval_period=1 \
  --eval_n_trajs=2 \
  --logging.output_dir './experiment_output' \
  --algo_cfg.loss_type=TD3
```

**若未装 MuJoCo**，同上命令即可，程序会自动 fallback 到 PyBullet；或显式用 bullet 环境：

```bash
python -m diffusion.trainer \
  --env 'bullet-walker2d-medium-replay-v0' \
  --n_epochs=2 \
  --n_train_step_per_epoch=100 \
  --eval_period=1 \
  --eval_n_trajs=2 \
  --logging.output_dir './experiment_output' \
  --algo_cfg.loss_type=TD3
```

跑通标志：无 Traceback，终端出现 `agent/*`、`average_return` 等指标，且有两个 epoch 的进度条和表格。

### 4.2 跑整个项目：正式训练（MuJoCo 环境，长跑）

确认小实验无误后，可跑完整项目的正式配置。

**配置说明**：
- **GPU**：1×RTX 4090（或同级）足够，EDP 是单卡 JAX，不需要多 GPU。
- **数据盘**：50G 免费数据盘足够。
- **训练时长**：按 `hps.py` 默认配置，每个 env 通常 **2000 epoch × 1000 step/epoch**，在 4090 上约需 **数小时到一天**（取决于 env 复杂度）。

**运行命令**（MuJoCo 环境）：

```bash
cd /root/edp  # 或 /root/autodl-tmp/edp（若项目在数据盘）
source venv/bin/activate
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210

# 方式 1：直接跑（会输出到终端，关网页会断）
python -m diffusion.trainer \
  --env 'walker2d-medium-replay-v2' \
  --logging.output_dir './experiment_output' \
  --algo_cfg.loss_type=TD3

# 方式 2：后台跑（推荐，关网页不断）
nohup python -m diffusion.trainer \
  --env 'walker2d-medium-replay-v2' \
  --logging.output_dir './experiment_output' \
  --algo_cfg.loss_type=TD3 \
  > train.log 2>&1 &

# 查看后台任务进度
tail -f train.log
```

**其他常用 env 示例**：

```bash
# halfcheetah-medium-v2
python -m diffusion.trainer --env 'halfcheetah-medium-v2' --logging.output_dir './exp_halfcheetah' --algo_cfg.loss_type=TD3

# hopper-medium-v2
python -m diffusion.trainer --env 'hopper-medium-v2' --logging.output_dir './exp_hopper' --algo_cfg.loss_type=TD3

# 用 IQL 算法
python -m diffusion.trainer --env 'walker2d-medium-replay-v2' --logging.output_dir './exp_iql' --algo_cfg.loss_type=IQL --norm_reward=True
```

**训练完成后检查**：
- 终端/日志里看 `average_normalizd_return`、`best_normalized_return` 等指标。
- `experiment_output/` 目录下有 wandb 子目录和可能的 checkpoint（若开了 `save_model`）。

---

## 五、结果与关机关闭实例

- **日志与权重**：在 `--logging.output_dir` 指定目录下（如 `./experiment_output`），会有 wandb 子目录和 ckpt（若开了 `save_model`）。
- **把结果弄回本机**：
  - **方式 A**：JupyterLab 左侧文件树 → 右键 `experiment_output` → Download（打包成 zip）。
  - **方式 B**：用 `scp` 从 AutoDL 拉回本机（需 SSH 权限）。
- **关机**：
  - 在 AutoDL 控制台对该实例执行 **关机**，避免持续扣费。
  - **数据盘内容会保留**（下次开机还在），但**系统盘上的环境（venv）可能保留也可能清空**（以平台说明为准）。
  - 建议：**重要代码和实验结果都放在数据盘**（如 `/root/autodl-tmp/edp`），避免丢失。

---

## 六、常见问题速查

| 现象 | 处理 |
|------|------|
| `No module named 'xxx'` | `pip install xxx`，或检查是否在正确的 venv 里 `source venv/bin/activate`。 |
| `jax.tree_map` / `tree_multimap` 报错 | 说明用的不是我们改过的代码，确保上传的是本机已修复的版本（或按本机修改在 AutoDL 上改同一处）。 |
| `Environment walker2d-medium-replay doesn't exist` | 未装 MuJoCo 时属正常，会 fallback 到 PyBullet；若已装 MuJoCo 仍报错，检查 `LD_LIBRARY_PATH`、`MUJOCO_PY_MUJOCO_PATH` 是否在运行前 export。 |
| CUDA/GPU 相关 warning | AutoDL 若分配了 GPU，装好带 CUDA 的 jax 可加速；CPU 也能跑，只是慢。 |
| JAX 装完仍是 CPU / `jax.devices()` 只有 CPU | 若用方式 A：确保是 CUDA 12 镜像。若用方式 B：该页面只有 jaxlib 0.4.x 的 CUDA wheel（无 0.6.2），需按 **3.3 方式 B** 用 jax 0.4.25 + jaxlib 0.4.25+cuda11.cudnn86 并装 optax&lt;0.2.7。 |
| MuJoCo env 未注册 / `No module named 'mjrl'` | 安装 mjrl：`pip install git+https://github.com/aravindr93/mjrl.git`；并确保已装 MuJoCo 2.1、libosmesa6-dev，且设置了 `LD_LIBRARY_PATH` 与 `MUJOCO_PY_MUJOCO_PATH`。 |
| 断连后任务没了 | 用 `nohup ... &` 或 `screen`/`tmux` 在后台跑，或使用 AutoDL 的「自定义训练」等长任务功能（视平台提供而定）。 |

---

按上述步骤，从「开实例 → 上传本机改过的项目 → 配环境 → 跑小实验」即可在 AutoDL 上从零跑通；之后若要改 env、改 epoch 或加算法，可以继续在同一环境上操作。
