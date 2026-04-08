# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

# MuJoCo + d4rl.locomotion：仅在使用 dataset=d4rl 的 Walker 等 D4RL 基准时需要。
# 毕设 ETF 路线（--dataset=etf）可不安装 mujoco_py / MuJoCo；导入失败时忽略即可。
def _try_import_mujoco_for_d4rl_benchmarks():
  try:
    if "MUJOCO_PY_MUJOCO_PATH" not in os.environ:
      os.environ["MUJOCO_PY_MUJOCO_PATH"] = os.path.expanduser("~/.mujoco/mujoco210")
    if "LD_LIBRARY_PATH" not in os.environ or ".mujoco" not in os.environ.get("LD_LIBRARY_PATH", ""):
      mujoco_bin = os.path.expanduser("~/.mujoco/mujoco210/bin")
      os.environ["LD_LIBRARY_PATH"] = mujoco_bin + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    import mujoco_py  # noqa: E402
    try:
      import d4rl.locomotion  # noqa: E402
    except Exception as e:
      sys.stderr.write("d4rl.locomotion failed (MuJoCo locomotion envs unavailable): %s\n" % e)
  except Exception as e:
    sys.stderr.write(
        "mujoco_py not available (OK for ETF-only; install for D4RL MuJoCo tasks): %s\n" % e
    )


_try_import_mujoco_for_d4rl_benchmarks()
from collections import deque
from functools import partial
from pathlib import Path

import absl
import absl.flags
import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from data import Dataset, DM2Gym, RandSampler, RLUPDataset
from diffusion.constants import (
  DATASET,
  DATASET_ABBR_MAP,
  DATASET_MAP,
  ENV,
  ENV_MAP,
  ENVNAME_MAP,
)
from diffusion.diffusion import (
  GaussianDiffusion,
  LossType,
  ModelMeanType,
  ModelVarType,
)
from diffusion.dql import DiffusionQL
from diffusion.hps import hyperparameters
from diffusion.nets import Critic, DiffusionPolicy, GaussianPolicy, Value
from utilities.jax_utils import batch_to_jax, next_rng
from utilities.replay_buffer import get_d4rl_dataset, get_etf_dataset
from utilities.sampler import TrajSampler
from utilities.utils import (
  Timer,
  WandBLogger,
  define_flags_with_default,
  get_user_flags,
  norm_obs,
  prefix_metrics,
  set_random_seed,
)
from viskit.logging import logger, setup_logger

FLAGS_DEF = define_flags_with_default(
  algo="DiffQL",
  # algo="DiffusionQL",
  type="model-free",
  env="walker2d-medium-replay-v2",
  dataset='d4rl',
  rl_unplugged_task_class='control_suite',
  max_traj_length=1000,
  save_model=False,
  seed=42,
  batch_size=256,
  reward_scale=1,
  reward_bias=0,
  clip_action=0.999,
  encoder_arch="64-64",
  policy_arch="256-256-256",
  qf_arch="256-256-256",
  orthogonal_init=False,
  policy_log_std_multiplier=1.0,
  policy_log_std_offset=-1.0,
  algo_cfg=DiffusionQL.get_default_config(),
  n_epochs=2000,
  n_train_step_per_epoch=1000,
  eval_period=10,
  eval_n_trajs=10,
  logging=WandBLogger.get_default_config(),
  qf_layer_norm=False,
  policy_layer_norm=False,
  activation="mish",
  obs_norm=False,
  act_method='',
  sample_method='ddpm',
  policy_temp=1.0,
  norm_reward=False,
  etf_train_csv="",
  etf_eval_csv="",
  portfolio_train_csv="",
  portfolio_eval_csv="",
  behavior_seed=42,
)

# 采样器策略：
# 给定当前观测 obs（以及 agent 的参数），负责在「评估 / 与环境交互」阶段输出动作。
# 可以：
#   - 直接用策略采 1 个动作（ddpm_act / dpm_act / ddim_act）；
#   - 也可以用 EAS 方式：先采 num_samples 个候选动作，再用 Q 网络选一个（*_ensemble_act）。
class SamplerPolicy(object):

  def __init__(  # 构造函数：只存下需要用到的模块与配置
    self, policy, qf=None, mean=0, std=1, ensemble=False, act_method='ddpm'
  ):
    self.policy = policy      # 扩散策略网络（DiffusionPolicy），负责“生成动作”
    self.qf = qf              # Q 网络（Critic），在 ensemble/EAS 下用来给候选动作打分
    self.mean = mean          # 观测归一化用的均值
    self.std = std            # 观测归一化用的标准差
    self.num_samples = 50     # ensemble/EAS 时，从策略采样的候选动作个数 N
    self.act_method = act_method  # 当前使用的采样方式：'ddpm' / 'dpm' / 'ddim' / 'ddpmensemble' / 'dpmensemble' / 'ensemble' 等

  # 更新内部保存的 params（来自 agent.train_params），评估前调用一次即可。
  def update_params(self, params):
    self.params = params
    return self

  # 最基础的 act：直接调用 policy.apply，一般用不到（评估时会用下方的 *_act 接口）。
  @partial(jax.jit, static_argnames=("self", "deterministic"))
  def act(self, params, rng, observations, deterministic):
    return self.policy.apply(
      params["policy"], rng, observations, deterministic, repeat=None
    )

  # ensemble_act：
  #   - 不指定 DDPM/DPM 方式，由策略自身的 sample_method 决定；
  #   - 采 num_samples 个候选动作；
  #   - 用双 Q 的 min(Q1,Q2) 给每个动作打分；
  #   - 按 Q 作为 logits 做一次 categorical 抽样（= 一种 EAS 实现）。
  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def ensemble_act(
    self, params, rng, observations, deterministic, num_samples
  ):
    rng, key = jax.random.split(rng)
    actions = self.policy.apply(
      params["policy"], key, observations, deterministic, repeat=num_samples
    )
    q1 = self.qf.apply(params['qf1'], observations, actions)
    q2 = self.qf.apply(params['qf2'], observations, actions)
    q = jnp.minimum(q1, q2)

    idx = jax.random.categorical(rng, q)
    return jnp.take(actions, idx, axis=-2)

  # 执行动作（ddpmensemble）
  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def ddpmensemble_act(
    self, params, rng, observations, deterministic, num_samples
  ):
    rng, key = jax.random.split(rng)
    actions = self.policy.apply(
      params["policy"],
      rng,
      observations,
      deterministic,
      repeat=num_samples,
      method=self.policy.ddpm_sample,
    )
    q1 = self.qf.apply(params['qf1'], observations, actions)
    q2 = self.qf.apply(params['qf2'], observations, actions)
    q = jnp.minimum(q1, q2)

    idx = jax.random.categorical(rng, q)
    return jnp.take(actions, idx, axis=-2)

  # dpmensemble_act：与 ddpmensemble_act 类似，只是内部用 DPM-Solver 少步采样。
  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def dpmensemble_act(
    self, params, rng, observations, deterministic, num_samples
  ):
    rng, key = jax.random.split(rng)
    actions = self.policy.apply(
      params["policy"],
      rng,
      observations,
      deterministic,
      repeat=num_samples,
      method=self.policy.dpm_sample,
    )
    q1 = self.qf.apply(params['qf1'], observations, actions)
    q2 = self.qf.apply(params['qf2'], observations, actions)
    q = jnp.minimum(q1, q2)

    idx = jax.random.categorical(rng, q)
    return jnp.take(actions, idx, axis=-2)

  # dpm_act：只采 1 个动作（不 ensemble），用 DPM-Solver 少步采样。
  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def dpm_act(self, params, rng, observations, deterministic, num_samples):
    return self.policy.apply(
      params["policy"],
      rng,
      observations,
      deterministic,
      method=self.policy.dpm_sample,
    )

  # ddim_act：只采 1 个动作，用 DDIM 采样。
  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def ddim_act(self, params, rng, observations, deterministic, num_samples):
    return self.policy.apply(
      params["policy"],
      rng,
      observations,
      deterministic,
      method=self.policy.ddim_sample,
    )

  # ddpm_act：只采 1 个动作，用 DDPM 采样。
  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def ddpm_act(self, params, rng, observations, deterministic, num_samples):
    return self.policy.apply(
      params["policy"],
      rng,
      observations,
      deterministic,
      method=self.policy.ddpm_sample,
    )

  def __call__(self, observations, deterministic=False):
    # 对单个观测做归一化，然后调用上面对应的 *_act 方法：
    #   - self.act_method = 'ddpm'      → 调 ddpm_act（DDPM 单采样）
    #   - self.act_method = 'dpm'       → 调 dpm_act（DPM-Solver 单采样）
    #   - self.act_method = 'ensemble'  → 调 ensemble_act（多采样 + Q 选）
    #   - self.act_method = 'ddpmensemble' / 'dpmensemble' → 对应的 EAS 版本
    observations = (observations - self.mean) / self.std
    actions = getattr(self, f"{self.act_method}_act")(
      self.params, next_rng(), observations, deterministic, self.num_samples
    )
    if isinstance(actions, tuple):
      actions = actions[0]
    assert jnp.all(jnp.isfinite(actions))
    return jax.device_get(actions)


class DiffusionTrainer: # 是什么：（一个“训练器”类，负责把“数据、策略、Q/V、算法、评估”全部组装起来，并跑训练循环。）

  # 只做“读配置、选激活函数、记 env 类型”，为后面的 _setup_* 做准备。
  def __init__(self): # 训练器的构造函数
    self._cfgs = absl.flags.FLAGS
    self._algo = DiffusionQL
    self._algo_type = 'DiffusionQL'
    
    # 按当前 env 从表里取梯度裁剪的系数，写回配置（ETF 等新 env 可不在表中，用默认值）。
    try:
      self._cfgs.algo_cfg.max_grad_norm = hyperparameters[self._cfgs.env]['gn']
    except KeyError:
      self._cfgs.algo_cfg.max_grad_norm = 4.0
    # 学习率衰减的总步数 = 总训练步数。
    self._cfgs.algo_cfg.lr_decay_steps = \
      self._cfgs.n_epochs * self._cfgs.n_train_step_per_epoch

    if self._cfgs.activation == 'mish':
      act_fn = lambda x: x * jnp.tanh(jax.nn.softplus(x))
    else:
      act_fn = getattr(jax.nn, self._cfgs.activation)

    # 把选好的激活函数存起来，后面建 Q、V、Policy 时都用它。
    self._act_fn = act_fn

    # 把“用户改过的所有参数”整理成一个字典，方便打日志、复现实验。
    self._variant = get_user_flags(self._cfgs, FLAGS_DEF)
    for k, v in self._cfgs.algo_cfg.items():
      self._variant[f"algo.{k}"] = v

    # get high level env
    env_name_full = self._cfgs.env
    for scenario_name in ENV_MAP:
      if scenario_name in env_name_full:
        self._env = ENV_MAP[scenario_name]
        break
    else:
      raise NotImplementedError

  def train(self):
    # 一次性把 logger、dataset、policy、qf、vf、agent、sampler_policy 都建好（内部会调下面那些 _setup_*）。
    # 即一次性把数据和网络建好，之后每个 epoch 直接训练、评估，不用再建东西。
    self._setup()

    # 初始化评估指标
    act_methods = self._cfgs.act_method.split('-')
    viskit_metrics = {} # 记录评估指标
    recent_returns = {method: deque(maxlen=10) for method in act_methods} # 记录最近10次评估得分
    best_returns = {method: -float('inf') for method in act_methods} # 记录每个采样方法的最佳评估得分
    # 训练循环
    for epoch in range(self._cfgs.n_epochs): # 每个 epoch 内循环 n_train_step_per_epoch 次，每次都用一个 batch 训练。
      metrics = {"epoch": epoch}

      with Timer() as train_timer: # 训练时间计时器
        for _ in tqdm.tqdm(range(self._cfgs.n_train_step_per_epoch)):
          batch = batch_to_jax(self._dataset.sample()) # 从 dataset 里取一个 batch 训练。
          metrics.update(prefix_metrics(self._agent.train(batch), "agent")) # 把训练结果更新到 metrics 里。

      with Timer() as eval_timer: # 评估时间计时器
        if epoch == 0 or (epoch + 1) % self._cfgs.eval_period == 0:

          # 评估循环
          for method in act_methods:
            # TODO: merge these two
            # 设置采样方法
            # 如果 act_method 不为空，则使用 act_method，否则使用 sample_method + "ensemble"
            self._sampler_policy.act_method = \
              method or self._cfgs.sample_method + "ensemble"
            # 如果 sample_method 为 ddim，则使用 ensemble 采样方法
            if self._cfgs.sample_method == 'ddim':
              self._sampler_policy.act_method = "ensemble"
            # 评估采样
            # 使用 eval_sampler 采样
            trajs = self._eval_sampler.sample(
              self._sampler_policy.update_params(self._agent.train_params),
              self._cfgs.eval_n_trajs,
              deterministic=True,
              obs_statistics=(self._obs_mean, self._obs_std, self._obs_clip),
            )
            # 计算评估指标（没必要细看后面）
            # 如果 act_methods 长度为 1，则 post 为空，否则为 "_" + method
            post = "" if len(act_methods) == 1 else "_" + method
            # 计算平均返回值
            metrics["average_return" +
                    post] = np.mean([np.sum(t["rewards"]) for t in trajs])
            # 计算平均轨迹长度
            metrics["average_traj_length" +
                    post] = np.mean([len(t["rewards"]) for t in trajs])
            # 计算平均归一化返回值
            metrics["average_normalizd_return" + post] = cur_return = np.mean(
              [
                self._eval_sampler.env.get_normalized_score(
                  np.sum(t["rewards"])
                ) for t in trajs
              ]
            )
            # 记录最近10次归一化返回值
            recent_returns[method].append(cur_return)
            # 计算平均10次归一化返回值
            metrics["average_10_normalized_return" +
                    post] = np.mean(recent_returns[method])
            # 计算最佳归一化返回值
            metrics["best_normalized_return" +
                    post] = best_returns[method] = max(
                      best_returns[method], cur_return
                    )
            # 计算平均完成率
            metrics["done" +
                    post] = np.mean([np.sum(t["dones"]) for t in trajs])

          # 保存模型
          if self._cfgs.save_model:
            save_data = {
              "agent": self._agent,
              "variant": self._variant,
              "epoch": epoch
            }
            self._wandb_logger.save_pickle(save_data, f"model_{epoch}.pkl")

      # 更新评估指标
      metrics["train_time"] = train_timer()
      metrics["eval_time"] = eval_timer()
      # 计算总时间
      metrics["epoch_time"] = train_timer() + eval_timer()
      # 记录评估指标
      self._wandb_logger.log(metrics)
      # 更新记录评估指标
      viskit_metrics.update(metrics)
      # 记录评估指标
      logger.record_dict(viskit_metrics)
      # 记录评估指标
      logger.dump_tabular(with_prefix=False, with_timestamp=False)

    # save model
    if self._cfgs.save_model:
      save_data = {
        "agent": self._agent,
        "variant": self._variant,
        "epoch": epoch
      }
      self._wandb_logger.save_pickle(save_data, "model_final.pkl")

  def _setup(self):

    set_random_seed(self._cfgs.seed)
    # setup logger
    self._wandb_logger = self._setup_logger()

    # setup dataset and eval_sample
    self._dataset, self._eval_sampler = self._setup_dataset()

    # setup policy
    self._policy = self._setup_policy()
    self._policy_dist = GaussianPolicy(
      self._action_dim, temperature=self._cfgs.policy_temp
    )

    # setup Q-function
    self._qf = self._setup_qf()
    self._vf = self._setup_vf()

    # setup agent
    self._agent = self._algo(
      self._cfgs.algo_cfg, self._policy, self._qf, self._vf, self._policy_dist
    )

    # setup sampler policy
    self._sampler_policy = SamplerPolicy(self._agent.policy, self._agent.qf)

  def _setup_qf(self):
    qf = Critic(
      self._observation_dim,
      self._action_dim,
      to_arch(self._cfgs.qf_arch),
      use_layer_norm=self._cfgs.qf_layer_norm,
      act=self._act_fn,
      orthogonal_init=self._cfgs.orthogonal_init,
    )
    return qf

  def _setup_vf(self):
    vf = Value(
      self._observation_dim,
      to_arch(self._cfgs.qf_arch),
      use_layer_norm=self._cfgs.qf_layer_norm,
      act=self._act_fn,
      orthogonal_init=self._cfgs.orthogonal_init,
    )
    return vf

  def _setup_policy(self):
    gd = GaussianDiffusion(
      num_timesteps=self._cfgs.algo_cfg.num_timesteps,
      schedule_name=self._cfgs.algo_cfg.schedule_name,
      model_mean_type=ModelMeanType.EPSILON,
      model_var_type=ModelVarType.FIXED_SMALL,
      loss_type=LossType.MSE,
      min_value=-self._max_action,
      max_value=self._max_action,
    )
    policy = DiffusionPolicy(
      diffusion=gd,
      observation_dim=self._observation_dim,
      action_dim=self._action_dim,
      arch=to_arch(self._cfgs.policy_arch),
      time_embed_size=self._cfgs.algo_cfg.time_embed_size,
      use_layer_norm=self._cfgs.policy_layer_norm,
      sample_method=self._cfgs.sample_method,
      dpm_steps=self._cfgs.algo_cfg.dpm_steps,
      dpm_t_end=self._cfgs.algo_cfg.dpm_t_end,
    )

    return policy

  def _setup_logger(self):
    env_name_high = ENVNAME_MAP[self._env]
    env_name_full = self._cfgs.env
    dataset_name_abbr = DATASET_ABBR_MAP[self._cfgs.dataset]

    logging_configs = self._cfgs.logging
    logging_configs["project"] = f"{self._cfgs.algo}-{env_name_high}-" + \
      f"{dataset_name_abbr}-{self._cfgs.algo_cfg.loss_type}"
    wandb_logger = WandBLogger(
      config=logging_configs, variant=self._variant, env_name=env_name_full
    )
    setup_logger(
      variant=self._variant,
      exp_id=wandb_logger.experiment_id,
      seed=self._cfgs.seed,
      base_log_dir=self._cfgs.logging.output_dir,
      include_exp_prefix_sub_dir=False,
    )

    return wandb_logger
    
  # 设置评估采样器
  def _setup_d4rl(self):
    env_id = self._cfgs.env
    # 设置环境
    try:
      env = gym.make(env_id)
    except gym.error.NameNotFound:
      import sys
      sys.stderr.write(
        "MuJoCo env %r not registered. Ensure: (1) mujoco_py and MuJoCo 2.1 at ~/.mujoco/mujoco210, "
        "(2) pip install mjrl (e.g. pip install git+https://github.com/aravindr93/mjrl.git), "
        "(3) LD_LIBRARY_PATH and MUJOCO_PY_MUJOCO_PATH are set.\n" % (env_id,)
      )
      raise
    eval_sampler = TrajSampler(env, self._cfgs.max_traj_length)

    # 设置归一化奖励
    norm_reward = self._cfgs.norm_reward
    if 'antmaze' in self._cfgs.env: # 如果环境为 Antmaze，则不归一化奖励
      norm_reward = False
    # 获取 D4RL 数据集

    dataset = get_d4rl_dataset(
      eval_sampler.env, # 环境
      self._cfgs.algo_cfg.nstep, # 步数
      self._cfgs.algo_cfg.discount, # 折扣因子
      norm_reward=norm_reward, # 归一化奖励
    )
    # 设置奖励
    dataset["rewards"] = dataset[
      "rewards"] * self._cfgs.reward_scale + self._cfgs.reward_bias
    dataset["actions"] = np.clip(
      dataset["actions"], -self._cfgs.clip_action, self._cfgs.clip_action
    )

    if self._env == ENV.Kitchen or self._env == ENV.Adroit or self._env == ENV.Antmaze:
      if self._cfgs.obs_norm:
        self._obs_mean = dataset["observations"].mean()
        self._obs_std = dataset["observations"].std()
        self._obs_clip = 10
      norm_obs(dataset, self._obs_mean, self._obs_std, self._obs_clip)

      if self._env == ENV.Antmaze:
        if self._cfgs.algo_cfg.loss_type == 'IQL':
          dataset["rewards"] -= 1
        else:
          dataset["rewards"] = (dataset["rewards"] - 0.5) * 4
      else:
        min_r, max_r = np.min(dataset["rewards"]), np.max(dataset["rewards"])
        dataset["rewards"] = (dataset["rewards"] - min_r) / (max_r - min_r)
        dataset["rewards"] = (dataset["rewards"] - 0.5) * 2

    # set sampler
    dataset = Dataset(dataset)
    sampler = RandSampler(dataset.size(), self._cfgs.batch_size)
    dataset.set_sampler(sampler)

    return dataset, eval_sampler

  def _setup_etf(self):
    """离线数据来自 data_thesis CSV + 行为策略；评估用 etf_eval_csv（默认 val）。"""
    from trading_env.gym_env import ETFGymEnv

    repo_root = Path(__file__).resolve().parent.parent
    train_csv = self._cfgs.etf_train_csv
    if not train_csv:
      train_csv = str(repo_root / "data_thesis" / "510300_hs300_etf_train.csv")
    eval_csv = self._cfgs.etf_eval_csv
    if not eval_csv:
      eval_csv = str(repo_root / "data_thesis" / "510300_hs300_etf_val.csv")

    eval_env = ETFGymEnv(eval_csv)
    eval_sampler = TrajSampler(eval_env, self._cfgs.max_traj_length)

    norm_reward = self._cfgs.norm_reward
    dataset = get_etf_dataset(
        train_csv,
        self._cfgs.algo_cfg.nstep,
        self._cfgs.algo_cfg.discount,
        norm_reward=norm_reward,
        behavior_seed=self._cfgs.behavior_seed,
    )
    dataset["rewards"] = (
        dataset["rewards"] * self._cfgs.reward_scale + self._cfgs.reward_bias
    )
    dataset["actions"] = np.clip(
        dataset["actions"], -self._cfgs.clip_action, self._cfgs.clip_action
    )

    dataset = Dataset(dataset)
    sampler = RandSampler(dataset.size(), self._cfgs.batch_size)
    dataset.set_sampler(sampler)

    return dataset, eval_sampler

  def _setup_portfolio(self):
    """离线数据来自 portfolio split CSV；评估用 portfolio_eval_csv（默认 val）。"""
    from trading_env.gym_env import PortfolioGymEnv
    from utilities.replay_buffer import get_portfolio_dataset

    repo_root = Path(__file__).resolve().parent.parent
    train_csv = self._cfgs.portfolio_train_csv
    if not train_csv:
      train_csv = str(repo_root / "data_thesis" / "hs300_top20_offline_train.csv")
    eval_csv = self._cfgs.portfolio_eval_csv
    if not eval_csv:
      eval_csv = str(repo_root / "data_thesis" / "hs300_top20_offline_val.csv")

    eval_env = PortfolioGymEnv(eval_csv)
    eval_sampler = TrajSampler(eval_env, self._cfgs.max_traj_length)

    norm_reward = self._cfgs.norm_reward
    dataset = get_portfolio_dataset(
        train_csv,
        self._cfgs.algo_cfg.nstep,
        self._cfgs.algo_cfg.discount,
        norm_reward=norm_reward,
        behavior_seed=self._cfgs.behavior_seed,
    )
    dataset["rewards"] = (
        dataset["rewards"] * self._cfgs.reward_scale + self._cfgs.reward_bias
    )
    dataset["actions"] = np.clip(
        dataset["actions"], -self._cfgs.clip_action, self._cfgs.clip_action
    )

    dataset = Dataset(dataset)
    sampler = RandSampler(dataset.size(), self._cfgs.batch_size)
    dataset.set_sampler(sampler)

    return dataset, eval_sampler

  def _setup_rlup(self):
    path = Path(__file__).absolute().parent.parent / 'data'
    dataset = RLUPDataset(
      self._cfgs.rl_unplugged_task_class,
      self._cfgs.env,
      str(path),
      batch_size=self._cfgs.batch_size,
      action_clipping=self._cfgs.clip_action,
    )

    env = DM2Gym(dataset.env)
    eval_sampler = TrajSampler(env, max_traj_length=self._cfgs.max_traj_length)

    return dataset, eval_sampler

  # 设置数据集
  def _setup_dataset(self):
    # 设置观测均值、标准差、截断值
    self._obs_mean = 0
    self._obs_std = 1
    self._obs_clip = np.inf
    # 设置数据集类型
    dataset_type = DATASET_MAP[self._cfgs.dataset]
    # 设置数据集
    if dataset_type == DATASET.D4RL:
      dataset, eval_sampler = self._setup_d4rl() # 设置 D4RL 数据集
    elif dataset_type == DATASET.RLUP:
      dataset, eval_sampler = self._setup_rlup() # 设置 RLUP 数据集
    elif dataset_type == DATASET.ETF:
      dataset, eval_sampler = self._setup_etf()
    elif dataset_type == DATASET.PORTFOLIO:
      dataset, eval_sampler = self._setup_portfolio()
    else:
      raise NotImplementedError
    # 设置观测维度、动作维度、最大动作
    self._observation_dim = eval_sampler.env.observation_space.shape[0]
    self._action_dim = eval_sampler.env.action_space.shape[0]
    self._max_action = float(eval_sampler.env.action_space.high[0])

    # 设置目标熵
    if self._cfgs.algo_cfg.target_entropy >= 0.0:
      action_space = eval_sampler.env.action_space
      self._cfgs.algo_cfg.target_entropy = -np.prod(action_space.shape).item()
    # 返回数据集和评估采样器
    return dataset, eval_sampler


def to_arch(string):
  return tuple(int(x) for x in string.split('-'))


if __name__ == '__main__':

  def main(argv):
    trainer = DiffusionTrainer() # 建训练器（这时会执行 __init__，只做配置和成员变量，还没建数据、网络）
    trainer.train() # 开始训练（这里才会 _setup、循环 epoch、评估等）
    os._exit(os.EX_OK)

  absl.app.run(main)
