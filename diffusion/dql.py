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

from copy import deepcopy
from functools import partial

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from core.core_api import Algo
from diffusion.diffusion import GaussianDiffusion
from utilities.jax_utils import (
  extend_and_repeat,
  mse_loss,
  next_rng,
  value_and_multi_grad,
)


def update_target_network(main_params, target_params, tau):
  return jax.tree_util.tree_map(
    lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
  )


class DiffusionQL(Algo):
  """
  整个算法的核心类：
  - 包含「扩散策略网络 policy」+「Q 网络 qf1/qf2」+「V 网络 vf（IQL 用）」+「高斯策略 policy_dist」。
  - 执行顺序可以概括为：
    DiffusionQL（类定义） → get_default_config（整理/覆盖超参） → __init__（根据配置初始化各网络和 TrainState）
    → _train_step_*（按 TD3/CRR/IQL 的规则执行一步训练） → train（对外暴露的一步训练接口，被 trainer 调用）。
  - 提供：
    - get_value_loss：Q/V 的 TD 更新（根据 loss_type 有细微差异）。
    - get_diff_loss：扩散 MSE（ELBO→MSE，对应 DDPM 的训练目标）。
    - 三个 _train_step_*：TD3 / CRR / IQL 三种算法下，如何组合 value loss + diff loss + guide loss。
    - train：被 trainer 调用的一步训练接口。
  """

  @staticmethod
  def get_default_config(updates=None):
    """
    返回 DiffusionQL 的默认超参数（一个 ConfigDict），支持用 updates 覆盖：
    - nstep / discount / tau / policy_tgt_freq：离线 RL 的基本配置（n 步 TD、折扣、target soft update 等）。
    - num_timesteps / schedule_name / time_embed_size：扩散模型相关（步数、β 调度、时间嵌入维度）。
    - alpha / use_pred_astart：guide loss 的系数 & 是否启用 action approximation（pred_astart 当动作）。
    - loss_type：选择 TD3 / CRR / IQL 中的哪一种算法。
    - CRR / IQL 相关的 hps：sample_actions、crr_beta、expectile、awr_temperature 等。
    - dpm_steps / dpm_t_end：DPM-Solver 的采样步数和结束时间步。
    """
    cfg = ConfigDict()
    cfg.nstep = 1
    cfg.discount = 0.99
    cfg.tau = 0.005
    cfg.policy_tgt_freq = 5
    cfg.num_timesteps = 100
    cfg.schedule_name = 'linear'
    cfg.time_embed_size = 16
    cfg.alpha = 2.  # NOTE 0.25 in diffusion rl but 2.5 in td3
    cfg.use_pred_astart = True
    cfg.max_q_backup = False
    cfg.max_q_backup_topk = 1
    cfg.max_q_backup_samples = 10

    # learning related
    cfg.lr = 3e-4
    cfg.diff_coef = 1.0
    cfg.guide_coef = 1.0
    cfg.lr_decay = False
    cfg.lr_decay_steps = 1000000
    cfg.max_grad_norm = 0.
    cfg.weight_decay = 0.

    cfg.loss_type = 'TD3'
    cfg.sample_logp = False

    cfg.adv_norm = False
    # CRR-related hps
    cfg.sample_actions = 10
    cfg.crr_ratio_upper_bound = 20
    cfg.crr_beta = 1.0
    cfg.crr_weight_mode = 'mle'
    cfg.fixed_std = True
    cfg.crr_multi_sample_mse = False
    cfg.crr_avg_fn = 'mean'
    cfg.crr_fn = 'exp'

    # IQL-related hps
    cfg.expectile = 0.7
    cfg.awr_temperature = 3.0

    # for dpm-solver
    cfg.dpm_steps = 15
    cfg.dpm_t_end = 0.001

    # useless
    cfg.target_entropy = -1
    if updates is not None:
      cfg.update(ConfigDict(updates).copy_and_resolve_references())
    return cfg

  def __init__(self, cfg, policy, qf, vf, policy_dist):
    """
    构造函数：
    - cfg：从外部传入的配置（会和 get_default_config 合并）。
    - policy：扩散策略网络（DiffusionPolicy）。
    - qf：Q 网络（Critic），这里会构建 qf1/qf2 两个副本。
    - vf：V 网络（IQL 使用）。
    - policy_dist：高斯策略（用于构造近似的 action 分布、log π）。

    主要做的事：
    1. 合并配置 self.config。
    2. 初始化 policy / qf / vf / policy_dist 的参数（init），并放进 TrainState 里，形成 self._train_states。
    3. 拷贝一份 target 参数，形成 self._tgt_params（用于 TD 目标和 soft update）。
    4. 记录 observation_dim / action_dim / max_action 等基础维度信息。
    """
    self.config = self.get_default_config(cfg)
    self.policy = policy
    self.qf = qf
    self.vf = vf
    self.policy_dist = policy_dist
    self.observation_dim = policy.observation_dim
    self.action_dim = policy.action_dim
    self.max_action = policy.max_action
    self.diffusion: GaussianDiffusion = self.policy.diffusion

    self._total_steps = 0
    self._train_states = {}

    policy_params = self.policy.init(
      next_rng(),
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
    )

    def get_lr(lr_decay=False):
      if lr_decay is True:
        return optax.cosine_decay_schedule(
          self.config.lr, decay_steps=self.config.lr_decay_steps
        )
      else:
        return self.config.lr

    def get_optimizer(lr_decay=False, weight_decay=cfg.weight_decay):
      if self.config.max_grad_norm > 0:
        opt = optax.chain(
          optax.clip_by_global_norm(self.config.max_grad_norm),
          optax.adamw(get_lr(lr_decay), weight_decay=weight_decay),
        )
      else:
        opt = optax.adamw(get_lr(), weight_decay=weight_decay)

      return opt

    self._train_states['policy'] = TrainState.create(
      params=policy_params,
      tx=get_optimizer(self.config.lr_decay, weight_decay=0.0),
      apply_fn=None
    )

    policy_dist_params = self.policy_dist.init(
      next_rng(), jnp.zeros((10, self.action_dim))
    )
    self._train_states['policy_dist'] = TrainState.create(
      params=policy_dist_params, tx=get_optimizer(), apply_fn=None
    )

    qf1_params = self.qf.init(
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim)),
    )
    qf2_params = self.qf.init(
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim)),
    )

    vf_params = self.vf.init(next_rng(), jnp.zeros((10, self.observation_dim)))

    self._train_states['qf1'] = TrainState.create(
      params=qf1_params, tx=get_optimizer(), apply_fn=None
    )
    self._train_states['qf2'] = TrainState.create(
      params=qf2_params, tx=get_optimizer(), apply_fn=None
    )
    self._train_states['vf'] = TrainState.create(
      params=vf_params,
      tx=get_optimizer(),
      apply_fn=None,
    )
    self._tgt_params = deepcopy(
      {
        'policy': policy_params,
        'qf1': qf1_params,
        'qf2': qf2_params,
        'vf': vf_params,
      }
    )
    model_keys = ['policy', 'qf1', 'qf2', 'vf', 'policy_dist']

    self._model_keys = tuple(model_keys)

  def get_value_loss(self, batch):
    """
    构造一个「value_loss_fn」，用于计算当前 batch 下 Q 的 TD 损失（不立刻求值，返回函数供 _train_step_* 里求梯度）。
    - 数学：y = r + (1-done)*γ*min(Q1'(s',a'), Q2'(s',a'))，a'=π'(s')；L = E[(Q1(s,a)-y)²] + E[(Q2(s,a)-y)²]。
    - 真正更新参数在 _train_step_* 里对 value_loss_fn 求梯度后 apply_gradients。
    """

    def value_loss_fn(params, tgt_params, rng):
      observations = batch['observations']
      actions = batch['actions']
      rewards = batch['rewards']
      next_observations = batch['next_observations']
      dones = batch['dones']

      # 算 TD target y（对 target 停梯度，不反传到 Q'、π'）
      if self.config.max_q_backup:
        # 可选分支：对 s' 采多组动作、用 Q 取 max/top-k 再双 Q 取 min，得到更保守的 target，无需深究
        samples = self.config.max_q_backup_samples
        next_action = self.policy.apply(
          tgt_params['policy'], rng, next_observations, repeat=samples
        )
        next_action = jnp.clip(next_action, -self.max_action, self.max_action)
        next_obs_repeat = jnp.repeat(
          jnp.expand_dims(next_observations, axis=1), samples, axis=1
        )
        tgt_q1 = self.qf.apply(tgt_params['qf1'], next_obs_repeat, next_action)
        tgt_q2 = self.qf.apply(tgt_params['qf2'], next_obs_repeat, next_action)

        tk = self.config.max_q_backup_topk
        if tk == 1:
          tgt_q = jnp.minimum(tgt_q1.max(axis=-1), tgt_q2.max(axis=-1))
        else:
          batch_idx = jax.vmap(lambda x, i: x[i], 0)
          tgt_q1_max = batch_idx(tgt_q1, jnp.argsort(tgt_q1, axis=-1)[:, -tk])
          tgt_q2_max = batch_idx(tgt_q2, jnp.argsort(tgt_q2, axis=-1)[:, -tk])
          tgt_q = jnp.minimum(tgt_q1_max, tgt_q2_max)
      else:
        # 标准 TD3：a' = π_target(s')，y = r + (1-done)*γ*min(Q1'(s',a'), Q2'(s',a'))
        next_action = self.policy.apply(
          tgt_params['policy'], rng, next_observations
        )
        tgt_q1 = self.qf.apply(
          tgt_params['qf1'], next_observations, next_action
        )
        tgt_q2 = self.qf.apply(
          tgt_params['qf2'], next_observations, next_action
        )
        tgt_q = jnp.minimum(tgt_q1, tgt_q2)
      tgt_q = rewards + (1 - dones) * self.config.discount * tgt_q
      tgt_q = jax.lax.stop_gradient(tgt_q)

      # 当前 Q 对 (s,a) 的估计，拟合同一 target y
      cur_q1 = self.qf.apply(params['qf1'], observations, actions)
      cur_q2 = self.qf.apply(params['qf2'], observations, actions)

      # Value loss = MSE(cur_q1, y) + MSE(cur_q2, y)，两个 Q 都拟合同一 y
      qf1_loss = mse_loss(cur_q1, tgt_q)
      qf2_loss = mse_loss(cur_q2, tgt_q)

      qf_loss = qf1_loss + qf2_loss
      return (qf1_loss, qf2_loss), locals()

    return value_loss_fn

  def get_diff_terms(self, params, observations, actions, dones, rng):
    """
    给定当前参数和 batch 的 (s, a)（a = 数据中的真实动作 x0），算一次扩散步所需中间量：
    随机 t、policy.loss 得到 x_t/model_output/逐样本 MSE、pred_astart、action_dist 与 log π，
    供 get_diff_loss 与 CRR/IQL 的 guide loss 使用。
    """
    # 为 batch 里每个样本随机采一个扩散时间步 t ∈ [0, T)；用 dones.shape 仅为了取 batch 维度 (batch_size,)，与 dones 语义无关
    rng, split_rng = jax.random.split(rng)
    ts = jax.random.randint(
      split_rng, dones.shape, minval=0, maxval=self.diffusion.num_timesteps
    )
    rng, split_rng = jax.random.split(rng)
    # 调用扩散策略的 loss 方法：对真实 a(x0) 加噪得 x_t、网络预测噪声，返回值 terms 为字典，含 x_t、model_output、逐样本 loss 等
    terms = self.policy.apply(
      params["policy"],
      split_rng,
      observations,
      actions,
      ts,
      method=self.policy.loss,
    )
    # pred_astart = 当前步预测的 x0（单步去噪结果），TD3 里当「动作」喂给 Q 算 guide loss
    if self.config.use_pred_astart:
      pred_astart = self.diffusion.p_mean_variance(
        terms["model_output"], terms["x_t"], ts
      )["pred_xstart"]
    else:
      # 不用 action approximation 时需完整前向采样得到动作，慢，一般不用
      rng, split_rng = jax.random.split(rng)
      pred_astart = self.policy.apply(
        params['policy'], split_rng, observations
      )
    terms["pred_astart"] = pred_astart

    # 以下为 CRR/IQL 用：构造可算 log π 的动作分布，供 guide loss；TD3 不依赖，可视为 Day4（CRR/IQL）内容
    action_dist = self.policy_dist.apply(params['policy_dist'], pred_astart)
    sample = pred_astart
    if self.config.sample_logp:
      rng, split_rng = jax.random.split(rng)
      sample = action_dist.sample(seed=split_rng)
    if self.config.fixed_std:
      # 可选：固定方差高斯，无需深究
      action_dist = distrax.MultivariateNormalDiag(
        pred_astart, jnp.ones_like(pred_astart)
      )
    log_prob = action_dist.log_prob(sample)
    terms['sample'] = sample
    terms['action_dist'] = action_dist
    terms['log_p'] = log_prob

    return terms, ts, log_prob

  def get_diff_loss(self, batch):
    """
    构造一个「diff_loss_fn」，不立刻求值；在 _train_step_* 里调用后得到扩散损失和 pred_astart。
    - 内部调 get_diff_terms → policy.loss → diffusion.training_losses；噪声 MSE 的公式在
      diffusion/diffusion.py 的 training_losses 中（x_t = q_sample(x0), target = ε, loss = ‖ε − model_output‖²）。
    - terms["loss"] 即逐样本 MSE，对 batch 求平均后为 diff_loss；与 pred_astart 一并供 policy_loss 使用。
    """

    def diff_loss(params, rng):
      observations = batch['observations']
      actions = batch['actions']
      dones = batch['dones']
      
      terms, ts, _ = self.get_diff_terms(
        params, observations, actions, dones, rng
      )
      # terms["loss"] 来自 nets.DiffusionPolicy.loss → diffusion.GaussianDiffusion.training_losses（噪声 MSE）
      diff_loss = terms["loss"].mean()
      pred_astart = terms["pred_astart"]

      # terms、ts 供 CRR/IQL 或 metrics 用；TD3 只用 diff_loss 和 pred_astart
      return diff_loss, terms, ts, pred_astart

    return diff_loss

  @partial(jax.jit, static_argnames=('self', 'policy_tgt_update'))
  def _train_step(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    if self.config.loss_type not in ['TD3', 'CRR', 'IQL']:
      raise NotImplementedError

    return getattr(self, f"_train_step_{self.config.loss_type.lower()}"
                  )(train_states, tgt_params, rng, batch, policy_tgt_update)

  def _train_step_td3(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    """
    TD3 模式下的一步训练：AC 框架，更新 Critic(Q1/Q2) 与 Actor(扩散策略)。
    流程：构造 value_loss / policy_loss → 分别求梯度 → 更新当前参数 → 软更新 target。

    三个 loss 的公式所在位置（本函数只负责「调用」它们并求梯度）：
    - value loss：在 get_value_loss 返回的 value_loss_fn 内部（tgt_q、cur_q1/2、MSE），见 get_value_loss。
    - diff loss：在 get_diff_loss 返回的 diff_loss_fn 内部，调 get_diff_terms → terms["loss"].mean()，见 get_diff_loss。
    - guide loss：在本函数内 policy_loss_fn 的 fn() 里，-λ * q.mean()，见下方 # Guide loss 注释块。
    """
    # 构造两个 loss 函数（仅定义计算图，不立刻求值）：
    # - value_loss_fn：更新 Critic 用（Q 的 TD 损失，双 Q + target）
    # - diff_loss_fn：在 policy_loss_fn 里用，得到 diff_loss 和 pred_astart，再与 guide_loss 组成 policy_loss
    value_loss_fn = self.get_value_loss(batch)
    diff_loss_fn = self.get_diff_loss(batch)

    def policy_loss_fn(params, tgt_params, rng):
      """
      TD3 下的策略损失（Actor）：
      - diff_loss：扩散拟合数据（噪声 MSE）
      - guide_loss：-λ * mean(Q(s, pred_astart))，用 Q 引导策略往高价值动作走
      - policy_loss = diff_loss + guide_coef * guide_loss
      """
      observations = batch['observations']

      rng, split_rng = jax.random.split(rng)
      # 一次前向得到扩散损失和 pred_astart（单步预测的 x0，当作“动作”用于 Q）
      diff_loss, _, _, pred_astart = diff_loss_fn(params, split_rng)

      # Guide loss：-λ * mean(Q)，mean 对 batch 求平均；λ = alpha / mean(|Q|) 随 Q 尺度自适应，避免梯度爆炸/消失
      def fn(key):
        q = self.qf.apply(params[key], observations, pred_astart)  # shape (batch_size,)，每个样本一个 Q 值
        lmbda = self.config.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())  # λ 自适应：Q 越大 λ 越小
        guide_loss = -lmbda * q.mean()  # 仅 guide 项；完整 policy_loss = diff_loss + guide_coef * guide_loss 在下面
        return lmbda, guide_loss

      lmbda, guide_loss = jax.lax.cond(
        jax.random.uniform(rng) > 0.5, partial(fn, 'qf1'), partial(fn, 'qf2')
      )

      policy_loss = diff_loss + self.config.guide_coef * guide_loss
      return (policy_loss,), locals()

    # 对 value_loss 求梯度：需要分别对 qf1、qf2 求导，故 value_and_multi_grad(..., 2)
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_qf), grads_qf = value_and_multi_grad(
      value_loss_fn, 2, has_aux=True
    )(params, tgt_params, rng)

    # 对 policy_loss 求梯度：只对 policy 求导，故 value_and_multi_grad(..., 1)
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_policy), grads_policy = value_and_multi_grad(
      policy_loss_fn, 1, has_aux=True
    )(params, tgt_params, rng)

    # 用梯度更新当前网络参数（Critic）
    train_states['qf1'] = train_states['qf1'].apply_gradients(
      grads=grads_qf[0]['qf1']
    )
    train_states['qf2'] = train_states['qf2'].apply_gradients(
      grads=grads_qf[1]['qf2']
    )

    # 用梯度更新当前网络参数（Actor = 扩散策略）
    train_states['policy'] = train_states['policy'].apply_gradients(
      grads=grads_policy[0]['policy']
    )

    # 软更新 target 参数：θ_tgt = τ*θ_cur + (1-τ)*θ_tgt，算 TD target 时用
    if policy_tgt_update:
      tgt_params['policy'] = update_target_network(
        train_states['policy'].params, tgt_params['policy'], self.config.tau
      )
    tgt_params['qf1'] = update_target_network(
      train_states['qf1'].params, tgt_params['qf1'], self.config.tau
    )
    tgt_params['qf2'] = update_target_network(
      train_states['qf2'].params, tgt_params['qf2'], self.config.tau
    )

    # 记录本步指标，供 logger / wandb 使用（无需深究各字段含义）
    metrics = dict(
      qf_loss=aux_qf['qf_loss'],
      qf1_loss=aux_qf['qf1_loss'],
      qf2_loss=aux_qf['qf2_loss'],
      cur_q1=aux_qf['cur_q1'].mean(),
      cur_q2=aux_qf['cur_q2'].mean(),
      tgt_q1=aux_qf['tgt_q1'].mean(),
      tgt_q2=aux_qf['tgt_q2'].mean(),
      tgt_q=aux_qf['tgt_q'].mean(),
      policy_loss=aux_policy['policy_loss'],
      guide_loss=aux_policy['guide_loss'],
      diff_loss=aux_policy['diff_loss'],
      lmbda=aux_policy['lmbda'].mean(),
      qf1_grad_norm=optax.global_norm(grads_qf[0]['qf1']),
      qf2_grad_norm=optax.global_norm(grads_qf[1]['qf2']),
      policy_grad_norm=optax.global_norm(grads_policy[0]['policy']),
      qf1_weight_norm=optax.global_norm(train_states['qf1'].params),
      qf2_weight_norm=optax.global_norm(train_states['qf2'].params),
      policy_weight_norm=optax.global_norm(train_states['policy'].params),
    )

    return train_states, tgt_params, metrics

  def _train_step_crr(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    # CRR 一步训练：value_loss 更新双 Q（同 TD3），policy_loss = diff_loss + guide_loss（优势加权 log π）
    value_loss_fn = self.get_value_loss(batch)
    diff_loss_fn = self.get_diff_loss(batch)

    def policy_loss_fn(params, tgt_params, rng):
      """
      CRR 策略损失：diff_loss + guide_loss；guide_loss = -E[λ·log π]，λ 由优势得到。
      """
      observations = batch['observations']
      actions = batch['actions']  # 数据中的真实动作，即 batch 里的 x0

      rng, split_rng = jax.random.split(rng)
      # 一次前向：得到 diff_loss、terms（含 action_dist、mse、ts_weights 等）
      diff_loss, terms, _, _ = diff_loss_fn(params, split_rng)
      action_dist = terms['action_dist']  # 以 pred_astart 为均值的高斯，用于算 log π

      # ---------- 估计 V(s) ≈ E_a~π[min(Q1,Q2)]：对每个 s 采多份动作，用双 Q 取 min 再平均 ----------
      replicated_obs = jnp.broadcast_to(
        observations, (self.config.sample_actions,) + observations.shape
      )
      rng, split_rng = jax.random.split(rng)
      if self.config.use_pred_astart:
        vf_actions = action_dist.sample(
          seed=split_rng, sample_shape=self.config.sample_actions
        )
      else:
        vf_actions = self.policy.apply(
          params['policy'], split_rng, replicated_obs
        )

      # ---------- 优势 A(s,a) = Q(s,a) - V(s) ----------
      cur_q1 = self.qf.apply(params['qf1'], observations, actions)
      cur_q2 = self.qf.apply(params['qf2'], observations, actions)
      v1 = self.qf.apply(params['qf1'], replicated_obs, vf_actions)
      v2 = self.qf.apply(params['qf2'], replicated_obs, vf_actions)
      v = jnp.minimum(v1, v2)  # 双 Q 取 min
      q_pred = jnp.minimum(cur_q1, cur_q2)
      avg_fn = getattr(jnp, self.config.crr_avg_fn)  # 通常为 mean，对 sample_actions 维平均得 V(s)
      adv = q_pred - avg_fn(v, axis=0)

      # ---------- 优势 → 权重 λ：exp(adv/β) 或 heaviside(adv)，并 stop_gradient ----------
      if self.config.crr_fn == 'exp':
        lmbda = jnp.minimum(
          self.config.crr_ratio_upper_bound,
          jnp.exp(adv / self.config.crr_beta)
        )
        if self.config.adv_norm:
          lmbda = jax.nn.softmax(adv / self.config.crr_beta)
      else:
        lmbda = jnp.heaviside(adv, 0)  # adv>0 为 1，否则为 0
      lmbda = jax.lax.stop_gradient(lmbda)

      # ---------- log π 近似：elbo 用 -ts_weights*mse；mle 用 action_dist.log_prob(actions)；其他为采样 MSE ----------
      if self.config.crr_weight_mode == 'elbo':
        log_prob = -terms['ts_weights'] * terms['mse']
      elif self.config.crr_weight_mode == 'mle':
        log_prob = action_dist.log_prob(actions)
      else:
        rng, split_rng = jax.random.split(rng)
        if not self.config.crr_multi_sample_mse:
          sampled_actions = action_dist.sample(seed=split_rng)
          log_prob = -((sampled_actions - actions)**2).mean(axis=-1)
        else:
          sampled_actions = action_dist.sample(
            seed=split_rng, sample_shape=self.config.sample_actions
          )
          log_prob = -(
            (sampled_actions - jnp.expand_dims(actions, axis=0))**2
          ).mean(axis=(0, -1))

      # ---------- guide_loss = -E[λ·log π]，即优势加权行为克隆 ----------
      guide_loss = -jnp.mean(log_prob * lmbda)

      policy_loss = self.config.diff_coef * diff_loss + \
        self.config.guide_coef * guide_loss
      # policy 与 policy_dist 用同一 loss 一起更新（value_and_multi_grad 返回两份梯度）
      losses = {'policy': policy_loss, 'policy_dist': policy_loss}
      return tuple(losses[key] for key in losses.keys()), locals()

    # ---------- 先算 Q 的梯度和策略的梯度（各 2 份：qf1/qf2、policy/policy_dist） ----------
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_qf), grads_qf = value_and_multi_grad(
      value_loss_fn, 2, has_aux=True
    )(params, tgt_params, rng)

    (_, aux_policy), grads_policy = value_and_multi_grad(
      policy_loss_fn, 2, has_aux=True
    )(params, tgt_params, rng)

    # ---------- 用梯度更新当前网络 Q1、Q2、policy、policy_dist(要算std) ----------
    train_states['qf1'] = train_states['qf1'].apply_gradients(
      grads=grads_qf[0]['qf1']
    )
    train_states['qf2'] = train_states['qf2'].apply_gradients(
      grads=grads_qf[1]['qf2']
    )
    train_states['policy'] = train_states['policy'].apply_gradients(
      grads=grads_policy[0]['policy']
    )
    train_states['policy_dist'] = train_states['policy_dist'].apply_gradients(
      grads=grads_policy[1]['policy_dist']
    )

    # ---------- 软更新 target 网络（policy 按 policy_tgt_update 决定是否更新） ----------
    if policy_tgt_update:
      tgt_params['policy'] = update_target_network(
        train_states['policy'].params, tgt_params['policy'], self.config.tau
      )
    tgt_params['qf1'] = update_target_network(
      train_states['qf1'].params, tgt_params['qf1'], self.config.tau
    )
    tgt_params['qf2'] = update_target_network(
      train_states['qf2'].params, tgt_params['qf2'], self.config.tau
    )

    # ---------- 记录指标（无需深究，仅用于日志/监控） ----------
    metrics = dict(
      qf_loss=aux_qf['qf_loss'],
      qf1_loss=aux_qf['qf1_loss'],
      qf2_loss=aux_qf['qf2_loss'],
      cur_q1=aux_qf['cur_q1'].mean(),
      cur_q2=aux_qf['cur_q2'].mean(),
      tgt_q1=aux_qf['tgt_q1'].mean(),
      tgt_q2=aux_qf['tgt_q2'].mean(),
      tgt_q=aux_qf['tgt_q'].mean(),
      policy_loss=aux_policy['policy_loss'],
      guide_loss=aux_policy['guide_loss'],
      diff_loss=aux_policy['diff_loss'],
      lmbda=aux_policy['lmbda'].mean(),
      qf1_grad_norm=optax.global_norm(grads_qf[0]['qf1']),
      qf2_grad_norm=optax.global_norm(grads_qf[1]['qf2']),
      policy_grad_norm=optax.global_norm(grads_policy[0]['policy']),
      qf1_weight_norm=optax.global_norm(train_states['qf1'].params),
      qf2_weight_norm=optax.global_norm(train_states['qf2'].params),
      policy_weight_norm=optax.global_norm(train_states['policy'].params),
    )
    if self.config.loss_type == 'CRR':
      metrics['adv'] = aux_policy['adv'].mean()
      metrics['log_prob'] = aux_policy['log_prob'].mean()

    return train_states, tgt_params, metrics

  def _train_step_iql(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    """
    IQL 单步训练：三阶段更新（顺序固定，实现里先 V、再 policy、再 Q）。
    - value_loss：expectile 回归拟合 V(s)，使 V 逼近「target Q」的 τ-expectile（非 max，防过估计）。
    - critic_loss：用 V(s') 做 bootstrap 更新 Q，TD target = r + (1-done)·γ·V(s')。
    - policy_loss：AWR 权重 ω∝exp((Q-V)/τ) 加权 log π(a|s)，再叠加扩散 diff_loss。
    细节见 IQL 论文与 docs/Day4、EDP_复试梳理。
    """
    diff_loss_fn = self.get_diff_loss(batch)

    def value_loss(train_params):
      """Expectile 回归：对 diff = Q(s,a) - V(s) 按 expectile 加权 MSE，只更新 vf。"""
      observations = batch['observations']
      actions = batch['actions']
      q1 = self.qf.apply(tgt_params['qf1'], observations, actions)
      q2 = self.qf.apply(tgt_params['qf2'], observations, actions)
      q_pred = jax.lax.stop_gradient(jnp.minimum(q1, q2))
      v_pred = self.vf.apply(train_params['vf'], observations)
      diff = q_pred - v_pred
      # expectile 权重：diff>0 用 τ，否则 1-τ，使 V 拟合 Q 的 τ-expectile（非均值）
      expectile_weight = jnp.where(
        diff > 0,
        self.config.expectile,
        1 - self.config.expectile,
      )

      expectile_loss = (expectile_weight * (diff**2)).mean()
      return (expectile_loss,), locals()

    def critic_loss(train_params):
      """Q 网络更新：TD target = r + γ V(s')（用 V 不用 max Q，IQL 特点）。"""
      observations = batch['observations']
      actions = batch['actions']
      next_observations = batch['next_observations']
      rewards = batch['rewards']
      dones = batch['dones']
      next_v = self.vf.apply(train_params['vf'], next_observations)

      discount = self.config.discount**self.config.nstep
      td_target = jax.lax.stop_gradient(
        rewards + (1 - dones) * discount * next_v
      )

      q1_pred = self.qf.apply(train_params['qf1'], observations, actions)
      q2_pred = self.qf.apply(train_params['qf2'], observations, actions)
      qf1_loss = mse_loss(q1_pred, td_target)
      qf2_loss = mse_loss(q2_pred, td_target)

      return (qf1_loss, qf2_loss), locals()

    def policy_loss(params, rng):
      """扩散 loss + AWR guide loss；guide 用 ω·log π(a|s)，ω=exp((Q-V)/τ) 或 softmax 版。"""
      observations = batch['observations']
      actions = batch['actions']
      rng, split_rng = jax.random.split(rng)
      diff_loss, terms, _, pred_astart = diff_loss_fn(params, split_rng)
      v_pred = self.vf.apply(train_params['vf'], observations)
      q1 = self.qf.apply(tgt_params['qf1'], observations, actions)
      q2 = self.qf.apply(tgt_params['qf2'], observations, actions)
      q_pred = jax.lax.stop_gradient(jnp.minimum(q1, q2))
      exp_a = jnp.exp((q_pred - v_pred) * self.config.awr_temperature)
      exp_a = jnp.minimum(exp_a, 100.0)

      if self.config.adv_norm:
        exp_a = jax.nn.softmax((q_pred - v_pred) * self.config.awr_temperature)

      action_dist = terms['action_dist']

      log_probs = action_dist.log_prob(actions)
      awr_loss = -(exp_a * log_probs).mean()
      guide_loss = awr_loss

      policy_loss = self.config.diff_coef * diff_loss + self.config.guide_coef * guide_loss
      # 同一标量 loss 对 policy 与 policy_dist 各求一次梯度，供 value_and_multi_grad(..., 2) 用
      losses = {'policy': policy_loss, 'policy_dist': policy_loss}

      return tuple(losses[key] for key in losses.keys()), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_value), value_grads = value_and_multi_grad(
      value_loss, 1, has_aux=True
    )(
      train_params
    )
    train_states['vf'] = train_states['vf'].apply_gradients(
      grads=value_grads[0]['vf']
    )

    rng, split_rng = jax.random.split(rng)
    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_policy), policy_grads = value_and_multi_grad(
      policy_loss, 2, has_aux=True
    )(train_params, split_rng)
    train_states['policy'] = train_states['policy'].apply_gradients(
      grads=policy_grads[0]['policy']
    )
    train_states['policy_dist'] = train_states['policy_dist'].apply_gradients(
      grads=policy_grads[1]['policy_dist']
    )

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_qf), qf_grads = value_and_multi_grad(
      critic_loss, 2, has_aux=True
    )(
      train_params
    )
    train_states['qf1'] = train_states['qf1'].apply_gradients(
      grads=qf_grads[0]['qf1']
    )
    train_states['qf2'] = train_states['qf2'].apply_gradients(
      grads=qf_grads[1]['qf2']
    )

    # Update target parameters
    if policy_tgt_update:
      tgt_params['policy'] = update_target_network(
        train_states['policy'].params, tgt_params['policy'], self.config.tau
      )
    tgt_params['qf1'] = update_target_network(
      train_states['qf1'].params, tgt_params['qf1'], self.config.tau
    )
    tgt_params['qf2'] = update_target_network(
      train_states['qf2'].params, tgt_params['qf2'], self.config.tau
    )

    metrics = dict(
      vf_loss=aux_value['expectile_loss'].mean(),
      vf_adv=aux_value['diff'].mean(),
      vf_pred=aux_value['v_pred'].mean(),
      next_v=aux_qf['next_v'].mean(),
      qf1_loss=aux_qf['qf1_loss'],
      qf2_loss=aux_qf['qf2_loss'],
      cur_q1=aux_qf['q1_pred'].mean(),
      cur_q2=aux_qf['q2_pred'].mean(),
      tgt_q=aux_qf['td_target'].mean(),
      policy_loss=aux_policy['policy_loss'],
      guide_loss=aux_policy['guide_loss'],
      diff_loss=aux_policy['diff_loss'],
      vf_grad=optax.global_norm(value_grads[0]['vf']),
      qf1_grad_norm=optax.global_norm(qf_grads[0]['qf1']),
      qf2_grad_norm=optax.global_norm(qf_grads[1]['qf2']),
      policy_grad_norm=optax.global_norm(policy_grads[0]['policy']),
      vf_weight_norm=optax.global_norm(train_states['vf'].params),
      qf1_weight_norm=optax.global_norm(train_states['qf1'].params),
      qf2_weight_norm=optax.global_norm(train_states['qf2'].params),
      policy_weight_norm=optax.global_norm(train_states['policy'].params),
    )

    return train_states, tgt_params, metrics

  def train(self, batch):
    self._total_steps += 1
    policy_tgt_update = (
      self._total_steps > 1000 and
      self._total_steps % self.config.policy_tgt_freq == 0
    )
    self._train_states, self._tgt_params, metrics = self._train_step(
      self._train_states, self._tgt_params, next_rng(), batch,
      policy_tgt_update
    )
    return metrics

  @property
  def model_keys(self):
    return self._model_keys

  @property
  def train_states(self):
    return self._train_states

  @property
  def train_params(self):
    return {key: self.train_states[key].params for key in self.model_keys}

  @property
  def total_steps(self):
    return self._total_steps
