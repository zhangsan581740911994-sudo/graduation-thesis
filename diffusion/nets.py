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

"""Networks for diffusion policy."""
from functools import partial
from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from diffusion.diffusion import (
  GaussianDiffusion,
  ModelMeanType,
  _extract_into_tensor,
)
from diffusion.dpm_solver import DPM_Solver, NoiseScheduleVP
from utilities.jax_utils import extend_and_repeat


def multiple_action_q_function(forward):

  def wrapped(self, observations, actions, **kwargs):
    multiple_actions = False
    batch_size = observations.shape[0]
    if actions.ndim == 3 and observations.ndim == 2:
      multiple_actions = True
      observations = extend_and_repeat(observations, 1, actions.shape[1])
      observations = observations.reshape(-1, observations.shape[-1])
      actions = actions.reshape(-1, actions.shape[-1])
    q_values = forward(self, observations, actions, **kwargs)
    if multiple_actions:
      q_values = q_values.reshape(batch_size, -1)
    return q_values

  return wrapped


def mish(x):
  return x * jnp.tanh(nn.softplus(x))


def sinusoidal_embedding(timesteps, dim, max_period=10000):
  """
  正弦位置编码：把标量时间步 t 变成 dim 维向量，无参数、连续、不同 t 得到不同向量。
  :param timesteps: [N]，每个样本一个时间步（可整数可小数）。
  :param dim: 输出维度（一半用 cos，一半用 sin，所以 half = dim//2）。
  :param max_period: 控制最低频率，和 Transformer 里 10000 一样。
  :return: [N, dim]，用作时间嵌入的「原材料」，后面 TimeEmbedding 的 MLP 会再加工。
  """
  half = dim // 2
  # 构造 half 个频率：从低频到高频。freqs[i] = max_period^(-i/half)，i=0..half-1
  # 这样不同维度对应不同「尺度」，有的对 t 变化敏感（高频），有的平滑（低频）
  freqs = jnp.exp(
    -jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
  )
  # args[n, i] = timesteps[n] * freqs[i]，形状 [N, half]
  args = jnp.expand_dims(timesteps, axis=-1) * freqs[None, :]
  # 一半维度 cos(args)，一半 sin(args)，拼成 [N, dim]。sin+cos 一起才能表达相位信息
  embd = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
  return embd


class TimeEmbedding(nn.Module):
  """
  时间步嵌入：把扩散时间步 t 编码成固定维度的向量，供策略网络使用。
  扩散模型需要知道「当前是第几步去噪」，不同 t 对应不同的噪声水平，网络行为应不同。
  """
  embed_size: int  # 输出嵌入维度（如 16）
  act: callable = mish  # 激活函数

  @nn.compact
  def __call__(self, timesteps):
    # 第一步：正弦位置编码，把标量 t 变成 [N, embed_size] 的向量（无参数）
    x = sinusoidal_embedding(timesteps, self.embed_size)
    # 第二步：两层 MLP 把正弦编码映射成「可学习的」时间表示
    x = nn.Dense(self.embed_size * 2)(x)  # 先扩到 2 倍维度
    x = self.act(x)
    x = nn.Dense(self.embed_size)(x)      # 再压回 embed_size
    return x  # 形状 [N, embed_size]，在 PolicyNet 里与 state、action 拼在一起


class PolicyNet(nn.Module):
  """
  扩散策略的「噪声预测网络」：输入 (状态, 当前步动作, 时间步)，输出预测的噪声（或 x0，由 diffusion 侧决定）。
  训练时：state=obs, action=x_t（加噪后的动作）, t=时间步 → 输出 ε_pred，和真实噪声算 MSE。
  """
  output_dim: int  # 等于 action_dim，输出维度与动作一致（预测噪声时就是动作维度）
  arch: Tuple = (256, 256, 256)  # 隐藏层宽度，三层 256（无需了解：记住默认即可）
  time_embed_size: int = 16
  act: callable = mish  # 激活函数（无需了解：知道是激活即可）
  use_layer_norm: bool = False  # 是否在隐藏层后加 LayerNorm（无需了解：默认 False）

  @nn.compact
  def __call__(self, state, action, t):
    # 时间步 t 编码成向量，与 state、action 拼在一起，这样网络同时知道「当前状态、当前加噪动作、第几步」
    time_embed = TimeEmbedding(self.time_embed_size, self.act)(t)
    x = jnp.concatenate([state, action, time_embed], axis=-1)

    # 若干层全连接：Dense → 可选 LayerNorm → 激活。arch=(256,256,256) 即三层 256 维
    for feat in self.arch:
      x = nn.Dense(feat)(x)
      if self.use_layer_norm:
        x = nn.LayerNorm()(x)
      x = self.act(x)

    # 最后一层映射到 output_dim（动作维度），即预测的噪声 ε
    x = nn.Dense(self.output_dim)(x)
    return x


class DiffusionPolicy(nn.Module):
  """
  扩散策略的完整封装：包含「噪声预测网络」+「采样接口」+「训练接口」。
  
  与 diffusion.py 的区别：
  - DiffusionPolicy（这里）：策略的完整封装，包含网络（PolicyNet）和采样/训练接口
  - GaussianDiffusion（diffusion.py）：扩散模型的数学工具类，只提供数学公式（q_sample、p_mean_variance、training_losses等），不包含网络
  
  关系：DiffusionPolicy 内部有一个 GaussianDiffusion 实例，用它来调用数学工具。
  """
  diffusion: GaussianDiffusion  # 数学工具类，提供扩散公式
  observation_dim: int
  action_dim: int
  arch: Tuple = (256, 256, 256)  # 无需了解：网络架构参数
  time_embed_size: int = 16
  act: callable = mish  # 无需了解：激活函数
  use_layer_norm: bool = False  # 无需了解：是否用 LayerNorm
  use_dpm: bool = False  # 无需了解：是否用 DPM（已废弃，用 sample_method 控制）
  sample_method: str = "ddpm"  # 采样方式：ddpm/ddim/dpm
  dpm_steps: int = 15  # 无需了解：DPM 采样步数（默认即可）
  dpm_t_end: float = 0.001  # 无需了解：DPM 结束时间步（默认即可）

  def setup(self):
    """初始化时创建噪声预测网络（PolicyNet）"""
    self.base_net = PolicyNet(
      output_dim=self.action_dim,
      arch=self.arch,
      time_embed_size=self.time_embed_size,
      act=self.act,
      use_layer_norm=self.use_layer_norm,
    )

  def __call__(self, rng, observations, deterministic=False, repeat=None):
    """
    生成动作：根据 sample_method 选择采样方式（ddpm/ddim/dpm）。
    训练时不用这个，训练用 loss()；评估/推理时用这个。
    """
    return getattr(self, f"{self.sample_method}_sample"
                  )(rng, observations, deterministic, repeat)

  def ddpm_sample(self, rng, observations, deterministic=False, repeat=None):
    """
    DDPM 采样：完整 T 步去噪，从噪声得到动作。
    
    关键理解：为什么 observations 要固定？
    - 生成时：给定一个状态 obs，要生成一个动作 a
    - 在整个去噪循环（T 步）中，obs 是固定的（因为是在同一个状态下生成动作）
    - 只有时间步 t 在变化（从 T-1 到 0），因为去噪过程需要逐步进行
    
    与训练时的区别：
    - 训练时：不同的样本有不同的状态和时间步（batch 内并行）
    - 生成时：给定一个状态，生成一个动作，状态固定，时间步从 T-1 到 0 变化
    
    无需了解：repeat 参数（用于 ensemble，评估时用）。
    """
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)

    shape = observations.shape[:-1] + (self.action_dim,)

    # 调用 diffusion.py 的 p_sample_loop，传入 base_net（噪声预测网络）
    # 把 observations 固定住：因为在整个去噪循环中，状态 obs 不变，只有时间步 t 在变化
    # 这样 p_sample_loop 循环时只需要传 (x_t, t)，不用每次都传 obs
    return self.diffusion.p_sample_loop(
      rng_key=rng,
      model_forward=partial(self.base_net, observations),  # 固定 obs，只传 (x_t, t)
      shape=shape,
      clip_denoised=True,
    )

  def dpm_sample(self, rng, observations, deterministic=False, repeat=None):
    """
    DPM-Solver 采样：少步 ODE 求解，比 DDPM 快。
    
    整体流程：
    1. 构造噪声调度（NoiseScheduleVP）：定义扩散过程的参数（alpha_t, sigma_t）
    2. 创建 DPM-Solver：传入噪声预测网络和噪声调度
    3. 从噪声开始：x_T ~ N(0,I)
    4. 调用 DPM-Solver.sample()：用 ODE 求解器少步去噪（默认 15 步），得到动作 x_0
    
    与 DDPM 的区别：
    - DDPM：完整 T 步去噪（如 100 步），每步都采样
    - DPM-Solver：少步 ODE 求解（如 15 步），用数值方法"跳步"去噪，更快
    
    注意：DPM-Solver 的详细实现在 dpm_solver.py 中，这里只是调用它的接口。
    无需深入理解 DPM-Solver 的 ODE 求解细节，知道「少步快速去噪」即可。
    
    :param observations: 环境的观测/状态（state），形状 [batch_size, obs_dim]
                        给定状态 s，生成动作 a（x_0）
    :param repeat: 无需了解：用于 ensemble（评估时采多个动作）
    """
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)
    noise_clip = True  # 无需了解：是否裁剪噪声（工程细节）

    shape = observations.shape[:-1] + (self.action_dim,)

    # ===== 第一步：构造噪声调度 =====
    # 噪声调度：定义扩散过程中不同时间步的噪声水平（alpha_t, sigma_t 等）
    # 与训练时网络预测的噪声不同：
    # - 训练时：网络预测的是「噪声 ε」（一个向量），用于计算 loss
    # - 噪声调度：定义的是「扩散过程的参数」（alpha_t, sigma_t），用于 DPM-Solver 的 ODE 求解
    # 简单理解：噪声调度 = 扩散过程的「时间表」，告诉 DPM-Solver「每个时间步该用多少噪声」
    ns = NoiseScheduleVP(
      schedule='discrete', alphas_cumprod=self.diffusion.alphas_cumprod
    )

    # ===== 第二步：包装噪声预测网络 =====
    # 无需了解：wrap_model 只是工程细节（噪声裁剪），不影响理解
    def wrap_model(model_fn):
      """无需了解：包装函数，用于噪声裁剪（工程细节）"""
      def wrapped_model_fn(x, t):
        t = (t - 1. / ns.total_N) * ns.total_N

        out = model_fn(x, t)
        # add noise clipping
        if noise_clip:
          t = t.astype(jnp.int32)
          x_w = _extract_into_tensor(
            self.diffusion.sqrt_recip_alphas_cumprod, t, x.shape
          )
          e_w = _extract_into_tensor(
            self.diffusion.sqrt_recipm1_alphas_cumprod, t, x.shape
          )
          max_value = (self.diffusion.max_value + x_w * x) / e_w
          min_value = (self.diffusion.min_value + x_w * x) / e_w

          out = out.clip(min_value, max_value)
        return out

      return wrapped_model_fn

    # ===== 第三步：创建 DPM-Solver =====
    # DPM-Solver 的详细实现在 dpm_solver.py 中（第 401 行开始）
    # 这里只是创建 DPM-Solver 对象，传入：
    # - model_fn：噪声预测网络（base_net，固定了 observations）
    # - noise_schedule：噪声调度（ns）
    # - predict_x0：是否预测 x_0（本项目预测噪声 ε，所以是 False）
    dpm_sampler = DPM_Solver(
      model_fn=wrap_model(partial(self.base_net, observations)),
      noise_schedule=ns,
      predict_x0=self.diffusion.model_mean_type is ModelMeanType.START_X,
    )
    
    # ===== 第四步：从噪声开始，调用 DPM-Solver 采样 =====
    x = jax.random.normal(rng, shape)  # x_T ~ N(0,I)，从纯噪声开始
    # DPM-Solver.sample() 的详细实现在 dpm_solver.py 第 1196 行
    # 它会用 ODE 求解器，少步（默认 15 步）去噪，得到动作 x_0
    out = dpm_sampler.sample(x, steps=self.dpm_steps, t_end=self.dpm_t_end)

    return out

  def ddim_sample(self, rng, observations, deterministic=False, repeat=None):
    """
    DDIM 采样：确定性、可跳步，比 DDPM 快。
    无需了解：repeat 参数（用于 ensemble）。
    """
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)

    shape = observations.shape[:-1] + (self.action_dim,)

    # 调用 diffusion.py 的 ddim_sample_loop
    return self.diffusion.ddim_sample_loop(
      rng_key=rng,
      model_forward=partial(self.base_net, observations),
      shape=shape,
      clip_denoised=True,
    )

  def forward(self, observations, actions, t):
    """
    前向传播：直接调用 base_net，输入 (obs, action, t)，输出预测的噪声。
    训练时通过 loss() 调用，这里单独暴露出来方便调试。
    """
    return self.base_net(observations, actions, t)

  def loss(self, rng_key, observations, actions, ts):
    """
    训练损失：调用 diffusion.py 的 training_losses，计算扩散 MSE。
    这是训练时的主要接口，在 dql.py 的 get_diff_loss 中被调用。
    """
    terms = self.diffusion.training_losses(
      rng_key,
      model_forward=partial(self.base_net, observations),  # 把 observations 固定住
      x_start=actions,  # 真实动作（x_0）
      t=ts  # 随机时间步
    )
    return terms
    # 下面两行是死代码（永远不会执行），可以忽略
    noise = jax.random.normal(rng_key, actions.shape, dtype=actions.dtype)
    out = self.base_net(observations, noise, ts * 0)
    return {'loss': jnp.square(out - actions)}

  @property
  def max_action(self):
    """返回动作的最大值（用于 clip）"""
    return self.diffusion.max_value


class Critic(nn.Module):
  observation_dim: int
  action_dim: int
  arch: Tuple = (256, 256, 256)
  act: callable = mish
  use_layer_norm: bool = False
  orthogonal_init: bool = False

  @nn.compact
  @multiple_action_q_function
  def __call__(self, observations, actions):
    x = jnp.concatenate([observations, actions], axis=-1)

    for feat in self.arch:
      if self.orthogonal_init:
        x = nn.Dense(
          feat,
          kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
          bias_init=jax.nn.initializers.zeros,
        )(
          x
        )
      else:
        x = nn.Dense(feat)(x)
      if self.use_layer_norm:
        x = nn.LayerNorm()(x)
      x = self.act(x)

    if self.orthogonal_init:
      x = nn.Dense(
        1,
        kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
        bias_init=jax.nn.initializers.zeros,
      )(
        x
      )
    else:
      x = nn.Dense(1)(x)
    return jnp.squeeze(x, -1)

  @property
  def input_size(self):
    return self.observation_dim + self.action_dim


class GaussianPolicy(nn.Module):
  action_dim: int
  log_std_min: float = -5.0
  log_std_max: float = 2.0
  temperature: float = 1.0

  @nn.compact
  def __call__(self, mean):
    log_stds = self.param(
      'log_stds', nn.initializers.zeros, (self.action_dim,)
    )
    log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
    return distrax.MultivariateNormalDiag(
      mean, jnp.exp(log_stds * self.temperature)
    )


class Value(nn.Module):
  observation_dim: int
  arch: Tuple = (256, 256, 256)
  act: callable = mish
  use_layer_norm: bool = False
  orthogonal_init: bool = False

  @nn.compact
  def __call__(self, observations):
    x = observations

    for feat in self.arch:
      if self.orthogonal_init:
        x = nn.Dense(
          feat,
          kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
          bias_init=jax.nn.initializers.zeros,
        )(
          x
        )
      else:
        x = nn.Dense(feat)(x)
      if self.use_layer_norm:
        x = nn.LayerNorm()(x)
      x = self.act(x)

    if self.orthogonal_init:
      x = nn.Dense(
        1,
        kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
        bias_init=jax.nn.initializers.zeros,
      )(
        x
      )
    else:
      x = nn.Dense(1)(x)
    return jnp.squeeze(x, -1)

  @property
  def input_size(self):
    return self.observation_dim
