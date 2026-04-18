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

"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection
of beta schedules.
"""

import enum
import math

import numpy as onp
import jax
import jax.numpy as np


def mean_flat(tensor):
  """
  Take the mean over all non-batch dimensions.
  """
  return tensor.mean(axis=list(range(1, len(tensor.shape))))

# 根据名称获取 beta 调度
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
  """
  Get a pre-defined beta schedule for the given name.

  The beta schedule library consists of beta schedules which remain similar
  in the limit of num_diffusion_timesteps.
  Beta schedules may be added, but should not be removed or changed once
  they are committed to maintain backwards compatibility.
  """

  # 线性调度
  if schedule_name == "linear":
    # Linear schedule from Ho et al, extended to work for any number of
    # diffusion steps.
    # NOTE: Double check beta start and end
    # 计算 scale 参数
    scale = 1000 / num_diffusion_timesteps
    # scale = 1.0
    beta_start = scale * 0.0001
    # 计算 beta 结束值
    beta_end = scale * 0.02
    # 返回 beta 数组（用 NumPy 避免在 GPU 上 JIT linspace，部分显卡会误编 sm_90a PTX）
    return onp.linspace(
      beta_start, beta_end, num_diffusion_timesteps, dtype=onp.float64
    )
  # 余弦调度
  elif schedule_name == "cosine":
    # 返回 beta 数组
    return betas_for_alpha_bar(
      num_diffusion_timesteps,
      lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2)**2,
    )
  # VP(Variance Preserving) 调度
  elif schedule_name == "vp":
    # 计算 T 参数
    T = num_diffusion_timesteps
    # 计算 t 数组
    t = onp.arange(1, T + 1)
    # 计算 b_max 参数
    b_max = 10.
    # 计算 b_min 参数
    b_min = 0.1
    # 计算 alpha 数组
    alpha = onp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    # 计算 beta 数组
    betas = 1 - alpha
    return betas
  else:
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
  """
  Create a beta schedule that discretizes the given alpha_t_bar function,
  which defines the cumulative product of (1-beta) over time from t = [0,1].

  :param num_diffusion_timesteps: the number of betas to produce.
  :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                    produces the cumulative product of (1-beta) up to that
                    part of the diffusion process.
  :param max_beta: the maximum beta to use; use values lower than 1 to
                   prevent singularities.
  """
  betas = []
  for i in range(num_diffusion_timesteps):
    t1 = i / num_diffusion_timesteps
    t2 = (i + 1) / num_diffusion_timesteps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
  return onp.array(betas)


class ModelMeanType(enum.Enum):
  """
  Which type of output the model predicts.
  """

  PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
  START_X = enum.auto()  # the model predicts x_0
  EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
  """
  What is used as the model's output variance.

  The LEARNED_RANGE option has been added to allow the model to predict
  values between FIXED_SMALL and FIXED_LARGE, making its job easier.
  """

  LEARNED = enum.auto()
  FIXED_SMALL = enum.auto()
  FIXED_LARGE = enum.auto()
  LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
  MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
  RESCALED_MSE = (
    enum.auto()
  )  # use raw MSE loss (with RESCALED_KL when learning variances)
  KL = enum.auto()  # use the variational lower-bound
  RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

  def is_vb(self):
    return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
  """
  Utilities for training and sampling diffusion models.

  Ported directly from here, and adapted over time to further experimentation.
  https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

  :param betas: a 1-D numpy array of betas for each diffusion timestep,
                starting at T and going to 1.
  :param model_mean_type: a ModelMeanType determining what the model outputs.
  :param model_var_type: a ModelVarType determining how variance is output.
  :param loss_type: a LossType determining the loss function to use.
  :param rescale_timesteps: if True, pass floating point timesteps into the
                            model so that they are always scaled like in the
                            original paper (0 to 1000).
  """
  # 初始化扩散模型
  def __init__(
    self,
    *,
    num_timesteps,
    schedule_name,
    model_mean_type,
    model_var_type,
    loss_type,
    min_value=-1.,
    max_value=1.,
    rescale_timesteps=False,
  ):
    # 保存参数
    self.schedule_name = schedule_name
    self.model_mean_type = model_mean_type
    self.model_var_type = model_var_type
    self.loss_type = loss_type
    self.rescale_timesteps = rescale_timesteps
    self.min_value = min_value
    self.max_value = max_value

    # 预计算全部在 CPU（NumPy）完成，再转为 JAX 数组；避免在部分 GPU 上误编 sm_90a PTX。
    betas = onp.asarray(
      get_named_beta_schedule(schedule_name, num_timesteps), dtype=onp.float32
    )
    assert len(betas.shape) == 1, "betas must be 1-D"
    assert (betas > 0).all() and (betas <= 1).all()

    self.num_timesteps = int(betas.shape[0])

    alphas = 1.0 - betas
    alphas_cumprod = onp.cumprod(alphas, axis=0)
    alphas_cumprod_prev = onp.append(1.0, alphas_cumprod[:-1])
    alphas_cumprod_next = onp.append(alphas_cumprod[1:], 0.0)
    assert alphas_cumprod_prev.shape == (self.num_timesteps,)

    sqrt_alphas_cumprod = onp.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = onp.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = onp.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = onp.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = onp.sqrt(1.0 / alphas_cumprod - 1)

    posterior_variance = (
      betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_log_variance_clipped = onp.log(
      onp.append(posterior_variance[1], posterior_variance[1:])
    )
    posterior_mean_coef1 = (
      betas * onp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
      (1.0 - alphas_cumprod_prev) * onp.sqrt(alphas) /
      (1.0 - alphas_cumprod)
    )
    ws = betas / (2 * (1.0 - alphas_cumprod) * alphas)
    normalized_ts_weights = ws * num_timesteps / ws.sum()

    self.betas = np.asarray(betas)
    self.alphas_cumprod = np.asarray(alphas_cumprod)
    self.alphas_cumprod_prev = np.asarray(alphas_cumprod_prev)
    self.alphas_cumprod_next = np.asarray(alphas_cumprod_next)
    self.sqrt_alphas_cumprod = np.asarray(sqrt_alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = np.asarray(sqrt_one_minus_alphas_cumprod)
    self.log_one_minus_alphas_cumprod = np.asarray(log_one_minus_alphas_cumprod)
    self.sqrt_recip_alphas_cumprod = np.asarray(sqrt_recip_alphas_cumprod)
    self.sqrt_recipm1_alphas_cumprod = np.asarray(sqrt_recipm1_alphas_cumprod)
    self.posterior_variance = np.asarray(posterior_variance)
    self.posterior_log_variance_clipped = np.asarray(posterior_log_variance_clipped)
    self.posterior_mean_coef1 = np.asarray(posterior_mean_coef1)
    self.posterior_mean_coef2 = np.asarray(posterior_mean_coef2)
    self.ts_weights = np.asarray(ws)
    self.normalized_ts_weights = np.asarray(normalized_ts_weights)

  def q_mean_variance(self, x_start, t):
    """
    计算前向分布 q(x_t | x_0) 的均值、方差和 log 方差。
    
    前向加噪公式：x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε，其中 ε ~ N(0,I)
    因此 q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)I)
    
    :param x_start: 干净输入 x_0，形状 [N x C x ...]（N 是 batch size，C 是特征维度）
    :param t: 时间步索引数组，形状 [N]，每个元素表示对应样本的时间步（0 表示第 1 步）
    :return: 三元组 (mean, variance, log_variance)，都是 x_start 的形状
             - mean: q(x_t | x_0) 的均值，即 √ᾱ_t·x_0
             - variance: q(x_t | x_0) 的方差，即 1-ᾱ_t
             - log_variance: 方差的 log 形式，即 log(1-ᾱ_t)，用于 KL 散度等计算
    
    用途：用于计算前向分布的统计量，在 _prior_bpd 等函数中用于计算 KL 散度
    """
    # 均值：μ_t = √ᾱ_t·x_0
    # _extract_into_tensor 根据时间步 t 从数组中提取对应的系数
    mean = (
      _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) *
      x_start
    )
    
    # 方差：σ²_t = 1 - ᾱ_t
    # 前向加噪时，噪声的方差是 1-ᾱ_t
    variance = _extract_into_tensor(
      1.0 - self.alphas_cumprod, t, x_start.shape
    )
    
    # log 方差：log(σ²_t) = log(1 - ᾱ_t)
    # 用于 KL 散度等需要 log-variance 的计算（避免数值不稳定）
    log_variance = _extract_into_tensor(
      self.log_one_minus_alphas_cumprod, t, x_start.shape
    )
    return mean, variance, log_variance
  
  # 前向加噪函数
  def q_sample(self, x_start, t, noise):
    """
    前向加噪：从 q(x_t | x_0) 采样，对干净数据 x_0 加噪得到 x_t。
    
    数学公式：x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
    其中 ε ~ N(0,I) 是标准高斯噪声，ᾱ_t 是累积保留比例。
    
    :param x_start: 干净输入 x_0，形状 [N x C x ...]（N 是 batch size，C 是特征维度）
                    在扩散策略中，x_0 是真实动作（来自数据集）
    :param t: 时间步索引数组，形状 [N]，每个元素表示对应样本的时间步（0 表示第 1 步）
              不同的样本可以有不同的时间步，实现 batch 内并行训练
    :param noise: 预采样的高斯噪声 ε，形状与 x_start 相同，通常从 N(0,I) 采样得到
                  注意：噪声需要提前采样好传入，而不是在函数内部采样
    :return: 加噪后的数据 x_t，形状与 x_start 相同
    
    用途：训练时用于对真实动作加噪，得到训练样本 (x_t, t)，然后让网络预测噪声
    调用位置：主要在 training_losses 中使用（第 790 行）
    
    设计说明：
    - 使用 self.sqrt_alphas_cumprod 等：系数在 __init__ 中预先计算好（所有时间步），避免重复计算
    - 使用 _extract_into_tensor：根据时间步 t 动态提取对应的系数（batch 内不同样本可能有不同时间步）
    """
    # 确保噪声和初始数据形状相同，防止维度不匹配的错误
    assert noise.shape == x_start.shape
    
    # 前向加噪公式：x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
    # 第一部分：√ᾱ_t·x_0（保留的信号部分）
    # _extract_into_tensor 根据时间步 t 从预先计算的数组中提取对应的系数
    signal_part = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    
    # 第二部分：√(1-ᾱ_t)·ε（添加的噪声部分）
    # _extract_into_tensor 根据时间步 t 从预先计算的数组中提取对应的系数
    noise_part = _extract_into_tensor(
      self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    ) * noise
    
    # 返回加噪后的数据
    return signal_part + noise_part
  
  # 前向过程的后验（给定 x_0 时 q(x_{t-1}|x_t,x_0) 的闭式，非学到的反向）
  def q_posterior_mean_variance(self, x_start, x_t, t):
    """
    计算前向过程的后验分布 q(x_{t-1} | x_t, x_0) 的均值、方差和 log 方差。
    
    这是前向过程的解析后验（闭式解，不依赖网络），通过贝叶斯公式和高斯分布的共轭性质推导得出。
    训练时用作目标，让网络学习的分布 p_θ(x_{t-1}|x_t) 去拟合这个后验。
    
    数学公式：
    - 均值：μ_post = c₁(t)·x_0 + c₂(t)·x_t，其中 c₁、c₂ 在 __init__ 中预先计算
    - 方差：σ²_post = β_t·(1-ᾱ_{t-1})/(1-ᾱ_t)
    
    :param x_start: 干净输入 x_0，形状 [N x C x ...]
    :param x_t: 加噪后的数据 x_t，形状与 x_start 相同
    :param t: 时间步索引数组，形状 [N]，每个元素表示对应样本的时间步
    :return: 三元组 (posterior_mean, posterior_variance, posterior_log_variance_clipped)
             - posterior_mean: 后验均值，形状与 x_start 相同
             - posterior_variance: 后验方差，形状与 x_start 相同
             - posterior_log_variance_clipped: 后验方差的 log 形式（做了 clipping 处理）
    
    用途：训练时用于计算 ELBO 中的 KL 项，生成时用于计算 p_mean_variance 的均值
    注意：使用 _extract_into_tensor 根据时间步 t 动态提取系数（batch 内不同样本可能有不同时间步）
    """
    # 确保初始数据和加噪后的数据形状相同
    assert x_start.shape == x_t.shape
    
    # 后验均值：μ_post = c₁(t)·x_0 + c₂(t)·x_t
    # _extract_into_tensor 根据时间步 t 从预先计算的系数数组中提取对应的值
    # 这样设计的原因：batch 内不同样本可能有不同的时间步，需要动态提取对应的系数
    posterior_mean = (
      _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
      _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    
    # 后验方差：σ²_post = β_t·(1-ᾱ_{t-1})/(1-ᾱ_t)
    # 在 __init__ 中预先计算好所有时间步的方差，这里根据 t 动态提取
    posterior_variance = _extract_into_tensor(
      self.posterior_variance, t, x_t.shape
    )
    
    # 后验方差的 log 形式（做了 clipping 处理，避免 log(0)）
    # 在 __init__ 中已经做了 clipping（第一个时间步用第二个时间步的值），这里直接提取
    posterior_log_variance_clipped = _extract_into_tensor(
      self.posterior_log_variance_clipped, t, x_t.shape
    )
    
    # 确保返回值的 batch 维度一致
    assert (
      posterior_mean.shape[0] == posterior_variance.shape[0] ==
      posterior_log_variance_clipped.shape[0] == x_start.shape[0]
    )
    return posterior_mean, posterior_variance, posterior_log_variance_clipped
  
  # 学到的反向去噪 p(x_{t-1}|x_t)，依赖模型输出
  def p_mean_variance(self, model_output, x, t, clip_denoised=True):
    """
    计算学到的反向分布 p(x_{t-1} | x_t) 的均值、方差和预测的 x_0。
    
    这是扩散模型的核心函数：用网络输出计算去噪分布。关键思想是：
    - 网络预测噪声 ε（或 x_0，或 x_{t-1}），根据 model_mean_type 决定
    - 从网络输出得到 pred_xstart（预测的 x_0）
    - 用 q_posterior_mean_variance 的公式，但把真实 x_0 换成预测的 pred_xstart
    - 这样得到 p(x_{t-1}|x_t) 的均值（方差由 schedule 固定或网络学习）
    
    数学公式：
    - 均值：μ_p = c₁(t)·pred_xstart + c₂(t)·x_t（用 q 的后验公式，x_0 换成预测值）
    - 方差：由 model_var_type 决定（固定或学习）
    
    :param model_output: 网络的输出，根据 model_mean_type 可能是：
                         - EPSILON：预测的噪声 ε（本项目默认）
                         - START_X：直接预测的 x_0
                         - PREVIOUS_X：直接预测的 x_{t-1}
    :param x: 当前时间步的数据 x_t，形状 [N x C x ...]
    :param t: 时间步索引数组，形状 [N]
    :param clip_denoised: 如果为 True，将 pred_xstart 裁剪到 [min_value, max_value]
    :return: 字典，包含以下键：
             - 'mean': p(x_{t-1}|x_t) 的均值，形状与 x 相同
             - 'variance': p(x_{t-1}|x_t) 的方差，形状与 x 相同
             - 'log_variance': 方差的 log 形式
             - 'pred_xstart': 网络预测的 x_0，形状与 x 相同
    
    用途：训练时用于计算 loss，生成时用于采样 x_{t-1}
    关键：用同一套 q 的后验公式，只是 x_0 换成预测值（训练时用真实 x_0，生成时用预测 x_0）
    """
    # 获取数据形状：B = batch size（批次大小）, C = action_dim（动作维度）
    # 在扩散策略中，x 是动作（action），形状通常是 [batch_size, action_dim]
    # 例如：x.shape = [32, 14] 表示 32 个样本，每个动作是 14 维 → B=32, C=14
    # x.shape[:2] 表示取形状的前两个维度
    # 这是 Python 的解包语法：将元组 (32, 14) 解包给 B 和 C
    B, C = x.shape[:2]

    # ===== 第一部分：计算方差 =====
    # 根据 model_var_type 决定方差是固定的还是学习的
    if self.model_var_type in [
      ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE
    ]:
      # 学习方差：网络同时输出均值和方差
      # 此时 model_output 的形状是 [B, C*2, ...]，前一半是均值，后一半是方差
      assert model_output.shape == (B, C * 2, *x.shape[2:])
      # 拆分：前一半是均值预测，后一半是方差预测
      # np.split(model_output, C, axis=1)：在 axis=1（特征维度）上分割
      # 将 [B, C*2, ...] 分成两个 [B, C, ...] 的数组
      # axis=1 表示在第 1 个维度（索引从 0 开始，所以是第 2 个维度）上分割
      model_output, model_var_values = np.split(model_output, C, axis=1)
      
      if self.model_var_type == ModelVarType.LEARNED:
        # 网络直接输出 log-variance
        model_log_variance = model_var_values
        model_variance = np.exp(model_log_variance)
      else:
        # LEARNED_RANGE：网络输出 [-1, 1] 的值，映射到 [min_var, max_var] 范围
        min_log = _extract_into_tensor(
          self.posterior_log_variance_clipped, t, x.shape
        )
        max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
        # 将 [-1, 1] 映射到 [min_log, max_log]
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = np.exp(model_log_variance)
    else:
      # 固定方差：方差由 schedule 决定，不学习（本项目默认 FIXED_SMALL）
      # 注意：这里用的是数学闭式解（预先在 __init__ 中计算好的数组）
      # 不是抽样，而是从预先计算的数组中提取对应时间步的值
      model_variance, model_log_variance = {
        ModelVarType.FIXED_LARGE:
          (
            np.append(self.posterior_variance[1], self.betas[1:]),
            np.log(np.append(self.posterior_variance[1], self.betas[1:])),
          ),
        ModelVarType.FIXED_SMALL:
          (
            self.posterior_variance,  # 使用后验方差（数学闭式解）
            self.posterior_log_variance_clipped,  # 使用后验 log-variance（数学闭式解）
          ),
      }[self.model_var_type]
      # 根据时间步 t 动态提取对应的方差
      # _extract_into_tensor 不是抽样，而是从预先计算的数组中提取对应时间步的值
      # 因为 batch 内不同样本可能有不同的时间步，需要动态提取
      model_variance = _extract_into_tensor(model_variance, t, x.shape)
      model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
    
    # ===== 第二部分：计算均值和 pred_xstart =====
    # 辅助函数：如果需要，将 pred_xstart 裁剪到有效范围
    def process_xstart(x):
      if clip_denoised:
        return x.clip(self.min_value, self.max_value)
      return x
    
    # 根据 model_mean_type 决定网络预测什么，然后转换成 pred_xstart
    # 注意：pred_xstart 有两个用途：
    # 1. 生成时：用于反向去噪（从 x_t 去噪到 x_{t-1}）
    # 2. 训练时：用于 action approximation（EDP/DQL 的 guide loss）
    #    在 dql.py 的 get_diff_terms 中，pred_xstart 被当作预测的动作用于计算 Q(s, a)
    if self.model_mean_type == ModelMeanType.PREVIOUS_X:
      # 网络直接预测 x_{t-1}
      # 这种情况较少见，需要从 x_{t-1} 反推 x_0
      pred_xstart = process_xstart(
        self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
      )
      model_mean = model_output  # 均值就是网络输出
    elif self.model_mean_type in [
      ModelMeanType.START_X, ModelMeanType.EPSILON
    ]:
      if self.model_mean_type == ModelMeanType.START_X:
        # 网络直接预测 x_0（较少用）
        pred_xstart = process_xstart(model_output)
      else:
        # 网络预测噪声 ε（本项目默认），需要反推 x_0
        # 公式：x_0 = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t
        # 这是标准的扩散模型公式，不是 EDP 特有的
        pred_xstart = process_xstart(
          self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        )
      
      # 关键步骤：用 q_posterior_mean_variance 的公式计算均值
      # 但把真实 x_0 换成预测的 pred_xstart
      # 这样得到 p(x_{t-1}|x_t) 的均值 = c₁(t)·pred_xstart + c₂(t)·x_t
      # 这是 DQL/EDP 的标准做法，不是 action approximation
      # action approximation 是在训练时用 pred_xstart 当作动作计算 guide loss（见 dql.py 第 311-313 行）
      model_mean, _, _ = self.q_posterior_mean_variance(
        x_start=pred_xstart, x_t=x, t=t
      )
    else:
      raise NotImplementedError(self.model_mean_type)

    # 确保所有返回值的形状一致
    assert (
      model_mean.shape == model_log_variance.shape == pred_xstart.shape ==
      x.shape
    )
    return {
      "mean": model_mean,
      "variance": model_variance,
      "log_variance": model_log_variance,
      "pred_xstart": pred_xstart,
    }

  # 后向去噪函数（简化版本，另一种实现方式）
  def p_mean_variance_(self, model_output, x, t, clip_denoised=True):

    # 处理初始数据
    def process_xstart(x):
      if clip_denoised:
        return x.clip(self.min_value, self.max_value)
      return x

    # 根据模型均值类型选择不同的均值
    if self.model_mean_type == ModelMeanType.START_X:
      pred_xstart = process_xstart(model_output)
    elif self.model_mean_type == ModelMeanType.EPSILON:
      pred_xstart = process_xstart(
        self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
      )

    posterior_mean, posterior_variance, posterior_log_variance = \
      self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
    return {
      "mean": posterior_mean,
      "variance": posterior_variance,
      "log_variance": posterior_log_variance,
      "pred_xstart": pred_xstart,
    }

  def _predict_xstart_from_eps(self, x_t, t, eps):
    assert x_t.shape == eps.shape
    return (
      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
      - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) *
      eps
    )

  def _predict_xstart_from_xprev(self, x_t, t, xprev):
    assert x_t.shape == xprev.shape
    return (  # (xprev - coef2*x_t) / coef1
        _extract_into_tensor(
            1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
        - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        )
        * x_t
    )

  def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
    return (
      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
      - pred_xstart
    ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

  def _scale_timesteps(self, t):
    if self.rescale_timesteps:
      return t.float() * (1000.0 / self.num_timesteps)
    return t

  def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
    """
    Compute the mean for the previous step, given a function cond_fn that
    computes the gradient of a conditional log probability with respect to
    x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
    condition on y.

    This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
    """
    gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
    new_mean = (
      p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
    )
    return new_mean

  def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
    """
    Compute what the p_mean_variance output would have been, should the
    model's score function be conditioned by cond_fn.

    See condition_mean() for details on cond_fn.

    Unlike condition_mean(), this instead uses the conditioning strategy
    from Song et al (2020).
    """
    alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

    eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
    eps = eps - (1 - alpha_bar).sqrt(
    ) * cond_fn(x, self._scale_timesteps(t), **model_kwargs)

    out = p_mean_var.copy()
    out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
    out["mean"], _, _ = self.q_posterior_mean_variance(
      x_start=out["pred_xstart"], x_t=x, t=t
    )
    return out

  def p_sample(
    self,
    rng,
    model_output,
    x,
    t,
    clip_denoised=True,
    cond_fn=None,
    model_kwargs=None,
  ):
    """
    从 x_t 采样 x_{t-1}：根据网络预测的噪声，计算均值，再加随机噪声采样。
    
    流程：
    1. 用 p_mean_variance 计算 p(x_{t-1}|x_t) 的均值 μ 和方差 σ²
       - 内部会预测 x_0（pred_xstart），但输出的是均值 μ = c₁(t)·pred_x0 + c₂(t)·x_t
    2. 在均值 μ 上加随机噪声：x_{t-1} = μ + σ·ε，其中 ε ~ N(0,I)
    3. 注意：不是直接在预测的 x_0 上加噪声，而是在均值 μ 上加噪声
    
    :param rng: JAX 随机数生成器密钥
    :param model_output: 网络预测的噪声 ε（或 x_0，由 model_mean_type 决定）
    :param x: 当前时间步的数据 x_t，形状 [N x C x ...]
    :param t: 时间步索引数组，形状 [N]
    :param clip_denoised: 无需了解：是否裁剪 pred_xstart（默认即可）
    :param cond_fn: 无需了解：条件生成函数（本项目不使用）
    :param model_kwargs: 无需了解：额外的模型参数（本项目不使用）
    :return: 字典，包含：
             - 'sample': 采样的 x_{t-1}，形状与 x 相同
             - 'pred_xstart': 预测的 x_0，形状与 x 相同
    """
    # 计算 p(x_{t-1}|x_t) 的均值 μ 和方差 σ²（内部会预测 x_0，但输出的是均值）
    out = self.p_mean_variance(model_output, x, t, clip_denoised=clip_denoised)
    # 采样随机噪声 ε ~ N(0,I)，形状与 x 相同
    noise = jax.random.normal(rng, x.shape, dtype=x.dtype)

    # 当 t=0 时，不加噪声（因为已经是 x_0 了）
    # nonzero_mask：t != 0 时为 1，t == 0 时为 0
    nonzero_mask = np.expand_dims((t != 0).astype(np.float32), axis=-1)
    
    # 无需了解：条件生成（本项目不使用）
    if cond_fn is not None:
      out["mean"] = self.condition_mean(
        cond_fn, out, x, t, model_kwargs=model_kwargs
      )
    
    # ===== 重要：DDPM 的随机噪声不能为 0 =====
    # 
    # DDPM 的采样公式：x_{t-1} = μ + σ·ε（当 t != 0 时）
    # 其中：
    #   - μ = out["mean"]（后验分布的均值，来自 q_posterior_mean_variance）
    #   - σ = exp(0.5 * out["log_variance"])（后验分布的方差）
    #   - ε ~ N(0,I)（随机噪声）
    # 
    # DDPM 的 σ 来自哪里？
    #   - 来自 posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    #   - 这是由 beta schedule 决定的（在 __init__ 中预先计算，第 240-241 行）
    #   - 因为 betas > 0（噪声强度不能为 0），所以 σ > 0（不能为 0）
    # 
    # 为什么 DDPM 不能取 eta=0？
    #   - DDPM 根本没有 eta 参数！eta 是 DDIM 特有的参数
    #   - DDPM 的随机性来自后验分布的方差 σ（由 beta schedule 决定），不能为 0
    #   - DDIM 的随机性来自 eta 参数，可以设为 0（sigma = eta * sqrt(...)）
    # 
    # 对比：
    #   - DDPM：x_{t-1} = μ + σ·ε，其中 σ > 0（由 beta schedule 决定），每步都有随机性
    #   - DDIM：x_{t-1} = mean_pred + sigma * noise，其中 sigma = eta * sqrt(...)
    #           当 eta=0 时，sigma=0，完全确定性
    # 
    # 采样公式：x_{t-1} = μ + σ·ε（当 t != 0 时）
    # 注意：这是在均值 μ 上加噪声，不是在预测的 x_0 上加噪声
    sample = out["mean"] + \
      nonzero_mask * np.exp(0.5 * out["log_variance"]) * noise
    return {"sample": sample, "pred_xstart": out["pred_xstart"]}

  def p_sample_loop(
    self,
    rng_key,
    model_forward,
    shape,
    clip_denoised=True,
    cond_fn=None,
    model_kwargs=None,
  ):
    """
    扩散模型的生成循环：从噪声 x_T 逐步去噪到 x_0（动作）。
    
    这是扩散模型生成时的核心函数：完整 T 步去噪，每步调用网络预测噪声，然后采样 x_{t-1}。
    
    流程：
    1. 从纯噪声 x_T ~ N(0,I) 开始
    2. 循环 T 步（从 T-1 到 0）：
       - 网络预测噪声：ε_pred = model_forward(x_t, t)
       - 采样 x_{t-1}：用 p_sample 根据预测的噪声采样下一步
    3. 返回最终的动作 x_0
    
    :param rng_key: JAX 随机数生成器密钥
    :param model_forward: 噪声预测网络的函数，输入 (x_t, t)，输出预测的噪声 ε
                         在 DiffusionPolicy 中，这是 partial(self.base_net, observations)
                         说明：partial 是 Python 的 functools.partial，可以"固定"函数的某些参数
                         例如：base_net(obs, x_t, t) 需要三个参数
                              partial(base_net, obs) 创建一个新函数，obs 已经固定
                              新函数只需要传 (x_t, t)，这样 p_sample_loop 调用时更方便
    :param shape: 输出数组的形状，例如 [batch_size, action_dim] = [32, 14]
                  表示 32 个样本，每个样本是 14 维向量（动作维度）
                  shape 在很多函数里用来指定输出数组的形状
    :param clip_denoised: 如果为 True，将 pred_xstart 裁剪到 [min_value, max_value]
                         无需了解：知道是裁剪即可
    :param cond_fn: 无需了解：条件生成函数（本项目不使用条件生成）
    :param model_kwargs: 无需了解：额外的模型参数（本项目不使用）
    :return: 生成的动作 x_0，形状与 shape 相同
    
    调用位置：在 nets.py 的 DiffusionPolicy.ddpm_sample 中被调用（第 156 行）
    用途：生成时从噪声逐步去噪得到动作
    """
    # ===== 第一步：从纯噪声开始 =====
    rng_key, sample_key = jax.random.split(rng_key)
    x = jax.random.normal(sample_key, shape)  # x_T ~ N(0,I)，形状 [batch_size, action_dim]

    # ===== 第二步：循环 T 步去噪 =====
    # 注意：循环结构看起来和 ddim_sample_loop 一样，但核心区别在于每步调用的采样函数不同：
    # - p_sample_loop 调用 p_sample（第 748 行）：x_{t-1} = μ + σ·ε（随机采样，必须完整 T 步）
    # - ddim_sample_loop 调用 ddim_sample（第 911 行）：x_{t-1} = mean_pred + sigma*noise（当 eta=0 时确定性，可以跳步）
    # 
    # p_sample 的采样公式（第 684-685 行）：每步都加随机噪声，必须完整 T 步
    # ddim_sample 的采样公式（第 817-825 行）：当 eta=0 时完全确定性，可以跳步加速
    indices = list(range(self.num_timesteps))[::-1]
    for i in indices:
      # 当前时间步 t = i（例如 T-1, T-2, ..., 0）
      # shape[:-1] 表示除了最后一维的所有维度（例如 shape=[32,14] → shape[:-1]=[32]）
      # 这样 t 的形状是 [batch_size]，每个样本都是同一个时间步 i
      t = np.ones(shape[:-1], dtype=np.int32) * i
      
      # 网络预测噪声：ε_pred = model_forward(x_t, t)
      model_output = model_forward(x, self._scale_timesteps(t))
      
      # 采样 x_{t-1}：从 x_t 采样 x_{t-1}（使用 DDPM 的随机采样公式）
      # 流程：x_t → p_mean_variance 算均值（内部会预测 x_0，但输出的是均值 μ）→ 在均值上加随机噪声 → x_{t-1}
      # 注意：不是直接在预测的 x_0 上加噪声，而是在均值 μ = c₁(t)·pred_x0 + c₂(t)·x_t 上加噪声
      # p_sample 的详细实现在第 636 行，采样公式在第 684-685 行：x_{t-1} = μ + σ·ε（每步都有随机性）
      rng_key, sample_key = jax.random.split(rng_key)
      out = self.p_sample(
        sample_key, model_output, x, t, clip_denoised, cond_fn, model_kwargs
      )
      
      # 更新 x：x = x_{t-1}，为下一步做准备
      x = out["sample"]
    
    # ===== 第三步：返回最终动作 =====
    # 循环结束后，x 就是 x_0（去噪完成，得到动作）
    return x

  def ddim_sample(
    self,
    rng_key,
    model_putput,
    x,
    t,
    clip_denoised=True,
    cond_fn=None,
    model_kwargs=None,
    eta=0.0,
  ):
    """
    用 DDIM 方式从 x_t 采样 x_{t-1}：确定性采样，可以跳步，比 DDPM 快。
    
    与 p_sample（DDPM）的区别：
    - DDPM：每步都加随机噪声，必须完整 T 步
    - DDIM：确定性采样（eta=0 时），可以跳步（如每 10 步采样一次），更快
    
    流程：
    1. 用 p_mean_variance 计算均值（和 DDPM 相同）
    2. 用 DDIM 公式计算 x_{t-1}（与 DDPM 的公式不同）
       - 当 eta=0 时：完全确定性，可以跳步
       - 当 eta>0 时：有随机性，但比 DDPM 的随机性小
    
    :param rng_key: JAX 随机数生成器密钥
    :param model_putput: 网络预测的噪声 ε（或 x_0）
    :param x: 当前时间步的数据 x_t
    :param t: 时间步索引
    :param clip_denoised: 无需了解：是否裁剪 pred_xstart（默认即可）
    :param cond_fn: 无需了解：条件生成函数（本项目不使用）
    :param model_kwargs: 无需了解：额外的模型参数（本项目不使用）
    :param eta: 无需了解：DDIM 的随机性参数（eta=0 时完全确定性，默认即可）
    :return: 字典，包含 'sample'（x_{t-1}）和 'pred_xstart'（预测的 x_0）
    
    注意：DDIM 的详细公式在第 791-799 行，无需深入理解，知道「确定性、可跳步」即可。
    """
    # 计算 p(x_{t-1}|x_t) 的均值（和 DDPM 相同）
    out = self.p_mean_variance(model_putput, x, t, clip_denoised=clip_denoised)
    
    # 无需了解：条件生成（本项目不使用）
    if cond_fn is not None:
      out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

    # 从 pred_xstart 反推噪声 ε（因为网络可能预测 x_0 而不是 ε）
    eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

    # DDIM 公式需要的系数
    alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
    alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
    
    # ===== 概念 1：eta（随机性参数）和 sigma（方差） =====
    # 
    # eta：DDIM 的随机性参数，是一个可以调节的数字（通常设为 0 或 0.5）
    #   - eta = 0：完全确定性（每次结果相同）
    #   - eta > 0：有随机性（每次结果可能不同）
    # 
    # sigma：DDIM 的方差，由 eta 计算得出
    #   公式：sigma = eta * sqrt(...)
    #   当 eta=0 时，sigma = 0 * sqrt(...) = 0（数学上，任何数乘以 0 都是 0）
    #   当 eta>0 时，sigma > 0（具体值取决于时间步 t）
    # 
    # 简单理解：
    #   - eta 就像"随机性的开关"：eta=0 关闭随机性，eta>0 开启随机性
    #   - sigma 是"随机性的大小"：sigma=0 表示没有随机性，sigma>0 表示有随机性
    sigma = (
      eta * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) *
      np.sqrt(1 - alpha_bar / alpha_bar_prev)
    )
    
    # DDIM 采样公式（Equation 12 in DDIM paper）
    noise = jax.random.normal(rng_key, x.shape, dtype=x.dtype)  # 随机噪声 ε ~ N(0,I)
    
    # ===== 重要：DDIM 的 mean_pred 与 DDPM 的 μ 是不同的！ =====
    # 
    # DDPM 的 μ（在 p_sample 中，第 524-526 行）：
    #   μ = c₁(t)·pred_xstart + c₂(t)·x_t
    #   这是后验分布 q(x_{t-1}|x_t, x_0) 的均值公式（用预测的 x_0 代替真实的 x_0）
    #   来自 q_posterior_mean_variance 函数（第 346 行）
    # 
    # DDIM 的 mean_pred（这里）：
    #   mean_pred = pred_xstart * sqrt(alpha_bar_prev) + sqrt(1 - alpha_bar_prev - sigma**2) * eps
    #   这是 DDIM 的特殊公式，与 DDPM 的 μ 不同
    #   注意：这里用到了 eps（从 pred_xstart 反推的噪声），而 DDPM 的 μ 不需要 eps
    # 
    # 为什么不同？
    #   - DDPM 和 DDIM 是两种不同的采样方法，它们的数学公式本身就不同
    #   - DDPM 基于随机过程（每步都加随机噪声），DDIM 基于确定性 ODE（可以没有随机性）
    #   - 所以它们的"均值"公式也不同
    # 
    # 均值：用 pred_x0 和 eps 计算（DDIM 的特殊公式）
    mean_pred = (
      out["pred_xstart"] * np.sqrt(alpha_bar_prev) +
      np.sqrt(1 - alpha_bar_prev - sigma**2) * eps
    )
    # 当 t=0 时，不加噪声
    nonzero_mask = np.expand_dims((t != 0).astype(np.float32), axis=-1)
    
    # ===== 概念 2：完全确定性（Deterministic）vs 随机性（Stochastic） =====
    # 
    # DDIM 采样公式：x_{t-1} = mean_pred + sigma * noise
    # 
    # 当 eta=0 时，sigma=0，所以：
    #   x_{t-1} = mean_pred + 0 * noise = mean_pred（没有随机项）
    # 
    # "完全确定性"的含义：
    #   - 确定性 = 如果输入相同，输出也完全相同（就像数学函数 f(x) = 2x，输入 3 总是得到 6）
    #   - 随机性 = 即使输入相同，每次输出可能不同（就像掷骰子，每次结果可能不同）
    # 
    # 具体例子：
    #   - DDPM（随机性）：x_{t-1} = μ + σ·ε，其中 ε 是随机噪声
    #     输入 x_100 = [1.0, 2.0]，t=100
    #     第一次采样：x_99 = [0.95, 1.98]（因为 ε 是随机的）
    #     第二次采样：x_99 = [0.97, 1.99]（因为 ε 不同，结果也不同）
    #   
    #   - DDIM（eta=0，完全确定性）：x_{t-1} = mean_pred（没有随机项）
    #     输入 x_100 = [1.0, 2.0]，t=100
    #     第一次采样：x_99 = [0.96, 1.98]
    #     第二次采样：x_99 = [0.96, 1.98]（完全相同，因为公式里没有随机项）
    # 
    # 为什么确定性重要？
    #   - 确定性意味着：从同一个 x_T 开始，去噪路径是唯一确定的
    #   - 就像"如果我知道起点和路线，我就能准确预测终点"
    #   - 这允许"跳步"：不需要每步都走，可以跳过中间步骤
    # 
    # ===== 概念 3：可以跳步（Subsampling） =====
    # 
    # 核心理解：DDIM 通过确定性，增大了步长（可以跳步）
    # 
    # "跳步"的含义：
    #   - 完整采样：x_T → x_{T-1} → x_{T-2} → ... → x_1 → x_0（需要 T 步，例如 T=1000）
    #   - 跳步采样：x_T → x_{T-10} → x_{T-20} → ... → x_0（只需要 T/10 步，例如 100 步）
    #   - 跳步就是"跳过中间步骤，直接跳到更远的步骤"，相当于"增大步长"（从 1 步变成 10 步）
    # 
    # 为什么 DDPM 不能跳步？
    #   - DDPM 每步都加随机噪声，路径是随机的
    #   - 就像"走迷宫时每一步都随机选择方向"
    #   - 如果跳过中间步骤，无法知道"如果按完整路径走，x_{t-10} 应该是什么"
    #   - 因为每一步的随机噪声都不同，路径不可预测
    #   - 必须一步步来：x_100 → x_99 → x_98 → ... → x_90（不能跳过）
    # 
    # 为什么 DDIM（eta=0）可以跳步？
    #   - DDIM 是确定性的：从 x_T 到 x_0 的路径是唯一确定的
    #   - 就像"走迷宫时每一步都按固定路线走"
    #   - 即使跳过中间步骤，也可以直接计算：x_{t-10} = f(x_t, t, t-10)
    #   - 因为路径是确定的，所以可以从任意时间步 t 直接跳到任意时间步 s（s < t）
    #   - 可以跳步：x_100 → x_90（直接跳过 x_99, x_98, ..., x_91）
    # 
    # 跳步的好处：
    #   - 更快：1000 步变成 100 步（快 10 倍）
    #   - 但可能精度略低：因为跳过了中间步骤，可能不如完整步骤精确
    # 
    # 跳步的实现（在 ddim_sample_loop 中）：
    #   完整步骤：indices = [999, 998, 997, ..., 1, 0]（1000 步）
    #   跳步（每 10 步）：indices = [990, 980, 970, ..., 10, 0]（100 步）
    #   代码：indices = list(range(0, self.num_timesteps, 10))[::-1]
    sample = mean_pred + nonzero_mask * sigma * noise
    return {"sample": sample, "pred_xstart": out["pred_xstart"]}

  def ddim_reverse_sample(
    self,
    model_putput,
    x,
    t,
    clip_denoised=True,
    eta=0.0,
  ):
    """
    Sample x_{t+1} from the model using DDIM reverse ODE.
    """
    assert eta == 0.0, "Reverse ODE only for deterministic path"
    out = self.p_mean_variance(model_putput, x, t, clip_denoised=clip_denoised)
    # Usually our model outputs epsilon, but we re-derive it
    # in case we used x_start or x_prev prediction.
    eps = (
      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x -
      out["pred_xstart"]
    ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
    alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

    # Equation 12. reversed
    mean_pred = (
      out["pred_xstart"] * np.sqrt(alpha_bar_next) +
      np.sqrt(1 - alpha_bar_next) * eps
    )

    return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

  def ddim_sample_loop(
    self,
    rng_key,
    model_forward,
    shape,
    clip_denoised=True,
    cond_fn=None,
    model_kwargs=None,
    eta=0.0,
  ):
    """
    DDIM 采样循环：从噪声 x_T 逐步去噪到 x_0（动作）。
    
    与 p_sample_loop（DDPM）的区别：
    - DDPM：每步都加随机噪声，完整 T 步去噪
    - DDIM：确定性采样（eta=0 时），可以跳步，更快
    
    流程：
    1. 从纯噪声开始：x_T ~ N(0,I)
    2. 循环 T 步（从 T-1 到 0）：
       - 网络预测噪声：ε_pred = model_forward(x_t, t)
       - 用 ddim_sample 采样 x_{t-1}（DDIM 公式，与 DDPM 不同）
    3. 返回最终的动作 x_0
    
    :param rng_key: JAX 随机数生成器密钥
    :param model_forward: 噪声预测网络的函数，输入 (x_t, t)，输出预测的噪声 ε
                         在 DiffusionPolicy 中，这是 partial(self.base_net, observations)
    :param shape: 输出数组的形状，例如 [batch_size, action_dim]
    :param clip_denoised: 无需了解：是否裁剪 pred_xstart（默认即可）
    :param cond_fn: 无需了解：条件生成函数（本项目不使用）
    :param model_kwargs: 无需了解：额外的模型参数（本项目不使用）
    :param eta: 无需了解：DDIM 的随机性参数（eta=0 时完全确定性，默认即可）
    :return: 生成的动作 x_0，形状与 shape 相同
    
    调用位置：在 nets.py 的 DiffusionPolicy.ddim_sample 中被调用（第 213 行）
    用途：生成时用 DDIM 方式从噪声逐步去噪得到动作（比 DDPM 快，可以跳步）
    """
    # ===== 第一步：从纯噪声开始 =====
    rng_key, sample_key = jax.random.split(rng_key)
    x = jax.random.normal(sample_key, shape)  # x_T ~ N(0,I)

    # ===== 第二步：循环 T 步去噪（DDIM 方式） =====
    # 注意：循环结构看起来和 p_sample_loop 一样，但核心区别在于每步调用的采样函数不同：
    # - p_sample_loop 调用 p_sample（第 748 行）：x_{t-1} = μ + σ·ε（随机采样，必须完整 T 步）
    # - ddim_sample_loop 调用 ddim_sample（第 911 行）：x_{t-1} = mean_pred + sigma*noise（当 eta=0 时确定性，可以跳步）
    # 
    # 虽然这里循环还是完整的 T 步，但 ddim_sample 内部使用的是 DDIM 的确定性公式（第 817-825 行），
    # 与 p_sample 的随机采样公式（第 684-685 行）完全不同。
    # 
    # 如果要实现跳步加速，可以修改 indices，例如：indices = list(range(0, self.num_timesteps, 10))[::-1]
    # 这样每 10 步采样一次，更快但可能精度略低。
    indices = list(range(self.num_timesteps))[::-1]
    for i in indices:
      # 当前时间步 t = i
      t = np.ones(shape[:-1], dtype=np.int32) * i
      
      # 网络预测噪声：ε_pred = model_forward(x_t, t)
      model_ouput = model_forward(x, self._scale_timesteps(t))
      
      # 用 DDIM 方式采样 x_{t-1}（与 DDPM 的 p_sample 不同）
      # ddim_sample 的详细实现在第 759 行，使用 DDIM 的确定性公式（第 817-825 行）
      # 核心区别：DDIM 的采样公式是 mean_pred + sigma*noise，当 eta=0 时 sigma=0（完全确定性）
      # 而 p_sample 的采样公式是 μ + σ·ε，每步都有随机性
      rng_key, sample_key = jax.random.split(rng_key)
      out = self.ddim_sample(
        sample_key, model_ouput, x, t, clip_denoised, cond_fn, model_kwargs,
        eta
      )
      
      # 更新 x：x = x_{t-1}，为下一步做准备
      x = out["sample"]
    
    # ===== 第三步：返回最终动作 =====
    # 循环结束后，x 就是 x_0（去噪完成，得到动作）
    return x

  def _vb_terms_bpd(self, model_ouput, x_start, x_t, t, clip_denoised=True):
    """
    Get a term for the variational lower-bound.

    The resulting units are bits (rather than nats, as one might expect).
    This allows for comparison to other papers.

    :return: a dict with the following keys:
             - 'output': a shape [N] tensor of NLLs or KLs.
             - 'pred_xstart': the x_0 predictions.
    """
    true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
      x_start=x_start, x_t=x_t, t=t
    )
    out = self.p_mean_variance(model_ouput, x_t, t, clip_denoised)
    kl = normal_kl(
      true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
    )
    kl = mean_flat(kl) / np.log(2.0)

    decoder_nll = -discretized_gaussian_log_likelihood(
      x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
    )
    assert decoder_nll.shape == x_start.shape
    decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

    # At the first timestep return the decoder NLL,
    # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
    output = np.where((t == 0), decoder_nll, kl)
    return {"output": output, "pred_xstart": out["pred_xstart"]}

  def training_losses(self, rng_key, model_forward, x_start, t):
    """
    计算扩散模型的训练损失（ELBO → MSE）。
    
    这是扩散模型训练的核心函数，实现了 DDPM 的训练目标：
    - 前向加噪：对真实数据 x_0 加噪得到 x_t
    - 网络预测：让网络预测噪声（或 x_0，或 x_{t-1}）
    - 计算损失：MSE loss（网络预测 vs 真实目标）
    
    数学原理：
    - ELBO（Evidence Lower Bound）可以简化为 MSE loss
    - 当 model_mean_type = EPSILON 时，target = noise，loss = ||noise - model_output||²
    - 当 model_mean_type = START_X 时，target = x_0，loss = ||x_0 - model_output||²
    - 当 model_mean_type = PREVIOUS_X 时，target = x_{t-1}，loss = ||x_{t-1} - model_output||²
    
    :param rng_key: JAX 随机数生成器密钥
    :param model_forward: 网络前向传播函数，输入 (x_t, t)，输出预测值
    :param x_start: 干净输入 x_0，形状 [N x C x ...]（N 是 batch size，C 是特征维度）
                     在扩散策略中，x_0 是真实动作（来自数据集）
    :param t: 时间步索引数组，形状 [N]，每个元素表示对应样本的时间步（0 表示第 1 步）
    :return: 返回一个结果集合（Python字典类型），包含以下内容：
             - 'loss': 训练损失，形状 [N]（每个样本一个损失值）
             - 'model_output': 网络输出
             - 'x_t': 加噪后的数据 x_t
             - 'ts_weights': 时间步权重（用于加权损失）
             - 'mse': MSE 损失（如果使用 MSE loss）
             - 'vb': 变分下界项（如果学习方差）
             
             说明：字典（dict）是 Python 的一种数据结构，可以存储多个"名字-值"对
             例如：terms = {"loss": 0.5, "mse": 0.3} 表示 loss=0.5, mse=0.3
             可以通过 terms["loss"] 访问 loss 的值
    
    调用位置：在 nets.py 的 DiffusionPolicy.loss 中被调用（第 210 行）
    用途：训练扩散策略网络，让网络学会预测噪声（或 x_0）
    """
    # ===== 第一步：前向加噪 =====
    # JAX 说明：JAX 是 Google 开发的科学计算库，类似于 NumPy，但支持：
    # - 自动微分（用于神经网络训练）
    # - GPU/TPU 加速（比 NumPy 快很多）
    # - 函数式编程（更适合并行计算）
    # 在这个项目中，jax.random.normal 用于生成随机数（类似 numpy.random.normal）
    # 采样标准高斯噪声 ε ~ N(0,I)
    noise = jax.random.normal(rng_key, x_start.shape, dtype=x_start.dtype)
    # 对真实数据 x_0 加噪得到 x_t：x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
    x_t = self.q_sample(x_start, t, noise=noise)
    
    # ===== 第二步：网络预测 =====
    # 网络输入：加噪后的数据 x_t 和时间步 t
    # 网络输出：根据 model_mean_type 预测噪声 ε（或 x_0，或 x_{t-1}）
    model_output = model_forward(x_t, self._scale_timesteps(t))

    # ===== 第三步：准备返回的结果集合（Python字典类型） =====
    # 字典（dict）：Python中的一种数据结构，可以存储多个"键-值"对
    # 例如：{"name": "张三", "age": 20} 表示 name=张三, age=20
    # 这里 terms 用来存储训练过程中需要的各种中间结果和最终损失
    # 可以理解为：一个"容器"，里面装了多个"变量"，每个变量都有一个"名字"（键）
    terms = {"model_output": model_output, "x_t": x_t}
    # 提取时间步权重：用于加权不同时间步的损失（让不同时间步的 loss 贡献更合理）
    # terms["ts_weights"] 表示在 terms 这个"容器"中添加一个名为 "ts_weights" 的变量
    terms["ts_weights"] = _extract_into_tensor(
      self.normalized_ts_weights, t, x_start.shape[:-1]
    )

    # ===== 第四步：根据 loss_type 计算损失 =====
    if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
      # KL 散度损失：使用完整的变分下界（ELBO）
      # 本项目不使用此选项，使用 MSE loss
      terms["loss"] = self._vb_terms_bpd(
        model_output, x_start=x_start, x_t=x_t, t=t, clip_denoised=False
      )["output"]
      if self.loss_type == LossType.RESCALED_KL:
        terms["loss"] *= self.num_timesteps
    elif self.loss_type in (LossType.MSE, LossType.RESCALED_MSE):
      # MSE 损失：ELBO 的简化形式（本项目默认使用此选项）
      
      # ===== 4.1：如果学习方差，需要额外计算变分下界项 =====
      if self.model_var_type in (
        ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE
      ):
        # 学习方差：网络同时输出均值和方差
        # 此时 model_output 的形状是 [B, C*2, ...]，前一半是均值，后一半是方差
        B, C = x_t.shape[:2]
        assert model_output.shape == (B, C * 2, *x_t.shape[2:])
        # 拆分：前一半是均值预测，后一半是方差预测
        model_output, model_var_values = np.split(model_output, C, axis=1)
        # 冻结均值部分（detach），只让方差部分参与变分下界计算
        # 这样可以让方差学习不影响均值预测
        frozen_out = np.concatenate(
          [model_output.detach(), model_var_values], axis=1
        )
        # 计算变分下界项（用于学习方差）
        terms["vb"] = self._vb_terms_bpd(
          frozen_out, x_start=x_start, x_t=x_t, t=t, clip_denoised=False
        )["output"]
        if self.loss_type == LossType.RESCALED_MSE:
          # 缩放变分下界项：除以 1000 以平衡 MSE 项和 VB 项
          # 如果不缩放，VB 项会干扰 MSE 项的学习
          terms["vb"] *= self.num_timesteps / 1000.0
      # 注意：本项目使用固定方差（FIXED_SMALL），所以不会进入这个分支

      # ===== 4.2：根据 model_mean_type 选择目标值 =====
      # 网络预测的目标取决于 model_mean_type：
      target = {
        ModelMeanType.PREVIOUS_X:
          # 网络预测 x_{t-1}，目标是从后验分布得到的真实 x_{t-1}
          self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
        ModelMeanType.START_X:
          # 网络预测 x_0，目标是真实的 x_0
          x_start,
        ModelMeanType.EPSILON:
          # 网络预测噪声 ε（本项目默认），目标是真实的噪声
          noise,
      }[self.model_mean_type]
      
      # 确保形状一致
      assert model_output.shape == target.shape == x_start.shape
      
      # ===== 4.3：计算 MSE 损失 =====
      # mean_flat：计算除了 batch 维度之外所有维度的平均值
      # 结果形状：[N]，每个样本一个损失值
      terms["mse"] = mean_flat((target - model_output)**2)
      
      # ===== 4.4：组合最终损失 =====
      # 如果学习方差，损失 = MSE + VB（变分下界项）
      # 如果固定方差（本项目），损失 = MSE
      if "vb" in terms:
        terms["loss"] = terms["mse"] + terms["vb"]
      else:
        terms["loss"] = terms["mse"]
    else:
      raise NotImplementedError(self.loss_type)

    return terms

  def training_losses_(self, rng_key, model_forward, x_start, t):
    noise = jax.random.normal(rng_key, x_start.shape, dtype=x_start.dtype)
    x_t = self.q_sample(x_start, t, noise=noise)

    model_output = model_forward(x_t, self._scale_timesteps(t))

    target = {
      ModelMeanType.START_X: x_start,
      ModelMeanType.EPSILON: noise,
    }[self.model_mean_type]
    mse = np.square(target - model_output)
    return {"loss": mse, "model_output": model_output}

  def _prior_bpd(self, x_start):
    """
    Get the prior KL term for the variational lower-bound, measured in
    bits-per-dim.

    This term can't be optimized, as it only depends on the encoder.

    :param x_start: the [N x C x ...] tensor of inputs.
    :return: a batch of [N] KL values (in bits), one per batch element.
    """
    batch_size = x_start.shape[0]
    t = np.array([self.num_timesteps - 1] * batch_size)
    qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
    kl_prior = normal_kl(
      mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
    )
    return mean_flat(kl_prior) / np.log(2.0)


def normal_kl(mean1, logvar1, mean2, logvar2):
  """
  Compute the KL divergence between two gaussians.
  Shapes are automatically broadcasted, so batches can be compared to
  scalars, among other use cases.
  """
  tensor = None
  for obj in (mean1, logvar1, mean2, logvar2):
    if isinstance(obj, np.ndarray):
      tensor = obj
      break
  assert tensor is not None, "at least one argument must be a Tensor"

  # Force variances to be Tensors. Broadcasting helps convert scalars to
  # Tensors, but it does not work for th.exp().
  logvar1, logvar2 = [
    x if isinstance(x, np.ndarray) else np.array(x)
    for x in (logvar1, logvar2)
  ]

  return 0.5 * (
    -1.0 + logvar2 - logvar1 + np.exp(logvar1 - logvar2) +
    ((mean1 - mean2)**2) * np.exp(-logvar2)
  )


def approx_standard_normal_cdf(x):
  """
  A fast approximation of the cumulative distribution function of the
  standard normal.
  """
  return 0.5 * (
    1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3)))
  )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
  """
  Compute the log-likelihood of a Gaussian distribution discretizing to a
  given image.
  :param x: the target images. It is assumed that this was uint8 values,
            rescaled to the range [-1, 1].
  :param means: the Gaussian mean Tensor.
  :param log_scales: the Gaussian log stddev Tensor.
  :return: a tensor like x of log probabilities (in nats).
  """
  assert x.shape == means.shape == log_scales.shape
  centered_x = x - means
  inv_stdv = np.exp(-log_scales)
  plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
  cdf_plus = approx_standard_normal_cdf(plus_in)
  min_in = inv_stdv * (centered_x - 1.0 / 255.0)
  cdf_min = approx_standard_normal_cdf(min_in)
  log_cdf_plus = np.log(cdf_plus.clip(a_min=1e-12))
  log_one_minus_cdf_min = np.log((1.0 - cdf_min).clip(a_min=1e-12))
  cdf_delta = cdf_plus - cdf_min
  log_probs = np.where(
    x < -0.999,
    log_cdf_plus,
    np.where(
      x > 0.999, log_one_minus_cdf_min, np.log(cdf_delta.clip(a_min=1e-12))
    ),
  )
  assert log_probs.shape == x.shape
  return log_probs


def _extract_into_tensor(arr, timesteps, broadcast_shape):
  """
  Extract values from a 1-D numpy array for a batch of indices.

  :param arr: the 1-D numpy array.
  :param timesteps: a tensor of indices into the array to extract.
  :param broadcast_shape: a larger shape of K dimensions with the batch
                          dimension equal to the length of timesteps.
  :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
  """
  res = arr[timesteps].astype(np.float32)
  while len(res.shape) < len(broadcast_shape):
    res = res[..., None]
  return np.broadcast_to(res, broadcast_shape)
