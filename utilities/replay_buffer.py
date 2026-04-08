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

"""Replay buffer and utils to retrieve the dataset."""

import numpy as np

from utilities.traj_dataset import get_nstep_dataset


class ReplayBuffer(object):

  def __init__(self, max_size, data=None):
    self._max_size = max_size
    self._next_idx = 0
    self._size = 0
    self._initialized = False
    self._total_steps = 0

    if data is not None:
      if self._max_size < data["observations"].shape[0]:
        self._max_size = data["observations"].shape[0]
      self.add_batch(data)

  def __len__(self):
    return self._size

  def _init_storage(self, observation_dim, action_dim):
    self._observation_dim = observation_dim
    self._action_dim = action_dim
    self._observations = np.zeros(
      (self._max_size, observation_dim), dtype=np.float32
    )
    self._next_observations = np.zeros(
      (self._max_size, observation_dim), dtype=np.float32
    )
    self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
    self._rewards = np.zeros(self._max_size, dtype=np.float32)
    self._dones = np.zeros(self._max_size, dtype=np.float32)
    self._next_idx = 0
    self._size = 0
    self._initialized = True

  def add_sample(self, observation, action, reward, next_observation, done):
    if not self._initialized:
      self._init_storage(observation.size, action.size)

    self._observations[self._next_idx, :] = np.array(
      observation, dtype=np.float32
    )
    self._next_observations[self._next_idx, :] = np.array(
      next_observation, dtype=np.float32
    )
    self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
    self._rewards[self._next_idx] = reward
    self._dones[self._next_idx] = float(done)

    if self._size < self._max_size:
      self._size += 1
    self._next_idx = (self._next_idx + 1) % self._max_size
    self._total_steps += 1

  def add_traj(self, observations, actions, rewards, next_observations, dones):
    for o, a, r, no, d in zip(
      observations, actions, rewards, next_observations, dones
    ):
      self.add_sample(o, a, r, no, d)

  def add_batch(self, batch):
    self.add_traj(
      batch["observations"],
      batch["actions"],
      batch["rewards"],
      batch["next_observations"],
      batch["dones"],
    )

  def sample(self, batch_size):
    indices = np.random.randint(len(self), size=batch_size)
    return self.select(indices)

  def select(self, indices):
    return dict(
      observations=self._observations[indices, ...],
      actions=self._actions[indices, ...],
      rewards=self._rewards[indices, ...],
      next_observations=self._next_observations[indices, ...],
      dones=self._dones[indices, ...],
    )

  def generator(self, batch_size, n_batchs=None):
    i = 0
    while n_batchs is None or i < n_batchs:
      yield self.sample(batch_size)
      i += 1

  @property
  def total_steps(self):
    return self._total_steps

  @property
  def data(self):
    return dict(
      observations=self._observations[:self._size, ...],
      actions=self._actions[:self._size, ...],
      rewards=self._rewards[:self._size, ...],
      next_observations=self._next_observations[:self._size, ...],
      dones=self._dones[:self._size, ...],
    )


def get_d4rl_dataset(env, nstep=1, gamma=0.9, norm_reward=False):
  """High-level helper: 从 D4RL env 构造一个训练用的 dict。

  你需要知道的点：
  - 这是 Trainer 调数据的唯一入口之一：_setup_d4rl() 里会调用这个函数。
  - 它内部会：
    1）调用 get_nstep_dataset 做 n-step 回报与按轨迹排序；
    2）返回一个键固定的 dict（obs / actions / next_obs / rewards / dones...），
       方便后面直接塞进 data.Dataset 再 sample。

  不需要纠结的点（可以先略过）：
  - sorting=True：get_nstep_dataset 里会按轨迹 return 排序，更多是数据组织细节；
  - dones_float：主要用于按 trajectory 切分、做 n-step 时的边界处理，训练 loss 本身并不会直接用到它。
  """
  # 之前可能支持 nstep==1 直接用 d4rl.qlearning_dataset，这里统一走 n-step 路径，
  # 并且 sorting=True：始终按轨迹 return 排好序再展开成 transition 级别的数据。
  dataset = get_nstep_dataset(
    env, nstep, gamma, sorting=True, norm_reward=norm_reward
  )

  # 将 get_nstep_dataset 返回的内部字段，整理成后续代码统一使用的键名。
  # 这些键就是后面 batch 里会出现的字段名。
  return dict(
    observations=dataset["observations"],
    actions=dataset["actions"],
    next_observations=dataset["next_observations"],
    rewards=dataset["rewards"],
    # terminals -> dones：转成 float32，供 TD 目标里 (1-done)*gamma 使用。
    dones=dataset["terminals"].astype(np.float32),
    # dones_float 只保留更细粒度的「episode 结束」信息，供 n-step / 轨迹逻辑使用。
    dones_float=dataset["dones_float"],
  )


def get_etf_dataset(
    csv_path,
    nstep,
    gamma,
    norm_reward=False,
    behavior_seed=0,
    policy=None,
):
  """与 get_d4rl_dataset 相同输出键，数据来自 ETF CSV + 行为策略 rollout。"""
  from utilities.etf_dataset import build_etf_nstep_dataset

  dataset = build_etf_nstep_dataset(
      csv_path,
      nstep=nstep,
      gamma=gamma,
      sorting=True,
      norm_reward=norm_reward,
      behavior_seed=behavior_seed,
      policy=policy,
  )
  return dict(
      observations=dataset["observations"],
      actions=dataset["actions"],
      next_observations=dataset["next_observations"],
      rewards=dataset["rewards"],
      dones=dataset["terminals"].astype(np.float32),
      dones_float=dataset["dones_float"],
  )


def get_portfolio_dataset(
    csv_path,
    nstep,
    gamma,
    norm_reward=False,
    behavior_seed=0,
    policy=None,
):
  """与 get_d4rl_dataset 相同输出键，数据来自面板收益率 CSV + 行为策略 rollout。"""
  from trading_env.panel_loader import load_returns_panel_csv
  from utilities.portfolio_dataset import build_portfolio_nstep_dataset

  returns, _ = load_returns_panel_csv(csv_path)
  dataset = build_portfolio_nstep_dataset(
      returns,
      nstep=nstep,
      gamma=gamma,
      sorting=True,
      norm_reward=norm_reward,
      behavior_seed=behavior_seed,
      policy=policy,
  )
  return dict(
      observations=dataset["observations"],
      actions=dataset["actions"],
      next_observations=dataset["next_observations"],
      rewards=dataset["rewards"],
      dones=dataset["terminals"].astype(np.float32),
      dones_float=dataset["dones_float"],
  )


def index_batch(batch, indices):
  indexed = {}
  for key in batch.keys():
    indexed[key] = batch[key][indices, ...]
  return indexed


def parition_batch_train_test(batch, train_ratio):
  train_indices = np.random.rand(batch["observations"].shape[0]) < train_ratio
  train_batch = index_batch(batch, train_indices)
  test_batch = index_batch(batch, ~train_indices)
  return train_batch, test_batch


def subsample_batch(batch, size):
  indices = np.random.randint(batch["observations"].shape[0], size=size)
  return index_batch(batch, indices)


def concatenate_batches(batches):
  concatenated = {}
  for key in batches[0].keys():
    concatenated[key] = np.concatenate(
      [batch[key] for batch in batches], axis=0
    ).astype(np.float32)
  return concatenated


def split_batch(batch, batch_size):
  batches = []
  length = batch["observations"].shape[0]
  keys = batch.keys()
  for start in range(0, length, batch_size):
    end = min(start + batch_size, length)
    batches.append({key: batch[key][start:end, ...] for key in keys})
  return batches


def split_data_by_traj(data, max_traj_length):
  dones = data["dones"].astype(bool)
  start = 0
  splits = []
  for i, done in enumerate(dones):
    if i - start + 1 >= max_traj_length or done:
      splits.append(index_batch(data, slice(start, i + 1)))
      start = i + 1

  if start < len(dones):
    splits.append(index_batch(data, slice(start, None)))

  return splits
