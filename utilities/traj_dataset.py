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

"""Utils to generate trajectory based dataset with multi-step reward."""

import collections

import gym
import numpy as np
from tqdm import tqdm

Batch = collections.namedtuple(
  "Batch",
  ["observations", "actions", "rewards", "masks", "next_observations"]
)


def split_into_trajectories(
  observations, actions, rewards, masks, dones_float, next_observations
):
  # 注意：下面这一段就是「先按 trajectory 切分」：
  # 把原始按时间拼接的一长串 (s, a, r, ..., done_float) 序列，
  # 按 dones_float[i] == 1.0 的位置切开，拆成多条 episode 级别的 trajectories。
  trajs = [[]]

  for i in tqdm(range(len(observations))):
    trajs[-1].append(
      (
        observations[i],
        actions[i],
        rewards[i],
        masks[i],
        dones_float[i],
        next_observations[i],
      )
    )
    if dones_float[i] == 1.0 and i + 1 < len(observations):
      trajs.append([])

  return trajs


class Dataset(object):

  def __init__(
    self,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    masks: np.ndarray,
    dones_float: np.ndarray,
    next_observations: np.ndarray,
    size: int,
  ):

    self.observations = observations
    self.actions = actions
    self.rewards = rewards
    self.masks = masks
    self.dones_float = dones_float
    self.next_observations = next_observations
    self.size = size

  def sample(self, batch_size: int) -> Batch:
    indx = np.random.randint(self.size, size=batch_size)
    return Batch(
      observations=self.observations[indx],
      actions=self.actions[indx],
      rewards=self.rewards[indx],
      masks=self.masks[indx],
      next_observations=self.next_observations[indx],
    )


class D4RLDataset(Dataset):

  def __init__(
    self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5
  ):
    import d4rl

    self.raw_dataset = dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
      lim = 1 - eps
      dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    dones_float = np.zeros_like(dataset["rewards"])

    for i in range(len(dones_float) - 1):
      if (
        np.linalg.
        norm(dataset["observations"][i + 1] - dataset["next_observations"][i])
        > 1e-6 or dataset["terminals"][i] == 1.0
      ):
        dones_float[i] = 1
      else:
        dones_float[i] = 0

    dones_float[-1] = 1

    super().__init__(
      dataset["observations"].astype(np.float32),
      actions=dataset["actions"].astype(np.float32),
      rewards=dataset["rewards"].astype(np.float32),
      masks=1.0 - dataset["terminals"].astype(np.float32),
      dones_float=dones_float.astype(np.float32),
      next_observations=dataset["next_observations"].astype(np.float32),
      size=len(dataset["observations"]),
    )


def compute_returns(traj):
  episode_return = 0
  for _, _, rew, _, _, _ in traj:
    episode_return += rew

  return episode_return


def get_traj_dataset(env, sorting=True, norm_reward=False):
  """从 D4RL env 拿到原始数据后，切成多条 trajectory，并做可选的排序/归一化。

  调用关系：
  - 这里是 get_nstep_dataset 的「前置步骤」：get_nstep_dataset 会先调用它拿到 trajs，
    再在每条 traj 上做 n-step 卷积；然后 get_d4rl_dataset 又在 get_nstep_dataset 之上做一层改装。
  - Trainer 最终只看 get_d4rl_dataset 返回的 dict，这里属于底层数据预处理。
  """
  # env 既可以是字符串（env 名），也可以是 gym.Env，这里统一成 env 实例。
  env = gym.make(env) if isinstance(env, str) else env
  # 用 D4RLDataset 封装 d4rl.qlearning_dataset(env)，得到一条长序列形式的原始数据。
  dataset = D4RLDataset(env)
  # 用前面写的 split_into_trajectories 按 dones_float 把长序列切成多条 traj。
  trajs = split_into_trajectories(
    dataset.observations,
    dataset.actions,
    dataset.rewards,
    dataset.masks,
    dataset.dones_float,
    dataset.next_observations,
  )
  # 若 sorting=True，则按整条 traj 的 return 排序（高/低回报在前/后），
  # 主要是数据组织习惯，对训练理解不是关键，可以先不深究为什么要排序。
  if sorting:
    trajs.sort(key=compute_returns)

  # 若 norm_reward=True，则按「轨迹 return 的极差」做一个简单归一化，
  # 把不同任务的 reward 尺度 roughly scale 到类似范围；细节可暂时忽略。
  if norm_reward:
    returns = [compute_returns(traj) for traj in trajs]
    norm = (max(returns) - min(returns)) / 1000
    for traj in tqdm(trajs):
      for i, ts in enumerate(traj):
        # ts 结构是 (obs, action, reward, mask, done_float, next_obs)
        # 这里只把第 3 个元素 reward 除以 norm，其它保持不变。
        traj[i] = ts[:2] + (ts[2] / norm,) + ts[3:]

  # 返回两部分：
  # - trajs：已按需要切好/排好/归一化的轨迹列表，供 get_nstep_dataset 做 n-step。
  # - raw_dataset：原始 d4rl.qlearning_dataset(env) 的 dict 版本（未排序），
  #   仅用于做一些一致性校验（长度/shape 等），训练过程一般不会直接用到。
  # NOTE: this raw_dataset is not sorted
  return trajs, dataset.raw_dataset


def nstep_reward_prefix(rewards, nstep=5, gamma=0.9):
  # 这里是一个独立的 n-step 前缀和实现：
  # 给定单条轨迹的 reward 序列 rewards[0:L]，和长度为 n 的折扣权重 [1, γ, ..., γ^{n-1}]，
  # 用一维卷积 np.convolve 实现
  #   R_t^{(n)} = sum_{i=0}^{n-1} γ^i * rewards[t+i]
  # 的「窗口滑动加权和」，并通过 [nstep-1:] 做好对齐与截断。
  gammas = np.array([gamma**i for i in range(nstep)])
  nstep_rewards = np.convolve(rewards, gammas)[nstep - 1:]
  return nstep_rewards


def get_nstep_dataset_from_trajs(trajs, nstep, gamma, raw_transition_count):
  """对已由 split_into_trajectories 得到的轨迹列表做 n-step 回报与 next_obs 对齐。"""
  gammas = np.array([gamma**i for i in range(nstep)])
  obss, acts, terms, next_obss, nstep_rews, dones_float = [], [], [], [], [], []
  for traj in trajs:
    L = len(traj)
    rewards = np.array([ts[2] for ts in traj])
    cum_rewards = np.convolve(rewards, gammas)[nstep - 1:]
    nstep_rews.append(cum_rewards)
    next_obss.extend([traj[min(i + nstep - 1, L - 1)][-1] for i in range(L)])
    obss.extend([traj[i][0] for i in range(L)])
    acts.extend([traj[i][1] for i in range(L)])
    terms.extend([bool(1 - traj[i][3]) for i in range(L)])
    dones_float.extend(traj[i][4] for i in range(L))

  dataset = {}
  dataset["observations"] = np.stack(obss)
  dataset["actions"] = np.stack(acts)
  dataset["next_observations"] = np.stack(next_obss)
  dataset["rewards"] = np.concatenate(nstep_rews)
  dataset["terminals"] = np.stack(terms)
  dataset["dones_float"] = np.stack(dones_float)

  assert len(dataset["rewards"]) == raw_transition_count
  assert dataset["next_observations"].shape[0] == raw_transition_count

  return dataset


def get_nstep_dataset(
  env, nstep=5, gamma=0.9, sorting=True, norm_reward=False
):
  trajs, raw_dataset = get_traj_dataset(env, sorting, norm_reward)
  raw_count = len(raw_dataset["rewards"])
  dataset = get_nstep_dataset_from_trajs(trajs, nstep, gamma, raw_count)
  assert dataset["next_observations"].shape == raw_dataset["next_observations"].shape
  return dataset
