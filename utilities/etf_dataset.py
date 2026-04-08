"""从 ETF CSV + 行为策略构造与 get_d4rl_dataset 一致的离线字典。"""

from __future__ import annotations

from typing import Optional

import numpy as np

from trading_env.behavior_policy import MixedBehaviorPolicy
from trading_env.env import ETFTradingEnv

from utilities.traj_dataset import (
    compute_returns,
    get_nstep_dataset_from_trajs,
    split_into_trajectories,
)


def collect_etf_transitions(
    csv_path: str,
    behavior_seed: int = 0,
    policy: Optional[MixedBehaviorPolicy] = None,
):
    """用行为策略在 ETFTradingEnv 上 rollout，得到与 D4RL 风格一致的 transition 序列。

    Returns
    -------
    dict with keys: observations, actions, rewards, terminals, masks,
                    dones_float, next_observations (each numpy, length N)
    """
    env = ETFTradingEnv(csv_path)
    if policy is None:
        policy = MixedBehaviorPolicy(seed=behavior_seed)

    obs_list, act_list, rew_list, next_obs_list = [], [], [], []
    obs = env.reset()
    done = False

    while not done:
        t = env.t
        row = env.df.iloc[t]
        prev_r = MixedBehaviorPolicy.prev_return_from_df(env.df, t)
        a = policy.act(row, prev_r)
        action_vec = np.array([a], dtype=np.float32)

        next_obs, r, done, _ = env.step(a)

        obs_list.append(np.asarray(obs, dtype=np.float32))
        act_list.append(action_vec)
        rew_list.append(np.float32(r))
        next_obs_list.append(np.asarray(next_obs, dtype=np.float32))
        obs = next_obs

    observations = np.stack(obs_list, axis=0)
    actions = np.stack(act_list, axis=0)
    rewards = np.asarray(rew_list, dtype=np.float32)
    next_observations = np.stack(next_obs_list, axis=0)
    n = observations.shape[0]

    terminals = np.zeros(n, dtype=np.float32)
    terminals[-1] = 1.0
    masks = 1.0 - terminals

    dones_float = np.zeros(n, dtype=np.float32)
    for i in range(n - 1):
        gap = np.linalg.norm(observations[i + 1] - next_observations[i])
        if gap > 1e-6 or terminals[i] == 1.0:
            dones_float[i] = 1.0
        else:
            dones_float[i] = 0.0
    dones_float[-1] = 1.0

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "masks": masks,
        "dones_float": dones_float,
        "next_observations": next_observations,
    }


def transitions_to_trajs(raw: dict, sorting: bool, norm_reward: bool):
    """将 collect_etf_transitions 输出转为 trajectory 列表，供 n-step 使用。"""
    trajs = split_into_trajectories(
        raw["observations"],
        raw["actions"],
        raw["rewards"],
        raw["masks"],
        raw["dones_float"],
        raw["next_observations"],
    )
    if sorting:
        trajs.sort(key=compute_returns)
    if norm_reward:
        returns = [compute_returns(traj) for traj in trajs]
        span = max(returns) - min(returns)
        norm = span / 1000.0 if span > 1e-12 else 1.0
        for traj in trajs:
            for i, ts in enumerate(traj):
                traj[i] = ts[:2] + (ts[2] / norm,) + ts[3:]
    return trajs


def build_etf_nstep_dataset(
    csv_path: str,
    nstep: int,
    gamma: float,
    sorting: bool = True,
    norm_reward: bool = False,
    behavior_seed: int = 0,
    policy: Optional[MixedBehaviorPolicy] = None,
):
    """返回与 get_nstep_dataset 相同键的字典。"""
    raw = collect_etf_transitions(csv_path, behavior_seed=behavior_seed, policy=policy)
    raw_count = raw["observations"].shape[0]
    trajs = transitions_to_trajs(raw, sorting=sorting, norm_reward=norm_reward)
    return get_nstep_dataset_from_trajs(trajs, nstep, gamma, raw_count)
