"""Gym.Env wrapper for ETFTradingEnv (observation_space / action_space for Trainer)."""

import numpy as np
import gym
from gym import spaces

from .env import ETFTradingEnv
from .panel_loader import load_returns_panel_csv
from .portfolio_env import PortfolioTradingEnv


class ETFGymEnv(gym.Env):
    """单标的 ETF 仿真；评估与 TrajSampler 兼容。"""

    metadata = {"render.modes": []}

    def __init__(self, csv_path: str, fee_rate: float = 0.001, initial_cash: float = 1.0):
        super().__init__()
        self._env = ETFTradingEnv(csv_path, fee_rate=fee_rate, initial_cash=initial_cash)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self):
        return np.asarray(self._env.reset(), dtype=np.float32)

    def step(self, action):
        a = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        obs, reward, done, info = self._env.step(a)
        obs = np.asarray(obs, dtype=np.float32)
        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        pass

    @staticmethod
    def get_normalized_score(episode_return_sum: float) -> float:
        """非 D4RL 环境无参考分；返回原始累计收益供日志使用。"""
        return float(episode_return_sum)


class PortfolioGymEnv(gym.Env):
    """多标的组合仿真；评估与 TrajSampler 兼容。"""

    metadata = {"render.modes": []}

    def __init__(
        self,
        csv_path: str,
        fee_rate: float = 0.001,
        initial_cash: float = 1.0,
        softmax_temp: float = 3.0,
    ):
        super().__init__()
        returns, _tickers = load_returns_panel_csv(csv_path)
        self._env = PortfolioTradingEnv(
            returns,
            fee_rate=fee_rate,
            initial_cash=initial_cash,
            softmax_temp=softmax_temp,
        )
        n = self._env.n_assets
        obs_dim = 2 * n
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n,), dtype=np.float32
        )

    def reset(self):
        return np.asarray(self._env.reset(), dtype=np.float32)

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        obs, reward, done, info = self._env.step(a)
        obs = np.asarray(obs, dtype=np.float32)
        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        pass

    @staticmethod
    def get_normalized_score(episode_return_sum: float) -> float:
        return float(episode_return_sum)
