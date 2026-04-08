"""多标的组合环境：动作经 softmax 得到 long-only 权重，奖励与论文 §5.1.1 一致。"""

from __future__ import annotations

import numpy as np


def logits_to_weights(logits: np.ndarray, temperature: float = 3.0) -> np.ndarray:
    """将 [-1,1] 上的 logits 映射到单纯形（用于 long-only 组合）。"""
    x = np.asarray(logits, dtype=np.float64) * float(temperature)
    x = x - np.max(x)
    e = np.exp(x)
    w = e / np.sum(e)
    return w.astype(np.float32)


class PortfolioTradingEnv:
    """
    多标的日频组合仿真。

    - 动作：各标的上的 raw logits，训练时 clip 到 [-1,1]，环境内乘温度后 softmax 得 w_t。
    - 奖励：r_t = w_{t-1}^T r_t - fee_rate * ||w_t - w_{t-1}||_1（与文档式子一致，L1 为各分量绝对值之和）。
    - 观测：concat([上一日各标的收益 r_{t-1}, 当前持仓权重 w_{t-1}])，维数 2N；首步 r_{t-1}=0，w 为等权。
    """

    def __init__(
        self,
        returns: np.ndarray,
        fee_rate: float = 0.001,
        initial_cash: float = 1.0,
        softmax_temp: float = 3.0,
    ):
        self.returns = np.asarray(returns, dtype=np.float32)
        if self.returns.ndim != 2:
            raise ValueError("returns must have shape (T, N).")
        self.n_assets = int(self.returns.shape[1])
        self.n_steps = int(self.returns.shape[0])
        self.fee_rate = float(fee_rate)
        self.initial_cash = float(initial_cash)
        self.softmax_temp = float(softmax_temp)

        self.t = 0
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.prev_day_returns = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = self.initial_cash
        self.done = False

    def reset(self):
        self.t = 0
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.prev_day_returns = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = self.initial_cash
        self.done = False
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.prev_day_returns, self.weights], axis=0).astype(
            np.float32
        )

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before next step().")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.n_assets:
            raise ValueError(
                f"action dim {action.shape[0]} != n_assets {self.n_assets}"
            )
        action = np.clip(action, -1.0, 1.0)
        w_new = logits_to_weights(action, temperature=self.softmax_temp)

        r_vec = self.returns[self.t]
        pnl = float(np.dot(self.weights, r_vec))
        turnover = float(np.sum(np.abs(w_new - self.weights)))
        reward = pnl - self.fee_rate * turnover

        self.portfolio_value *= 1.0 + reward
        self.weights = w_new
        self.prev_day_returns = r_vec.copy()
        self.t += 1

        if self.t >= self.n_steps - 1:
            self.done = True

        next_obs = (
            self._get_obs()
            if not self.done
            else np.zeros(2 * self.n_assets, dtype=np.float32)
        )
        info = {
            "pnl": pnl,
            "turnover": turnover,
            "portfolio_value": self.portfolio_value,
            "weights": self.weights.copy(),
        }
        return next_obs, float(reward), self.done, info
